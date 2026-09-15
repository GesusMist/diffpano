"""K runner: six fixed perspective-native trajectories, transient ERP RGB only."""
import argparse
import csv
import gc
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
from diffpano.camera import build_camera_sampler
from diffpano.config import load_experiment_config
from diffpano.diagnostics import tensor_to_pil
from diffpano.erp_local_consensus import ERPLocalCurrentStatePipeline, geometry_preflight, camera_digest, native_digest
from diffpano.initialization import set_random_seed,load_directional_prompts
from diffpano.metadata import save_run_metadata
from diffpano.native_multidiffusion import prepare_native_backend
from diffpano.pipelines import build_view_denoiser
from diffpano.trajectory import conditioning_digest
from diffpano.warp import StandardWarpOperator
from scripts.generate import _configure_denoiser
from scripts.current_state_experiment import check_runtime,read,write
from scripts.paired_rgb_endpoint import timed


def checked_config(spec):
    c=load_experiment_config(spec['config']);j=load_experiment_config(spec['j_config'])
    a,b=c.to_dict(),j.to_dict()
    for key in ('canvas','global_pipeline','view','erp','sampling'):a.pop(key);b.pop(key)
    if a!=b:raise AssertionError('K changes settings beyond its geometry')
    if c.model.pipeline!=spec['backend']:raise AssertionError('Wrong backend')
    if c.view.fov_x!=100. or c.view.fov_y!=100.:raise AssertionError('First K pass uses one fixed 100-degree recipe')
    return c


def geometry(spec,device=torch.device('cpu')):
    c=checked_config(spec)
    cameras=build_camera_sampler(c.sampling,c.view,c.experiment.seed).sample(0,c.generation.num_inference_steps)
    operator=StandardWarpOperator(c.warp,c.fusion)
    stats,counts=geometry_preflight(cameras,(c.erp.height,c.erp.width),operator,device)
    return c,cameras,operator,stats,counts


def geometry_only(specs):
    out=Path('outputs/vae-residual-controls/report/K/geometry');out.mkdir(parents=True,exist_ok=True)
    for backend,spec in specs.items():
        c,cameras,_,stats,counts=geometry(spec)
        torch.save(counts,out/(backend+'-contributors.pt'))
        tensor_to_pil((counts[0].expand(3,-1,-1)/counts.max())*2-1).save(out/(backend+'-contributors.png'))
        write(out/(backend+'.json'),dict(backend=backend,cameras=[asdict(v) for v in cameras],camera_sha256=camera_digest(cameras),
            erp_size=[c.erp.height,c.erp.width],view_size=[c.view.height,c.view.width],coverage=stats))
        print('Coverage',backend,stats,flush=True)


def run(spec):
    gate=read('outputs/vae-residual-controls/report/J/k_gate.json')
    if not gate['passed']:raise AssertionError('J gate has not passed')
    reference=read(Path(spec['j_output'])/'comparison.json')
    pre=read(Path('outputs/vae-residual-controls/report/K/geometry')/(spec['backend']+'.json'))
    if pre['coverage']['coverage_percent']!=100.:raise AssertionError('CPU coverage gate failed')
    folder=Path(spec['output']);folder.mkdir(parents=True,exist_ok=False)
    write(folder/'spec.json',spec)
    write(folder/'repository.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        dirty_status=subprocess.check_output(['git','status','--short'],text=True)))
    started=time.perf_counter();config=checked_config(spec)
    if torch.cuda.get_device_name()!=spec['gpu']:raise AssertionError('Use the paired A40 GPU type')
    set_random_seed(config.experiment.seed)
    backend=build_view_denoiser(config);_configure_denoiser(config,backend)
    prepared=prepare_native_backend(config,backend)
    if tuple(backend.rgb_spatial_shape_for_native(config.native_multidiffusion.patch_size,config.native_multidiffusion.patch_size))!=(config.view.height,config.view.width):
        raise AssertionError('K must retain the J local model resolution')
    conditioning=backend.conditioning_for_prompt_indices(prepared,[8],batch_size=config.generation.batch_size)
    if conditioning_digest(conditioning)!=reference['conditioning_sha256']:raise AssertionError('J/K conditioning mismatch')
    save_run_metadata(str(folder/'runtime_preflight.json'),config,backend,SimpleNamespace(steps=[]),'')
    baseline=read(Path(spec['j_output'])/'generation/metadata.json')
    keys=check_runtime(read(folder/'runtime_preflight.json'),baseline)
    _,cameras,operator,coverage,counts=geometry(spec,backend.device)
    if camera_digest(cameras)!=pre['camera_sha256'] or coverage!=pre['coverage']:raise AssertionError('CPU/GPU camera coverage differs')
    pipe=ERPLocalCurrentStatePipeline(backend=backend,cameras=cameras,erp_size=(config.erp.height,config.erp.width),
        warp_operator=operator,flow_transition=spec['backend']!='sd2')
    initial=pipe.initialize_local_states(config.experiment.seed,config.generation.batch_size)
    hashes=[native_digest(s) for s in initial];torch.save([s.cpu() for s in initial],folder/'initial_local_states.pt')
    write(folder/'initialization.json',dict(policy='one CPU Gaussian generator stream in canonical fixed-camera order',
        seed=config.experiment.seed,local_native_shapes=[list(s.shape) for s in initial],sha256_by_camera=hashes,
        camera_sha256=pipe.camera_sha256,j_k_bit_identical_initialization=False,erp_latent_field=False))
    (folder/'snapshots').mkdir()
    indices={0,(len(backend.timesteps)-1)//4,(len(backend.timesteps)-1)//2,3*(len(backend.timesteps)-1)//4,len(backend.timesteps)-1}
    def snapshot(index,rgb):
        if index in indices:tensor_to_pil(rgb[0]).save(folder/'snapshots'/('clean_erp_step_%03d.png'%index))
    result,timing=timed(lambda:pipe.run(initial,conditioning,snapshot_callback=snapshot))
    if [native_digest(s) for s in initial]!=hashes:raise AssertionError('Initial local states mutated')
    for key,value in timing.items():setattr(result,key,value)
    tensor_to_pil(result.erp_rgb[0]).save(folder/'final_erp.png')
    tensor_to_pil(result.final_consensus_erp[0]).save(folder/'final_consensus_erp.png')
    for i,rgb in enumerate(result.final_views):tensor_to_pil(rgb[0]).save(folder/('final_view_%02d.png'%i))
    torch.save(counts.cpu(),folder/'contributors.pt')
    tensor_to_pil((counts[0].expand(3,-1,-1)/counts.max())*2-1).save(folder/'contributors.png')
    save_run_metadata(str(folder/'metadata.json'),config,backend,result,str(folder/'final_erp.png'))
    check_runtime(read(folder/'metadata.json'),baseline)
    write(folder/'transition_patches.json',result.transition_records)
    records=[dict(step=s.step_index,timestep=s.scheduler_timestep,coverage_percent=s.coverage_percent,
        multi_contributor_percent=s.multi_contributor_percent,**s.state_statistics,**{k+'_seconds':v for k,v in s.timings_seconds.items()}) for s in result.steps]
    for filename,rows in (('steps.csv',records),('transition_patches.csv',result.transition_records)):
        with (folder/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(folder/'comparison.json',dict(experiment='K',backend=spec['backend'],config=config.to_dict(),
        prompt=load_directional_prompts(config.prompt.path)[2],negative_prompt='',seed=config.experiment.seed,
        job=os.environ.get('SLURM_JOB_ID'),node=platform.node(),gpu=torch.cuda.get_device_name(),
        timing=timing,total_seconds=time.perf_counter()-started,cameras=[asdict(c) for c in cameras],coverage=coverage,
        camera_sha256=pipe.camera_sha256,audit=result.audit,initial_local_sha256=hashes,
        j_reference=spec['j_output'],runtime_fields_checked=keys,geometry_change_only=True,
        final_pairwise_overlap=result.final_pairwise))
    print('Completed K',spec['backend'],timing,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['all','sd35','flux','sana','sd2'],default='all')
    parser.add_argument('--geometry-only',action='store_true')
    args=parser.parse_args();specs=read('configs/experiments/erp_later/k-all-models.json')
    if args.geometry_only:geometry_only(specs);return
    failures=[]
    for b,spec in specs.items():
        if args.backend in ('all',b):
            try:run(spec)
            except Exception:
                # Preserve each failure and continue independent backends in the
                # requested order; the aggregate audit will reject missing runs.
                if args.backend!='all':raise
                import traceback
                traceback.print_exc()
                failures.append(b)
            gc.collect();torch.cuda.empty_cache()
    if failures:raise RuntimeError("K backends failed: "+", ".join(failures))

if __name__=='__main__':main()
