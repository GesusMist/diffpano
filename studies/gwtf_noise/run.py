"""Per-backend zero-denoiser GPU gate, then unchanged B0C0D1 generation."""
import argparse
import contextlib
import hashlib
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import torch
from diffpano.bridge_factorial import BridgeFactorialPipeline,make_operator,angular_geometry,digest
from diffpano.consensus_audit import camera_slots,image_metrics,view_metrics,milestones
from diffpano.diagnostics import tensor_to_pil
from diffpano.erp_noise_initialization import ERPNoiseConfig,initialize_erp_noise,native_cameras,primary_noise_size
from diffpano.gwtf_noise_initialization import GWTFlowNoiseConfig,initialize_gwtf_shared_noise
from diffpano.factorial_diagnostics import camera_boundary_metric
from diffpano.initialization import set_random_seed
from diffpano.noise_v_diagnostics import NoiseVAudit
from diffpano.pipelines import build_view_denoiser
from diffpano.trajectory import conditioning_digest
from scripts.generate import _configure_denoiser
from scripts.dense_erp_experiment import prepare,schedule_metadata,canonical_schedule,seam_metric
from scripts.bridge_factorial_run import metric_summary,environment
from studies.gwtf_noise.common import *
from studies.gwtf_noise.statistics import state_statistics


@contextlib.contextmanager
def count_calls(b,name,forbid=False):
    module=b.model if name=='pixeldit' else (b.pipeline.unet if name=='sd2' else b.pipeline.transformer)
    saved=[];counts=dict(denoiser=0,encode=0,decode=0,initialize=0)
    def wrap(obj,method,key):
        original=getattr(obj,method)
        def invoke(*a,**kw):
            counts[key]+=1
            if forbid and key in ('denoiser','encode','decode'):raise AssertionError('Forbidden initializer call: '+key)
            return original(*a,**kw)
        saved.append((obj,method,original));setattr(obj,method,invoke)
    wrap(module,'forward','denoiser');wrap(b,'initialize_native_state','initialize')
    if name!='pixeldit':wrap(b.pipeline.vae,'encode','encode');wrap(b.pipeline.vae,'decode','decode')
    try:yield counts
    finally:
        for obj,method,original in reversed(saved):setattr(obj,method,original)


def checkpoint_inventory(c,b):
    if c.model.pipeline=='pixeldit':
        p=Path(b.checkpoint_path)
        return dict(checkpoint=str(p),resolved=str(p.resolve()),bytes=p.stat().st_size,
            upstream_commit=b.official_commit,config_sha256=sha(c.pixeldit.config_path),
            content_address=p.resolve().name,note='Cached content address and pinned upstream revision; no full weight rehash')
    folder=CACHE/('models--'+c.model.id.replace('/','--'))/'snapshots'/c.model.revision
    records={}
    for p in sorted(folder.rglob('*')):
        if p.is_file() and p.suffix in ('.json','.safetensors','.bin'):
            records[str(p.relative_to(folder))]=dict(bytes=p.stat().st_size,content_address=p.resolve().name,
                sha256=sha(p) if p.suffix=='.json' else None)
    return dict(revision=c.model.revision,snapshot=str(folder),files=records,note='Pinned immutable revision plus cached blob addresses; no independent full weight rehash')


def runtime_audit(c,b,p,conditions,slots,old):
    now=dict(model_checkpoint=getattr(b,'checkpoint_path',None) or c.model.path or c.model.id,
        model_revision=getattr(b,'official_commit',None) or c.model.revision,
        prepared_schedule=canonical_schedule(schedule_metadata(b)),
        conditioning_sha256=hashlib.sha256(''.join(conditioning_digest(v) for v in conditions).encode()).hexdigest(),
        prompt_indices=slots,camera_sha256=p.camera_sha256,camera_geometry_sha256=digest(angular_geometry(p.cameras)),
        native_channels=b.native_channels,local_native_resolution=list(b.native_spatial_shape_for_rgb(c.view.height,c.view.width)),
        local_RGB_resolution=[c.view.height,c.view.width],noise_grid=list(primary_noise_size(native_cameras(b,p.cameras))),
        native_scale=float(b.native_initial_noise_sigma),bridge_mode=p.bridge_mode,native_arithmetic_dtype='torch.float32',
        vae_dtype=str(b.pipeline.vae.dtype) if c.model.pipeline!='pixeldit' else None)
    differences={k:dict(historical=old.get(k),current=v) for k,v in now.items() if old.get(k)!=v}
    coefficients=None
    if c.model.pipeline=='sd2':
        from diffusers import DDIMScheduler
        ref=DDIMScheduler.from_config(old['prepared_schedule']['config']);ref.set_timesteps(len(old['prepared_schedule']['timesteps']))
        actual=b.pipeline.scheduler
        equal=torch.equal(ref.alphas_cumprod.cpu(),actual.alphas_cumprod.cpu()) and torch.equal(ref.final_alpha_cumprod.cpu(),actual.final_alpha_cumprod.cpu())
        intervals=[]
        for t in b.timesteps:
            t=int(t);prev=t-actual.config.num_train_timesteps//actual.num_inference_steps
            a=actual.alphas_cumprod[t];n=actual.alphas_cumprod[prev] if prev>=0 else actual.final_alpha_cumprod
            intervals.append(dict(t=t,alpha=float(a.sqrt()),sigma=float((1-a).sqrt()),next_alpha=float(n.sqrt()),next_sigma=float((1-n).sqrt())))
        coefficients=dict(exact_match=equal,intervals=intervals,historical_evidence='Reconstructed from saved full DDIM config under matched diffusers version; historical metadata did not serialize alpha array')
        if not equal:differences['ddim_alphas']='Prepared alpha arrays differ from reconstructed historical scheduler'
    return now,dict(matched=not differences,differences=differences,checked_fields=list(now),ddim_coefficients=coefficients,
        full_config_match=c.to_dict()==old['config'],checkpoint_provenance=checkpoint_inventory(c,b))


@torch.no_grad()
def run(name):
    manifest=require_validation(statistical=True);start=time.perf_counter()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('GPU Slurm allocation required')
    c,stub,cams,old,_=backend_geometry(name)
    assert c.to_dict()==manifest['models'][name]['config']
    folder=ROOT/name
    if (folder/'metadata.json').exists() or (folder/'gpu-preflight.json').exists():raise FileExistsError(folder)
    set_random_seed(0);b=build_view_denoiser(c);_configure_denoiser(c,b)
    p=BridgeFactorialPipeline(backend=b,cameras=cams,erp_size=(c.erp.height,c.erp.width),warp_operator=make_operator(c),backend_name=name)
    conditions,slots=prepare(c,b,p);provenance,audit=runtime_audit(c,b,p,conditions,slots,old)
    assert audit['full_config_match']
    for key in ('camera_sha256','camera_geometry_sha256','noise_grid','native_channels','local_native_resolution','native_scale'):
        assert provenance[key]==old[key],key
    h,w=provenance['noise_grid'];cpu=read(ROOT/'statistical-preflight'/(name+'.json'))
    config=GWTFlowNoiseConfig(h,w,0)
    init_start=time.perf_counter();rng=torch.random.get_rng_state().clone()
    with count_calls(b,name,forbid=True) as calls:
        states,init=initialize_gwtf_shared_noise(b,cams,config,c.generation.batch_size,camera_slots=list(range(89)))
        init_seconds=time.perf_counter()-init_start
        assert init['initial_local_sha256']==cpu['exact_seed0']['gwtf']['record']['initial_local_sha256'],'CPU/GPU native scaling replay mismatch'
        exact=state_statistics(states,b,cams)
        replay,rr=initialize_gwtf_shared_noise(b,cams,config,c.generation.batch_size,camera_slots=list(range(89)),execution_order=list(reversed(range(89))))
        assert rr['initial_local_sha256']==init['initial_local_sha256'];del replay
    assert calls==dict(denoiser=0,encode=0,decode=0,initialize=178)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(s.shape==(c.generation.batch_size,b.native_channels,*provenance['local_native_resolution']) and bool(torch.isfinite(s).all()) for s in states)
    gpu=dict(passed=True,backend=name,provenance=provenance,control_audit=audit,exact_initialization=init,
        statistics=exact,replay_bitwise_identical=True,calls=calls,environment=environment(),source_hashes=source_hashes())
    write(folder/'gpu-preflight.json',gpu)
    print('GPU ZERO-DENOISER PREFLIGHT PASSED',name,'reuse control',audit['matched'],flush=True)
    # Mismatched dynamic settings trigger an isolated fresh current control.
    if not audit['matched']:
        current,current_init=initialize_erp_noise(b,cams,ERPNoiseConfig('V-shared-erp',h,w,0))
        generate(name,'current-control',c,b,p,conditions,current,current_init,provenance,audit,0.,None,start)
        del current
    generate(name,'gwtf',c,b,p,conditions,states,init,provenance,audit,init_seconds,exact,start)


@torch.no_grad()
def generate(name,method,c,b,p,conditions,states,init,provenance,control_audit,init_seconds,exact,start):
    require_validation(statistical=True)
    folder=ROOT/name if method=='gwtf' else ROOT/name/method
    stage_audit=NoiseVAudit(p.cameras,len(b.timesteps));records=[];selected=milestones(len(b.timesteps))
    def progress(step,total,metrics):
        if step+1 in selected:records.append(dict(step=step+1,**metrics))
        print(name,method,step+1,'/',total,'current error',metrics['current_state_error_max'],flush=True)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
    with count_calls(b,name) as counts:
        result=p.run_dense(states,conditions,expected_initial_sha256=init['initial_local_sha256'],stage_audit=stage_audit,progress=progress)
    torch.cuda.synchronize();runtime=time.perf_counter()-t
    expected=89*c.generation.num_inference_steps
    assert result.audit['guided_predictions']==counts['denoiser']==expected
    assert counts['initialize']==0
    assert counts['encode']==(0 if name=='pixeldit' else 2*expected)
    assert counts['decode']==(0 if name=='pixeldit' else expected+89)
    final_metrics=dict(erp=image_metrics(result.erp_rgb,spherical=True),views=[dict(slot=i,**view_metrics(p.diagnostic_operator.erp_to_perspective(result.erp_rgb,p.cameras[i]))) for i in camera_slots(p.cameras)])
    boundary=camera_boundary_metric(result.erp_rgb,p.cameras)
    image=tensor_to_pil(result.erp_rgb.detach().cpu().clone()[0]);wrap=seam_metric(image)
    summary=metric_summary(final_metrics,records,result.audit['stage_audit'],wrap)
    summary.update(camera_boundary_gradient=boundary['boundary_gradient'],camera_boundary_ratio=boundary['ratio'])
    result.audit['stage_audit']['note']='Same selected aligned local RGB pairs and milestone diagnostics as historical factorial; no extra VAE/denoiser calls.'
    result.audit.update(initialization=method,vae_bridge=p.bridge_mode,local_residual_never_warped=True,persistent_erp_native_state=False)
    filename='gwtf-final.png' if method=='gwtf' else 'final_result.png'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/filename).exists() or (folder/'metadata.json').exists():raise FileExistsError(folder)
    image.save(folder/filename)
    metadata=dict(backend=name,method=method,config=c.to_dict(),**provenance,control_audit=control_audit,
        initialization=init,initialization_seconds=init_seconds,exact_initial_statistics=exact,
        audit=result.audit,milestones=records,aggregate_metrics=result.metrics,final_metrics=final_metrics,summary=summary,
        wrap=wrap,camera_boundary_seam=boundary,stage_seconds=result.stage_seconds,runtime_seconds=runtime,total_seconds=time.perf_counter()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        host_process_max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,actual_transformer_forward_invocations=counts['denoiser'],
        vae_calls={k:counts[k] for k in ('encode','decode')},environment=environment(),source_hashes=source_hashes(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        reused_current_control=str(BASE/'cells'/name/CELL) if control_audit['matched'] else str(ROOT/name/'current-control'),
        output=str(folder/filename),limitations=['One ruins prompt and seed 0','Matched-ray/pair measures are not full 3D coherence','HF is not perceptual quality'])
    write(folder/'metadata.json',metadata)
    print('COMPLETED',name,method,runtime,'seconds',counts,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('backend',choices=BACKENDS);run(parser.parse_args().backend)
