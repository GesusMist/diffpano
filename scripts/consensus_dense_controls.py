"""P/S/T: guarded matched dense controls, compact outputs, no baseline overwrite."""
import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from dataclasses import replace
from pathlib import Path
import torch
from diffpano.camera import spherediff_camera_cover
from diffpano.config import load_experiment_config
from diffpano.consensus_audit import StageAudit,camera_slots,image_metrics,view_metrics,montage
from diffpano.dense_consensus import DenseERPLocalCurrentStatePipeline
from diffpano.dense_geometry import contributor_stats
from diffpano.diagnostics import tensor_to_pil
from diffpano.initialization import set_random_seed
from diffpano.pipelines import build_view_denoiser
from diffpano.projection import ProjectionCache
from diffpano.trajectory import conditioning_digest
from diffpano.warp import StandardWarpOperator
from scripts.generate import _configure_denoiser
from scripts.dense_erp_experiment import ROOT as BASE,read,prepare,schedule_metadata,canonical_schedule,config_differences
from scripts.consensus_spatial_controls import ROOT,require_validation


def make_config(label,name):
    c=load_experiment_config(f'configs/experiments/erp_later/{name}-l.yaml')
    c.dense_consensus.experiment=label
    if label=='S':c.fusion=replace(c.fusion,mode='weighted_average',weight_mode='spherediff_center')
    if label=='T':c.erp=replace(c.erp,height=c.erp.height*2,width=c.erp.width*2)
    c.validate()
    return c


def pairing(c):
    baseline=read(BASE/'L'/c.model.pipeline/'metadata.json')
    differences=config_differences(baseline['config'],c.to_dict())
    label=c.dense_consensus.experiment
    expected={'dense_consensus.experiment':['L',label]}
    if label=='S':expected.update({'fusion.mode':['average','weighted_average'],'fusion.weight_mode':['uniform','spherediff_center']})
    if label=='T':expected.update({'erp.height':[1024,2048],'erp.width':[2048,4096]})
    if differences!=expected:raise AssertionError(f'Unexpected paired config changes: {differences}')
    if hashlib.sha256(Path(c.prompt.path).read_bytes()).hexdigest()!=baseline['prompt_sha256']:raise AssertionError('Prompt changed')
    return baseline,differences


def weight_stats(pipe):
    acc=pipe._accumulator(1)
    for camera in pipe.cameras:
        rgb=torch.ones(1,3,camera.height,camera.width,device=pipe.backend.device)
        acc.accumulate(pipe.operator.perspective_to_erp(rgb,camera,pipe.erp_size))
    output=acc.finalize();w=output.accumulated_weight
    def summary(x):
        return dict(min=float(x.min()),p01=float(torch.quantile(x.flatten(),.01)),median=float(x.median()),p99=float(torch.quantile(x.flatten(),.99)),max=float(x.max()),mean=float(x.mean()))
    result=dict(contributors=contributor_stats(output.contributor_count),accumulated_weight=summary(w),uncovered=int((w==0).sum()),small_positive=int(((w>0)&(w<1e-6)).sum()))
    if acc.squared_weight is not None:result['effective_contributors']=summary(w.square()/acc.squared_weight.clamp_min(torch.finfo(w.dtype).tiny))
    torch.testing.assert_close(output.erp_rgb,torch.ones_like(output.erp_rgb),atol=2e-6,rtol=2e-6)
    return result


@torch.no_grad()
def run(label,name):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Model inference requires GPU compute allocation')
    folder=ROOT/label/name
    if folder.exists():raise FileExistsError(folder)
    require_validation()
    q_gate=read(ROOT/'Q/metrics.json')
    if not read(ROOT/'validation.json')['passed'] or not q_gate['passed'] or not q_gate['completed']:raise AssertionError('Validation and complete Q must pass')
    if label in {'S','T'} and not read(ROOT/'review.json')['proceed_st']:raise AssertionError('P/R evidence review is required')
    c=make_config(label,name);baseline,differences=pairing(c)
    geometry=read(c.dense_consensus.geometry_file)
    cameras=spherediff_camera_cover(c.view,overlap_fraction=geometry['overlap_fraction'])
    q=read(ROOT/'Q/metrics.json');verified=next(r for r in q['geometry'] if r['operator']=='L' and r['erp_size']==[c.erp.height,c.erp.width])
    if verified['coverage_percent']!=100 or verified['minimum']<1:raise AssertionError('New resolution geometry not verified')
    started=time.perf_counter();set_random_seed(c.experiment.seed)
    backend=build_view_denoiser(c);_configure_denoiser(c,backend)
    op=StandardWarpOperator(c.warp,c.fusion,ProjectionCache(max_entries=2,cpu_fallback=True))
    pipe=DenseERPLocalCurrentStatePipeline(backend=backend,cameras=cameras,erp_size=(c.erp.height,c.erp.width),warp_operator=op)
    if pipe.camera_sha256!=baseline['camera_sha256'] or pipe.camera_sha256!=verified['camera_sha256']:raise AssertionError('Camera mismatch')
    conditions,slots=prepare(c,backend,pipe)
    schedule=schedule_metadata(backend)
    if canonical_schedule(schedule)!=canonical_schedule(baseline['scheduler']):raise AssertionError('Actual schedule mismatch')
    condhash=hashlib.sha256(''.join(conditioning_digest(v) for v in conditions).encode()).hexdigest()
    if condhash!=baseline['audit']['conditioning_sha256']:raise AssertionError('Conditioning mismatch')
    source=getattr(backend,'checkpoint_path',None) or c.model.path or c.model.id
    revision=getattr(backend,'official_commit',None) or c.model.revision
    if source!=baseline['model_checkpoint'] or revision!=baseline['model_revision']:raise AssertionError('Model provenance mismatch')
    weights=weight_stats(pipe)
    states=pipe.initialize_local_states(c.experiment.seed,c.generation.batch_size)
    forward_count=[0]
    module=backend.pipeline.transformer if name=='flux' else backend.model
    original_forward=module.forward
    def counted(*args,**kwargs):
        forward_count[0]+=1
        return original_forward(*args,**kwargs)
    module.forward=counted
    audit=StageAudit(cameras,len(backend.timesteps),name) if label=='P' else None
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();loop_start=time.perf_counter()
    result=pipe.run_dense(states,conditions,stage_audit=audit,expected_initial_sha256=baseline['audit']['initial_local_sha256'],
        progress=lambda step,total,metrics:print(label,name,step+1,'/',total,flush=True))
    torch.cuda.synchronize();runtime=time.perf_counter()-loop_start;module.forward=original_forward
    # Only detached display copies are mapped/clamped. Raw metrics remain raw.
    final_metrics=dict(erp=image_metrics(result.erp_rgb,spherical=True),views=[dict(slot=i,**view_metrics(op.erp_to_perspective(result.erp_rgb,cameras[i]))) for i in camera_slots(cameras)])
    metadata=dict(experiment=label,backend=name,config=c.to_dict(),config_diff_from_L=differences,baseline=str(BASE/'L'/name/'metadata.json'),
        model_checkpoint=source,model_revision=revision,prepared_schedule=schedule,conditioning_sha256=condhash,prompt_indices=slots,
        camera_sha256=pipe.camera_sha256,geometry_verification=verified,weights=weights,
        runtime_seconds=runtime,total_seconds=time.perf_counter()-started,stage_seconds=result.stage_seconds,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        host_max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        actual_transformer_forward_invocations=forward_count[0],audit=result.audit,final_metrics=final_metrics,
        slurm_job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),node=platform.node(),
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version()),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff'])).hexdigest(),
        limitations=['FLUX historical L did not record a resolved revision; checkpoint ID, settings, conditioning and initial state are checked.'] if name=='flux' else [])
    if name=='flux':
        ref=Path(os.environ['HF_HUB_CACHE'])/'models--ModelsLab--flux.1-dev/refs/main'
        metadata['current_cached_revision']=ref.read_text().strip() if ref.exists() else None
    folder.mkdir(parents=True,exist_ok=False)
    tensor_to_pil(result.erp_rgb.detach().cpu().clone()[0]).save(folder/('terminal_result.png' if label=='P' else 'final_result.png'))
    if audit is not None:
        tensor_to_pil(audit.last_clean[0]).save(folder/'last_clean_consensus.png')
        montage(audit.rows,folder/'stage_montage.png')
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('COMPLETED',label,name,'runtime',runtime,'guided',result.audit['guided_predictions'],'forwards',forward_count[0],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('label',choices=['P','S','T']);p.add_argument('backend',choices=['flux','pixeldit'])
    a=p.parse_args();run(a.label,a.backend)
