"""Four guarded V runs: unchanged S sampler, explicit one-time initialization."""
import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
import torch
from diffpano.camera import spherediff_camera_cover
from diffpano.consensus_audit import camera_slots,view_metrics,image_metrics,milestones
from diffpano.dense_consensus import DenseERPLocalCurrentStatePipeline
from diffpano.diagnostics import tensor_to_pil
from diffpano.erp_noise_initialization import ERPNoiseConfig,initialize_erp_noise,native_cameras,primary_noise_size,states_digest
from diffpano.initialization import set_random_seed
from diffpano.noise_v_diagnostics import NoiseVAudit
from diffpano.pipelines import build_view_denoiser
from diffpano.projection import ProjectionCache
from diffpano.trajectory import conditioning_digest
from diffpano.warp import StandardWarpOperator
from scripts.consensus_dense_controls import weight_stats
from scripts.dense_erp_experiment import prepare,schedule_metadata,canonical_schedule
from scripts.generate import _configure_denoiser
from scripts.noise_v_common import ROOT,S_ROOT,CACHE,read,s_config,require_validation,source_hashes


@torch.no_grad()
def run(name,variant):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('V requires a GPU compute allocation')
    require_validation();pre=read(ROOT/'initialization-preflight.json')
    if not pre['completed'] or not pre['passed']:raise AssertionError('Completed initialization preflight required')
    if name=='flux' and not read(ROOT/'execution.json').get('pixeldit_reviewed',False):raise AssertionError('PixelDiT validity review required before FLUX')
    folder=ROOT/name/variant
    if folder.exists():raise FileExistsError(folder)
    c,s=s_config(name);geometry=pre['backends'][name]
    if name=='flux':
        current=(CACHE/'models--ModelsLab--flux.1-dev/refs/main').read_text().strip()
        if current!=s['current_cached_revision']:raise AssertionError('FLUX cached source changed since S')
    started=time.perf_counter();set_random_seed(c.experiment.seed)
    backend=build_view_denoiser(c);_configure_denoiser(c,backend)
    cameras=spherediff_camera_cover(c.view)
    op=StandardWarpOperator(c.warp,c.fusion,ProjectionCache(max_entries=2,cpu_fallback=True))
    pipe=DenseERPLocalCurrentStatePipeline(backend=backend,cameras=cameras,erp_size=(c.erp.height,c.erp.width),warp_operator=op)
    if pipe.camera_sha256!=s['camera_sha256'] or pipe.camera_sha256!=geometry['camera_sha256']:raise AssertionError('Camera mismatch')
    conditions,slots=prepare(c,backend,pipe);schedule=schedule_metadata(backend)
    if canonical_schedule(schedule)!=canonical_schedule(s['prepared_schedule']):raise AssertionError('Schedule mismatch')
    condhash=hashlib.sha256(''.join(conditioning_digest(v) for v in conditions).encode()).hexdigest()
    if condhash!=s['conditioning_sha256']:raise AssertionError('Conditioning mismatch')
    source=getattr(backend,'checkpoint_path',None) or c.model.path or c.model.id
    revision=getattr(backend,'official_commit',None) or c.model.revision
    if source!=s['model_checkpoint'] or revision!=s['model_revision']:raise AssertionError('Model provenance mismatch')
    primary=primary_noise_size(native_cameras(backend,cameras))
    if list(primary)!=geometry['primary_noise_size'] or backend.native_channels!=geometry['native_channels']:raise AssertionError('Native/noise dimensions differ from preflight')
    if backend.native_initial_noise_sigma!=geometry['native_scale']:raise AssertionError('Scaling mismatch')
    weights=weight_stats(pipe)
    if weights!=s['weights']:raise AssertionError('S center weighting changed')
    # Cheap initializer replay only: verify S's original initial-state hash, no model replay.
    direct=pipe.initialize_local_states(c.experiment.seed,c.generation.batch_size)
    direct_hash=states_digest(direct);del direct
    if direct_hash!=s['audit']['initial_local_sha256']:raise AssertionError('Direct initializer does not reproduce S')
    init=ERPNoiseConfig(variant,*primary,c.experiment.seed);init_start=time.perf_counter()
    states,init_record=initialize_erp_noise(backend,cameras,init,c.generation.batch_size)
    init_seconds=time.perf_counter()-init_start
    expected=next(x for x in geometry['records'] if x['scale']=='primary')
    if init_record['map_sha256']!=expected['maps_sha256']:raise AssertionError('Initialization maps differ from preflight')
    # Cross-variant first-camera equality is checked after both independent jobs
    # complete, before interpreting the pair or releasing the FLUX stage.
    forward_count=[0];module=backend.pipeline.transformer if name=='flux' else backend.model
    original=module.forward
    def counted(*args,**kwargs):forward_count[0]+=1;return original(*args,**kwargs)
    module.forward=counted;audit=NoiseVAudit(cameras,len(backend.timesteps));records=[];selected=milestones(len(backend.timesteps))
    def progress(step,total,metrics):
        if step+1 in selected:records.append(dict(step=step+1,**metrics))
        print(name,variant,step+1,'/',total,flush=True)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();loop_start=time.perf_counter()
    result=pipe.run_dense(states,conditions,stage_audit=audit,expected_initial_sha256=init_record['initial_local_sha256'],progress=progress)
    torch.cuda.synchronize();runtime=time.perf_counter()-loop_start;module.forward=original
    expected_calls=len(cameras)*len(backend.timesteps)
    if result.audit['guided_predictions']!=expected_calls or forward_count[0]!=s['actual_transformer_forward_invocations']:raise AssertionError('Model call count changed')
    result.audit['initialization']=variant+'; one-time CPU source field(s); see initialization metadata'
    result.audit['transient_erp_native_field_at_initialization']=True
    final_metrics=dict(erp=image_metrics(result.erp_rgb,spherical=True),views=[dict(slot=i,**view_metrics(op.erp_to_perspective(result.erp_rgb,cameras[i]))) for i in camera_slots(cameras)])
    metadata=dict(experiment='V',backend=name,variant=variant,config=dict(sampler=c.to_dict(),noise_initialization=asdict(init)),
                  sampler_config_diff_from_S={},s_reference=str(S_ROOT/name/'metadata.json'),model_checkpoint=source,model_revision=revision,
                  current_cached_revision=s.get('current_cached_revision'),prepared_schedule=schedule,conditioning_sha256=condhash,prompt_indices=slots,
                  camera_sha256=pipe.camera_sha256,clean_rgb_erp_size=[c.erp.height,c.erp.width],weights=weights,
                  initialization=init_record,initialization_seconds=init_seconds,direct_S_initializer_sha256=direct_hash,
                  initialization_preflight_summary={k:expected[k] for k in ['aggregate','matched_source_agreement','overlapping_pairs']},
                  runtime_seconds=runtime,total_seconds=time.perf_counter()-started,stage_seconds=result.stage_seconds,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                  host_max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                  actual_transformer_forward_invocations=forward_count[0],audit=result.audit,milestone_agreement=records,final_metrics=final_metrics,
                  slurm_job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),node=platform.node(),
                  environment=dict(torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version()),
                  git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  git_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff'])).hexdigest(),source_hashes=source_hashes(),
                  limitations=['One seed; S historical per-step local predictions were not saved.','FLUX historical resolved revision unavailable; cached source matches S recorded cache.'] if name=='flux' else ['One seed; S historical per-step local predictions were not saved.'])
    folder.mkdir(parents=True,exist_ok=False)
    tensor_to_pil(result.erp_rgb.detach().cpu().clone()[0]).save(folder/'final_result.png')
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('COMPLETED',name,variant,'seconds',runtime,'guided',expected_calls,'forwards',forward_count[0],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=['pixeldit','flux']);p.add_argument('variant',choices=['V-independent-erp','V-shared-erp']);a=p.parse_args();run(a.backend,a.variant)
