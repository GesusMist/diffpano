"""R: real PixelDiT native/shared/independent initialization control."""
import json
import os
import platform
import resource
import time
from pathlib import Path
import torch
from diffpano.config import load_experiment_config
from diffpano.consensus_audit import image_metrics,montage,thumbnail
from diffpano.diagnostics import tensor_to_pil
from diffpano.erp_local_consensus import native_digest
from diffpano.initialization import set_random_seed
from diffpano.native_multidiffusion import prepare_native_backend
from diffpano.pipelines import build_view_denoiser
from diffpano.planar import build_planar_patch_layout,extract_planar_patch
from diffpano.planar_initialization_control import paired_run,independent_run,error
from diffpano.trajectory import conditioning_digest
from scripts.consensus_spatial_controls import ROOT,require_validation
from scripts.dense_erp_experiment import read,schedule_metadata
from scripts.generate import _configure_denoiser


@torch.no_grad()
def run():
    folder=ROOT/'R'
    if folder.exists():raise FileExistsError(folder)
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('GPU allocation required')
    require_validation()
    c=load_experiment_config('configs/experiments/native_multidiffusion/pixeldit.yaml')
    historical_path=next(Path('outputs/native-controls/pixeldit-native_multidiffusion-global_native_canvas-average-uniform').rglob('metadata.json'))
    historical=read(historical_path)
    if c.to_dict()!=historical['config']:raise AssertionError('Native config differs from successful historical control')
    set_random_seed(c.experiment.seed);started=time.perf_counter()
    backend=build_view_denoiser(c);_configure_denoiser(c,backend);bank=prepare_native_backend(c,backend)
    condition=backend.conditioning_for_prompt_indices(bank,[8],batch_size=1)
    if backend.checkpoint_path!=historical['model']['source'] or backend.official_commit!=historical['model']['official_commit']:raise AssertionError('Checkpoint differs')
    schedule=schedule_metadata(backend)
    if schedule['sigmas']!=historical['pixeldit_flow_schedule']:raise AssertionError('Schedule differs from successful native run')
    g=c.native_multidiffusion;layout=build_planar_patch_layout(g.canvas_height,g.canvas_width,g.patch_size,g.stride)
    initial=backend.sample_initial_native_state(batch_size=1,native_height=g.canvas_height,native_width=g.canvas_width,
        generator=torch.Generator(device=backend.device).manual_seed(c.experiment.seed))
    initial_hash=native_digest(initial);shared_hashes=[native_digest(extract_planar_patch(initial,p)) for p in layout.patches]
    forward_count=[0];original=backend.model.forward
    def counted(*args,**kwargs):
        forward_count[0]+=1
        return original(*args,**kwargs)
    backend.model.forward=counted
    torch.cuda.reset_peak_memory_stats();loop_start=time.perf_counter()
    native,shared,records,paired_calls=paired_run(backend,g,initial,condition,lambda r:print('R paired step',r['step'],'input MAE',r['model_inputs']['mae'],'oracle',r['same_input_algebraic_oracle']['max_abs'],flush=True))
    paired_seconds=time.perf_counter()-loop_start
    # Report propagated bf16 model-boundary differences separately. The strict
    # float32 same-input oracle above is the solver/aggregation validation gate.
    final_error=error(native,shared)
    folder.mkdir(parents=True,exist_ok=False)
    for name,value in [('native',native),('shared',shared)]:
        tensor_to_pil(value.detach().cpu().clone()[0]).save(folder/(name+'.png'))
    (folder/'metrics.json').write_text(json.dumps(dict(status='paired_complete_awaiting_review', comparisons=records,
        final_native_shared=final_error, solver_oracle_passed=True),indent=2)+'\n')
    print('R paired complete; waiting for evidence review before independent initialization',flush=True)
    review_path=ROOT/'execution.json'
    while not review_path.exists() or not read(review_path).get('r_independent_reviewed',False):
        time.sleep(5)
    generator=torch.Generator(device=backend.device).manual_seed(c.experiment.seed)
    independent=[backend.sample_initial_native_state(batch_size=1,native_height=g.patch_size,native_width=g.patch_size,generator=generator) for _ in layout.patches]
    independent_hashes=[native_digest(x) for x in independent]
    if len(set(independent_hashes))!=len(independent):raise AssertionError('Independent patches accidentally share identical noise')
    torch.cuda.synchronize();independent_start=time.perf_counter()
    output,independent_records,independent_calls=independent_run(backend,g,independent,condition,lambda r:print('R independent step',r['step'],flush=True))
    torch.cuda.synchronize();independent_seconds=time.perf_counter()-independent_start
    backend.model.forward=original
    metadata=dict(experiment='R',config=c.to_dict(),historical_metadata=str(historical_path),model_checkpoint=backend.checkpoint_path,model_revision=backend.official_commit,
        schedule=schedule,conditioning_sha256=conditioning_digest(condition),seed=c.experiment.seed,
        initialization=dict(native='single CUDA FP32 unit Gaussian canvas; backend sigma scaling',shared='exact clones of same global canvas crops; global canvas discarded by local algorithm',
            independent='sequential distinct patch draws from one reset CUDA generator, same seed and marginal scale',scale=backend.native_initial_noise_sigma,
            global_sha256=initial_hash,shared_patch_sha256=shared_hashes,independent_patch_sha256=independent_hashes),
        comparisons=records,independent_trajectory=independent_records,final_native_shared=final_error,
        solver_oracle_passed=True,oracle_tolerance='32 * float32 epsilon * max(1, absolute next-state maximum); applies to same-input algebra, not propagated bf16 model differences',
        native_metrics=image_metrics(native),shared_metrics=image_metrics(shared),independent_metrics=image_metrics(output),
        paired_guided_predictions=paired_calls,independent_guided_predictions=independent_calls,actual_transformer_forwards=forward_count[0],
        paired_seconds=paired_seconds,independent_seconds=independent_seconds,total_seconds=time.perf_counter()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,host_max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        slurm_job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),node=platform.node(),vae_calls=0)
    for name,value in [('native',native),('shared',shared),('independent',output)]:tensor_to_pil(value.detach().cpu().clone()[0]).save(folder/(name+'.png'))
    montage([('R PixelDiT',[(name,thumbnail(value,(512,256))) for name,value in [('native',native),('shared',shared),('independent',output)]])],folder/'comparison.png',cell=(512,256))
    (folder/'metrics.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('R COMPLETED',final_error,flush=True)

if __name__=='__main__':run()
