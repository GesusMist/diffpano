"""Dense L/M/N/O launcher with opt-in N/O 10%-interval clean ERP snapshots."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
import torch
from diffpano.camera import spherediff_camera_cover
from diffpano.config import load_experiment_config
from diffpano.dense_consensus import DenseERPLocalCurrentStatePipeline, prepare_camera_conditioning
from diffpano.dense_geometry import contributor_stats
from diffpano.diagnostics import tensor_to_pil
from diffpano.erp_local_consensus import camera_digest
from diffpano.initialization import set_random_seed, load_directional_prompts
from diffpano.pipelines import build_view_denoiser
from diffpano.planar_pipeline import _negative_prompt
from diffpano.projection import ProjectionCache
from diffpano.warp import StandardWarpOperator, LaplacianPyramidWarpOperator
from diffpano.conditioning import expand_directional_prompts, camera_prompt_indices
from diffpano.trajectory import conditioning_digest
from scripts.generate import _configure_denoiser

ROOT = Path('outputs/vae-residual-controls/20260915-dense-lm')
NO_ROOT = Path('outputs/vae-residual-controls/20260916-dense-no')


def read(path): return json.loads(Path(path).read_text())


def config_differences(a,b,prefix=''):
    result={}
    for key in sorted(set(a)|set(b)):
        path=prefix+key
        if isinstance(a.get(key),dict) and isinstance(b.get(key),dict):
            result.update(config_differences(a[key],b[key],path+'.'))
        elif a.get(key)!=b.get(key):result[path]=[a.get(key),b.get(key)]
    return result


def check_l_pairing(config):
    reference_path='configs/experiments/erp_later/'+config.model.pipeline+'-l.yaml'
    reference=load_experiment_config(reference_path)
    baseline=read(ROOT/'L'/config.model.pipeline/'metadata.json')
    if reference.to_dict()!=baseline['config']:
        raise AssertionError('The L config differs from the actual completed L run')
    differences=config_differences(reference.to_dict(),config.to_dict())
    expected={'dense_consensus.experiment':['L',config.dense_consensus.experiment],
              'fusion.mode':['average','detail_preserving_average']}
    if config.dense_consensus.experiment=='O':
        expected.update({'warp.mode':['standard','lpw'],'warp.lpw.levels':[4,5],'warp.lpw.lod_mode':['jacobian','none']})
    if differences!=expected:raise AssertionError('Unexpected L ablation difference: '+str(differences))
    if hashlib.sha256(Path(config.prompt.path).read_bytes()).hexdigest()!=baseline['prompt_sha256']:
        raise AssertionError('Prompt contents changed from L')
    return reference_path


def schedule_metadata(backend):
    scheduler=getattr(getattr(backend,'pipeline',None),'scheduler',None)
    result=dict(timesteps=backend.timesteps.detach().cpu().tolist(),class_name=type(scheduler).__name__ if scheduler is not None else 'PixelDiTFirstOrderSolver')
    if scheduler is not None:
        result['config']={k:sorted(v) if isinstance(v,set) else v for k,v in dict(scheduler.config).items()}
        if hasattr(scheduler,'sigmas'):result['sigmas']=scheduler.sigmas.detach().cpu().tolist()
    else:result.update(sigmas=backend.solver.schedule.detach().cpu().tolist(),flow_shift=backend.solver.flow_shift)
    return result


def canonical_schedule(schedule):
    result=dict(schedule)
    if 'config' in result:
        result['config']=dict(result['config'])
        if '_use_default_values' in result['config']:
            result['config']['_use_default_values']=sorted(result['config']['_use_default_values'])
    return result


def check_pairing(config):
    if config.dense_consensus.experiment in {'N','O'}:
        return check_l_pairing(config)
    backend = config.model.pipeline
    reference_path = ('configs/experiments/erp_later/'+backend+'-k.yaml' if backend != 'pixeldit'
        else 'configs/experiments/implied_endpoint_consensus/pixeldit.yaml')
    reference = load_experiment_config(reference_path)
    a,b = config.to_dict(),reference.to_dict()
    for key in ('global_pipeline','view','sampling','dense_consensus','canvas','erp','consensus_transition'):
        a.pop(key,None);b.pop(key,None)
    if a != b:raise AssertionError('Settings beyond intentional geometry/prompt routing changed')
    if config.view.height != reference.native_multidiffusion.patch_size * {'sd2':8,'sd35':8,'flux':8,'sana':32,'pixeldit':1}[backend]:
        raise AssertionError('Local native resolution differs from original control')
    return reference_path


def build_dense_pipeline(config, backend):
    geometry = read(config.dense_consensus.geometry_file)
    verified = next(r for r in geometry['full_resolution_verification'] if r['erp_size']==[config.erp.height,config.erp.width] and r['view_size']==[config.view.height,config.view.width])
    minimum = 5 if config.dense_consensus.experiment == 'M' else 1
    if verified['coverage_percent'] != 100 or verified['minimum'] < minimum:
        raise AssertionError('Full-resolution CPU geometry preflight has not passed')
    cameras = spherediff_camera_cover(config.view,overlap_fraction=geometry['overlap_fraction'])
    if len(cameras) != geometry['camera_count'] or camera_digest(cameras) != verified['camera_sha256']:
        raise AssertionError('Cameras do not match verified geometry')
    cache = ProjectionCache(max_entries=4,cpu_fallback=True)
    if config.dense_consensus.experiment in {'N','O'} and geometry['experiment']!='L':
        raise AssertionError('N/O must use L geometry, never M')
    op = (LaplacianPyramidWarpOperator(config.warp,config.fusion,cache,periodic_reconstruction=True)
          if config.dense_consensus.experiment=='O' else StandardWarpOperator(config.warp,config.fusion,cache))
    pipe = DenseERPLocalCurrentStatePipeline(backend=backend,cameras=cameras,erp_size=(config.erp.height,config.erp.width),warp_operator=op,flow_transition=config.model.pipeline!='sd2')
    return pipe,geometry,minimum


def prepare(config, backend, pipe):
    backend.prepare(num_steps=config.generation.num_inference_steps,view_height=config.view.height,view_width=config.view.width)
    lines = load_directional_prompts(config.prompt.path)
    prompts = lines if config.dense_consensus.prompt_assignment == 'spherediff_directional' else [lines[2]]*5
    bank = backend.prepare_prompt_conditioning(prompts,_negative_prompt(config))
    return prepare_camera_conditioning(backend,bank,pipe.cameras,config.dense_consensus.prompt_assignment,config.generation.batch_size)


def seam_metric(image):
    import numpy as np
    rgb = np.asarray(image,dtype=np.float64)/255.
    seam = float(np.abs(rgb[:,0]-rgb[:,-1]).mean())
    differences = np.abs(np.diff(rgb,axis=1)).mean(axis=(0,2))
    nearby = float(np.concatenate((differences[:8],differences[-8:])).mean())
    return dict(boundary_gradient=seam,nearby_gradient=nearby,ratio=seam/max(nearby,1e-12),range='PNG RGB [0,1]',nearby_radius=8)


def run_config(config):
    reference = check_pairing(config)
    label = config.dense_consensus.experiment;name = config.model.pipeline
    is_no = label in {"N","O"}
    output_root = NO_ROOT if is_no else ROOT
    folder = output_root/label/name
    if folder.exists():raise FileExistsError('Refusing to overwrite '+str(folder))
    # Validate the shared preflight before loading a model.
    geometry = read(config.dense_consensus.geometry_file)
    if label=='M' and any(r['minimum']<5 for r in geometry['full_resolution_verification']):raise AssertionError('M preflight failed')
    gate=read(output_root/'validation.json')
    if not gate['passed']:raise AssertionError('Regression gate has not passed')
    started = time.perf_counter();set_random_seed(config.experiment.seed)
    backend = build_view_denoiser(config);_configure_denoiser(config,backend)
    pipe,geometry,minimum = build_dense_pipeline(config,backend)
    conditionings,slots = prepare(config,backend,pipe)
    baseline_path = Path('outputs/vae-residual-controls/20260910-lookingglass-v1')/name/'K/metadata.json'
    if baseline_path.exists():
        baseline = read(baseline_path)
        if backend.timesteps.detach().cpu().tolist() != baseline['scheduler_timesteps']:
            raise AssertionError('Prepared schedule differs from original K')
        actual_scheduler = backend.pipeline.scheduler
        if baseline['scheduler_sigmas'] is not None and actual_scheduler.sigmas.detach().cpu().tolist() != baseline['scheduler_sigmas']:
            raise AssertionError('Prepared sigmas differ from original K')
    l_baseline = read(ROOT/'L'/name/'metadata.json') if is_no else None
    if is_no:
        if canonical_schedule(schedule_metadata(backend))!=canonical_schedule(l_baseline['scheduler']):
            raise AssertionError('Actual prepared schedule differs from L')
        if pipe.camera_sha256!=l_baseline['camera_sha256']:
            raise AssertionError('Camera digest differs from L')
        l_config=load_experiment_config(reference)
        l_cameras=spherediff_camera_cover(l_config.view,overlap_fraction=geometry['overlap_fraction'])
        l_slots=camera_prompt_indices(l_cameras,expand_directional_prompts(load_directional_prompts(l_config.prompt.path)).directions).tolist()
        if slots!=l_slots:raise AssertionError('Per-camera prompt indices differ from L')
        condition_hash=hashlib.sha256(''.join(conditioning_digest(c) for c in conditionings).encode()).hexdigest()
        if condition_hash!=l_baseline['audit']['conditioning_sha256']:
            raise AssertionError('Effective conditioning differs from L')
        actual_source=getattr(backend,'checkpoint_path',None) or config.model.path or config.model.id
        actual_revision=getattr(backend,'official_commit',None) or config.model.revision
        if actual_source!=l_baseline['model_checkpoint'] or actual_revision!=l_baseline['model_revision']:
            raise AssertionError('Actual model provenance differs from L')
    gpu_stats = contributor_stats(pipe.precompute_geometry(minimum))
    states = pipe.initialize_local_states(config.experiment.seed,config.generation.batch_size)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    loop_start = time.perf_counter()
    def progress(step,total,metrics):
        print(label,name,'step',step+1,'/',total,'current error',metrics['current_state_error_max'],flush=True)
    snapshots=[]
    if is_no:folder.mkdir(parents=True,exist_ok=False)
    def save_snapshot(record,rgb):
        percentages=record['requested_percentages'];filename='consensus_%03d.png'%percentages[0]
        tensor_to_pil(rgb[0]).save(folder/filename)
        snapshots.append(dict(record,requested_percentage=percentages[0],filename=filename,
            semantics='predicted-clean consensus ERP; not decoded x_next'))
    result = pipe.run_dense(states,conditionings,required_minimum=minimum,progress=progress,
        snapshot_callback=save_snapshot if is_no else None,
        expected_initial_sha256=l_baseline['audit']['initial_local_sha256'] if is_no else None)
    torch.cuda.synchronize();runtime = time.perf_counter()-loop_start
    image = tensor_to_pil(result.erp_rgb[0])
    pipeline = getattr(backend,'pipeline',None);scheduler=getattr(pipeline,'scheduler',None)
    schedule = dict(timesteps=backend.timesteps.detach().cpu().tolist(),class_name=type(scheduler).__name__ if scheduler is not None else 'PixelDiTFirstOrderSolver')
    if scheduler is not None:
        schedule['config']={k:sorted(v) if isinstance(v,set) else v for k,v in dict(scheduler.config).items()}
        if hasattr(scheduler,'sigmas'):schedule['sigmas']=scheduler.sigmas.detach().cpu().tolist()
    else:
        schedule.update(sigmas=backend.solver.schedule.detach().cpu().tolist(),flow_shift=backend.solver.flow_shift)
    from collections import Counter
    lines=load_directional_prompts(config.prompt.path)
    metadata = dict(experiment=label,backend=name,config=config.to_dict(),reference_config=reference,
        model_checkpoint=getattr(backend,'checkpoint_path',None) or config.model.path or config.model.id,
        model_revision=getattr(backend,'official_commit',None) or config.model.revision,
        prompt_file=config.prompt.path,prompt_sha256=hashlib.sha256(Path(config.prompt.path).read_bytes()).hexdigest(),
        prompt_lines=lines,prompt_unique_texts=len(set(lines)),
        negative_prompt=(_negative_prompt(config) or config.pixeldit.negative_prompt) if name=='pixeldit' else _negative_prompt(config),seed=config.experiment.seed,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        git_dirty_status=subprocess.check_output(['git','status','--short'],text=True),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),node=platform.node(),gpu=torch.cuda.get_device_name(),
        geometry='spherediff_fixed' if label!='M' else 'dense_min5',camera_count=len(pipe.cameras),
        camera_sha256=pipe.camera_sha256,geometry_file=config.dense_consensus.geometry_file,
        fov=[config.view.fov_x,config.view.fov_y],erp_size=[config.erp.height,config.erp.width],view_size=[config.view.height,config.view.width],
        coverage=gpu_stats,prompt_assignment=config.dense_consensus.prompt_assignment,
        prompt_slot_histogram=dict(sorted(Counter(slots).items())),semantic_band_histogram=dict(sorted(Counter(i//4 for i in slots).items())),
        scheduler=schedule,steps=len(backend.timesteps),guidance_scale=getattr(backend,'cfg_scale',config.generation.guidance_scale),
        runtime_seconds=runtime,total_seconds=time.perf_counter()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        seam=seam_metric(image),aggregate_metrics=result.metrics,stage_seconds=result.stage_seconds,audit=result.audit,
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version()),
        storage_policy='final_result.png and metadata.json only; no intermediate images or tensors')
    if is_no:
        metadata.update(snapshots=snapshots,prompt_indices=slots,
            config_diff_from_L=config_differences(l_baseline['config'],config.to_dict()),
            matched_L_metadata=str(ROOT/'L'/name/'metadata.json'),
            periodic_pyramid_reconstruction=label=='O',
            storage_policy='nine clean-consensus snapshots, terminal final_result.png, one metadata.json; no per-view/tensor artifacts')
        if len(snapshots)!=9:raise AssertionError('Expected nine snapshots for the current schedules')
    else:folder.mkdir(parents=True,exist_ok=False)
    image.save(folder/'final_result.png')
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('COMPLETED',label,name,'seconds',runtime,'calls',result.audit['guided_predictions'],flush=True)
    return folder


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    run_config(load_experiment_config(p.parse_args().config))

if __name__=='__main__':main()
