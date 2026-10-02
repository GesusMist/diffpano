"""Explicit new camera tuples through the frozen ordinary consensus pipeline."""
import time
from dataclasses import replace
import torch
from .common import *
from .cameras import stable_ids,routing,angular_hash
from studies.all_prompts.audit import configuration
from studies.all_prompts.runtime import ordinary_run
from studies.tt_cea.pipeline import ExperimentalPipeline
from studies.tt_cea.runtime import prepare_schedule
from studies.gwtf_noise.run import count_calls,runtime_audit
from diffpano.erp_noise_initialization import native_cameras,primary_noise_size,states_digest
from diffpano.gwtf_noise_initialization import initialize_gwtf_shared_noise,GWTFlowNoiseConfig


def setup(backend,prompt_path):
    import numpy as np
    from studies.original_spherediff.common import seed_global
    from diffpano.pipelines import build_view_denoiser
    from scripts.generate import _configure_denoiser
    from diffpano.initialization import load_directional_prompts
    from diffpano.planar_pipeline import _negative_prompt
    c,_,_,_,old=configuration(backend,str(prompt_path))
    seed_global(torch,np)
    b=build_view_denoiser(c);_configure_denoiser(c,b)
    table=tuple(replace(v,eligible=False) for v in prepare_schedule(b,c,STEPS[backend]))
    assert len(table)==len(b.timesteps)==STEPS[backend]
    assert b.dtype==torch.bfloat16
    bank=b.prepare_prompt_conditioning(load_directional_prompts(str(prompt_path)),_negative_prompt(c))
    return c,b,old,table,bank


def group(row,c,b,old,bank):
    from diffpano.dense_consensus import prepare_camera_conditioning
    from scripts.dense_erp_experiment import canonical_schedule,schedule_metadata
    cams=requested_camera(row,c.view);m=len(cams)
    ids=stable_ids(row['strategy'],m,layout_seed=row['layout_seed'] or 0)
    conditions,slots=prepare_camera_conditioning(b,bank,cams,c.dense_consensus.prompt_assignment,1)
    route=routing(cams,prompt_record(REPO/'prompts'/(row['prompt']+'.txt'))[0]['effective_lines'])
    assert slots==route['indices'] and len(conditions)==m
    p=ExperimentalPipeline(b,cams,c,projection=row['projection'])
    provenance,audit=runtime_audit(c,b,p,conditions,slots,old)
    allowed={'camera_sha256','camera_geometry_sha256','conditioning_sha256','prompt_indices'}
    assert set(audit['differences'])<=allowed,audit['differences']
    assert safe(canonical_schedule(schedule_metadata(b)))==safe(old['prepared_schedule'])
    assert angular_hash(cams)==provenance['camera_geometry_sha256']
    assert p.canvas.spec.projection==row['projection']
    if row['backend']=='pixeldit':
        assert b.cfg_scale==c.pixeldit.cfg_scale and list(b.interval_guidance)==c.pixeldit.interval_guidance
    else:
        assert b.guidance_scale==c.generation.guidance_scale
        if row['backend']=='flux':assert b.true_cfg_scale==c.generation.true_cfg_scale
    emit('RUNTIME_AUDIT',row=row,provenance=provenance,allowed_differences=sorted(allowed),
        actual_differences=sorted(audit['differences']),semantic_band_counts=route['semantic_band_counts'],
        actual_guidance=b.cfg_scale if row['backend']=='pixeldit' else b.guidance_scale,
        camera_ids=ids,unchanged_configuration_sha256=digest(c.to_dict()),
        checkpoint_provenance=audit['checkpoint_provenance'])
    return p,conditions,ids,provenance,route


@torch.no_grad()
def initialize(b,name,cameras,ids,*,reverse=False):
    m=len(cameras);h,w=primary_noise_size(native_cameras(b,cameras))
    rng=torch.random.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all()
    with count_calls(b,name,forbid=True) as calls:
        states,record=initialize_gwtf_shared_noise(b,cameras,GWTFlowNoiseConfig(h,w,0),1,camera_slots=ids,
            execution_order=list(reversed(range(m))) if reverse else None)
    assert calls==dict(denoiser=0,encode=0,decode=0,initialize=m)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(a,z) for a,z in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    assert len(states)==m and states_digest(states)==record['initial_local_sha256']
    for state,cam in zip(states,cameras):
        assert state.shape==(1,b.native_channels,*b.native_spatial_shape_for_rgb(cam.height,cam.width))
        assert state.dtype==torch.float32 and state.device.type=='cpu' and bool(torch.isfinite(state).all())
    assert record['scaling_applications']==1 and record['denoiser_calls']==record['vae_calls']==0
    emit('INITIALIZATION',backend=name,angular_geometry_sha256=angular_hash(cameras),initial_local_sha256=record['initial_local_sha256'],
        source_shape=record['source_shape'],source_sha256=record['source_sha256'],calls=calls,scaling=record['scaling'],scaling_applications=1)
    return states,record


def compact_config(row,c,b,p,provenance,route,init,details,counts,source,seconds):
    cams=p.cameras;native=b.native_spatial_shape_for_rgb(c.view.height,c.view.width)
    precision=str(b.dtype).replace('torch.','')
    g=dict(seed=0,num_inference_steps=len(b.timesteps),guidance_scale=float(b.cfg_scale if row['backend']=='pixeldit' else b.guidance_scale),
        true_cfg_scale=float(getattr(b,'true_cfg_scale',c.generation.true_cfg_scale)),batch_size=1,
        prepared_schedule_sha256=digest(safe(provenance['prepared_schedule'])))
    backend=dict(name=row['backend'],model_id=provenance['model_checkpoint'],model_revision=provenance['model_revision'],precision=precision,
        local_rgb_height=c.view.height,local_rgb_width=c.view.width,native_channels=b.native_channels,native_height=native[0],native_width=native[1],
        vae_tiling=bool(getattr(getattr(getattr(b,'pipeline',None),'vae',None),'use_tiling',False)),model_cpu_offload=c.model.cpu_offload)
    if row['backend']=='pixeldit':
        backend.update(config_path=c.pixeldit.config_path,config_sha256=sha(c.pixeldit.config_path))
        g.update(negative_prompt=b.negative_prompt,interval_guidance=list(b.interval_guidance),flow_shift=float(b.solver.flow_shift),solver_class=type(b.solver).__name__)
    return dict(study='camera_patching',method='diffpano',
        camera=dict(strategy=row['strategy'],num_cameras=len(cams),layout_seed=row['layout_seed'],fibonacci_phase=row['fibonacci_phase'],
            angular_geometry_sha256=angular_hash(cams),fov_x_deg=cams[0].fov_x,fov_y_deg=cams[0].fov_y,fixed_during_trajectory=True,
            roll=0.,semantic_band_counts=route['semantic_band_counts'],prompt_assignment_sha256=digest(route['indices'])),
        projection=dict(consensus_canvas=p.canvas.spec.projection,height=p.canvas.spec.height,width=p.canvas.spec.width,final_export='erp',final_width=4096,final_height=2048),
        backend=backend,generation=g,prompt=dict(name=row['prompt'],path='prompts/'+row['prompt']+'.txt',sha256=sha(REPO/'prompts'/(row['prompt']+'.txt'))),
        initialization=dict(method='gwtflow',source_seed=0,initial_local_sha256=init['initial_local_sha256'],source_shape=init['source_shape'],
            source_sha256=init['source_sha256'],camera_slots_policy='strategy:N:index' if row['strategy']=='fibonacci' else 'strategy:N:layout_seed:index',
            native_scaling=init['scaling'],scaling_applications=1,transport_dtype='CPU FP32',denoiser_calls=0,vae_calls=0),
        fusion=dict(mode=c.fusion.mode,weight_mode=c.fusion.weight_mode,temperature=c.fusion.spherediff_temperature,warp_mode=c.warp.mode,
            perspective_to_canvas_interpolation=c.warp.perspective_to_erp.interpolation,canvas_to_perspective_interpolation=c.warp.erp_to_perspective.interpolation),
        bridge=dict(mode='not_applicable_pixeldit' if row['backend']=='pixeldit' else p.bridge_mode),
        trajectory=dict(transition=c.consensus_transition.mode,time_travel=details['time_travel_enabled'],replay_count=details['replay_count'],
            backward_count=details['backward_calls'],original_noise_reinjection_count=details['original_noise_reinjections'],counts=counts,
            final_cea_export_count=1 if row['projection']=='cea' else 0,original_noise_bank_retained=False),
        source={k:source[k] for k in ('git_commit','git_dirty','fingerprint')},
        execution=dict(job_id=os.environ['SLURM_JOB_ID'],array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
            generation_seconds=seconds,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30))


@torch.no_grad()
def generate(row,prompt_path,source,preflight):
    from diffpano.diagnostics import tensor_to_pil
    c,b,old,table,bank=setup(row['backend'],prompt_path)
    p,conditions,ids,provenance,route=group(row,c,b,old,bank);del bank
    states,init=initialize(b,row['backend'],p.cameras,ids)
    expected_init=preflight['initialization_hashes'][row['strategy']]
    assert init['initial_local_sha256']==expected_init,'Preflight/prompt/projection initialization mismatch'
    started=time.perf_counter()
    def progress(step,total,record):
        assert record['min_contributors']>0 and record['minimum_weight_sum']>0
        emit('PROGRESS',row=row,step=step,total=total,diagnostic=record)
    with count_calls(b,row['backend']) as counts:
        holder=[states];del states
        native,erp,details=ordinary_run(p,holder.pop(),conditions,table,progress)
    assert counts==expected_counts(len(p.cameras),len(table),row['backend']=='pixeldit')
    assert details['guided_predictions']==counts['denoiser']
    assert details['backward_calls']==details['replay_count']==details['original_noise_reinjections']==0
    assert erp.shape==(1,3,2048,4096) and bool(torch.isfinite(erp).all())
    assert angular_hash(p.cameras)==provenance['camera_geometry_sha256']
    seconds=time.perf_counter()-started
    record=compact_config(row,c,b,p,provenance,route,init,details,counts,source,seconds)
    emit('GENERATION_VALIDATED',row=row,counts=counts,details=details,generation_seconds=seconds)
    return tensor_to_pil(erp[0].cpu()),record
