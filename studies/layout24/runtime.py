"""Bookkeeping around unchanged GWTF, StudyPipeline, and ordinary_run."""
from dataclasses import asdict
import fcntl,time
import torch
import numpy as np
from studies.layout24.common import *
from studies.layout24.configuration import resolved
from studies.gradient_blending.operator import StudyPipeline
from studies.gradient_blending.run import benchmark_schedule
from studies.camera_patching.runtime import initialize
from studies.gwtf_noise.run import runtime_audit,count_calls
from studies.all_prompts.common import emit


def load_backend(row):
    from studies.original_spherediff.common import seed_global
    from diffpano.pipelines import build_view_denoiser
    from scripts.generate import _configure_denoiser
    c,*_=resolved(row);seed_global(torch,np)
    backend=build_view_denoiser(c);_configure_denoiser(c,backend)
    return backend,benchmark_schedule(backend,c)


def setup(row,backend,table):
    from diffpano.initialization import load_directional_prompts
    from diffpano.planar_pipeline import _negative_prompt
    from diffpano.dense_consensus import prepare_camera_conditioning
    c,cameras,owners,slots,settings,geometry,old=resolved(row)
    assert safe(asdict(c))==row['semantic']['configuration']
    assert safe(geometry)==row['semantic']['geometry'] and settings.matches_record(row['semantic']['solver'])
    info,_=prompt(row['prompt']);assert info==row['semantic']['prompt']
    bank=backend.prepare_prompt_conditioning(load_directional_prompts(str(prompt_path(row['prompt']))),_negative_prompt(c))
    conditions,prompt_slots=prepare_camera_conditioning(backend,bank,cameras,c.dense_consensus.prompt_assignment,1);del bank
    assert list(prompt_slots)==geometry['route']['indices']
    pipeline=StudyPipeline(backend,cameras,c,settings,owners)
    provenance,audit=runtime_audit(c,backend,pipeline,conditions,prompt_slots,old)
    allowed={'conditioning_sha256','prompt_indices'}
    if row['layout']!='old89':allowed|={'camera_sha256','camera_geometry_sha256'}
    assert set(audit['differences'])<=allowed,audit['differences']
    assert provenance['camera_geometry_sha256']==geometry['angular_sha256']
    assert safe(provenance['prepared_schedule'])==row['semantic']['schedule']
    assert len(table)==len(backend.timesteps)==20 and all(not item.eligible for item in table)
    assert pipeline.bridge_mode=='local_identity_preserving'
    return pipeline,conditions,slots,dict(provenance=safe(provenance),audit=safe(audit),geometry=geometry,prompt=info,fusion=settings.to_record())


def verify_initial(row,initialization):
    expected=row['semantic']['expected_initialization']
    for key in ('initial_local_sha256','source_sha256','source_shape','scaling','scaling_applications'):
        if key in expected:assert safe(initialization[key])==expected[key],(row['layout'],key)
    assert initialization['scaling_applications']==1


def register_group(row,config):
    """Only same prompt/layout must match; no false equality across camera banks."""
    key=row['prompt_identity']+'/'+row['layout'];path=OUT/'runtime-groups'/key
    path.mkdir(parents=True,exist_ok=True)
    value=dict(prompt_sha256=config['prompt']['original_sha256'],conditioning_sha256=config['provenance']['conditioning_sha256'],
        initial_local_sha256=config['initialization']['initial_local_sha256'],source_noise_sha256=config['initialization']['source_sha256'],
        geometry_sha256=config['geometry']['angular_sha256'],noise_slots=config['geometry']['noise_slots'],
        schedule_sha256=config['schedule_sha256'],source_sha256=row['source_sha256'])
    historical_checks=[]
    for target in read(OUT/'generation_plan.json')['rows']:
        if target['state']!='reused' or target['prompt']!=row['prompt']:continue
        source=target['source']
        if source.get('layout')!=row['layout'] or source.get('method')!='diffpano':continue
        if source.get('conditioning_sha256') is not None:
            assert source['conditioning_sha256']==value['conditioning_sha256'],'Historical RGB/select conditioning differs'
            historical_checks.append(dict(case_id=target['case_id'],conditioning_sha256=source['conditioning_sha256']))
    with (path/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (path/'identity.json').exists():assert read(path/'identity.json')==value,'RGB/select shared-state identity changed'
        else:write(path/'identity.json',value)
    return dict(**value,historical_conditioning_checks=historical_checks)


def raw(x):
    return dict(minimum=float(x.min()),maximum=float(x.max()),out_of_range_fraction=float(((x<-1)|(x>1)).float().mean()),mean_rgb=x.mean((0,2,3)).tolist())


@torch.no_grad()
def generate(row,directory,validation=None):
    from studies.all_prompts.runtime import ordinary_run
    from studies.gradient_blending.diagnostics import save_rgb
    load=time.perf_counter();backend,table=load_backend(row)
    pipeline,conditions,slots,config=setup(row,backend,table)
    states,initialization=initialize(backend,'flux',pipeline.cameras,slots);initialization=safe(initialization)
    verify_initial(row,initialization)
    config.update(initialization=initialization,configuration=row['semantic']['configuration'],
        refinement=dict(last_fraction=0.,coupled_intervals=20,independent_intervals=0),
        camera_count=row['camera_count'],FOV_x=80,FOV_y=80,local_RGB_resolution=[1024,1024],local_native_resolution=[128,128],ERP_resolution=[2048,4096],
        camera_geometry_sha256=config['geometry']['angular_sha256'],schedule_sha256=digest(config['provenance']['prepared_schedule']))
    config['same_layout_group_identity']=register_group(row,config)
    write(directory/'config.json',config)
    setup_seconds=time.perf_counter()-load;setup_peak=torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    def progress(step,total,record):
        assert record.get('phase')!='independent_refinement'
        assert record['min_contributors']>0 and record['minimum_weight_sum']>0
        fusion=pipeline.canvas.records[-1]
        assert fusion['mode']==row['gradient_mode'] and fusion.get('gradient_mix_alpha') is None
        if row['gradient_mode']=='poisson_select':assert fusion['converged'] and fusion['reference_mode']=='screened'
        current=dict(step=step,total=total,elapsed_seconds=time.perf_counter()-start,record=record,fusion=fusion)
        write(directory/'progress.json',safe(current));emit('LAYOUT24_PROGRESS',case=row['case_id'],**current)
    with count_calls(backend,'flux') as counts:
        holder=[states];del states
        native,erp,details=ordinary_run(pipeline,holder.pop(),conditions,table,progress)
    torch.cuda.synchronize();wall=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
    assert counts==expected_counts(row),counts
    assert details['guided_predictions']==counts['denoiser']
    assert len(pipeline.canvas.records)==21 and pipeline.canvas.records[-1]['terminal']
    for fusion in pipeline.canvas.records:
        assert fusion['mode']==row['gradient_mode'] and fusion.get('gradient_mix_alpha') is None
        if row['gradient_mode']=='poisson_select':assert fusion['converged'] and fusion['reference_mode']=='screened'
    assert details['backward_calls']==details['replay_count']==details['original_noise_reinjections']==0
    assert erp.shape==(1,3,2048,4096) and erp.dtype==torch.float32 and bool(torch.isfinite(erp).all())
    staging=directory/('staging-'+os.environ['SLURM_JOB_ID']);staging.mkdir(exist_ok=True)
    save_rgb(staging/'final.png',erp);torch.save(erp.detach().cpu().contiguous(),staging/'final-unclamped-fp32.pt')
    names=['final.png','final-unclamped-fp32.pt']
    for name in names:os.replace(str(staging/name),str(directory/name))
    staging.rmdir();names.append('config.json')
    stage_seconds={}
    for record in pipeline.interval_records:
        for key,value in record['stage_seconds'].items():stage_seconds[key]=stage_seconds.get(key,0)+value
    stage_seconds.update(details['terminal_seconds'])
    return safe(dict(**config,counts=counts,details=details,intervals=pipeline.interval_records,fusions=pipeline.canvas.records,
        poisson_solves=21 if row['gradient_mode']=='poisson_select' else 0,generation_seconds=wall,setup_seconds=setup_seconds,
        solver_seconds=sum(f.get('solve_seconds',0) for f in pipeline.canvas.records),stage_seconds=stage_seconds,final_raw=raw(erp),
        generation_peak_allocated_gib=peak/2**30,peak_allocated_gib=max(setup_peak,peak)/2**30,
        artifacts={name:sha(directory/name) for name in names},gpu=torch.cuda.get_device_name()))
