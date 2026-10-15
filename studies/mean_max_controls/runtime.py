"""Confidence-selected controls through the unchanged ordinary_run trajectory."""
from dataclasses import asdict
import time
import torch
import numpy as np
from studies.mean_max_controls.common import *
from studies.mean_max_controls.configuration import resolved
from studies.mean_max_old89.common import effective_prompt
from studies.gradient_blending.operator import StudyPipeline
from studies.gradient_blending.run import benchmark_schedule
from studies.camera_patching.runtime import initialize
from studies.gwtf_noise.run import runtime_audit,count_calls
from studies.all_prompts.common import safe,emit


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
    c,cameras,owners,noise_slots,settings,geometry,old=resolved(row)
    assert safe(asdict(c))==row['semantic']['configuration']
    assert safe(geometry)==row['semantic']['geometry'] and settings.matches_record(row['semantic']['solver'])
    with effective_prompt(row['prompt']) as (info,path):
        assert info==row['semantic']['prompt']
        bank=backend.prepare_prompt_conditioning(load_directional_prompts(str(path)),_negative_prompt(c))
    conditions,prompt_slots=prepare_camera_conditioning(backend,bank,cameras,c.dense_consensus.prompt_assignment,1)
    del bank
    assert list(prompt_slots)==geometry['route']['indices']
    pipeline=StudyPipeline(backend,cameras,c,settings,owners)
    provenance,audit=runtime_audit(c,backend,pipeline,conditions,prompt_slots,old)
    assert set(audit['differences'])<={'conditioning_sha256'},audit['differences']
    assert provenance['camera_geometry_sha256']==geometry['angular_sha256']
    assert safe(provenance['prepared_schedule'])==row['semantic']['schedule']
    assert len(table)==len(backend.timesteps)==20 and all(not item.eligible for item in table)
    return pipeline,conditions,noise_slots,dict(provenance=safe(provenance),audit=safe(audit),geometry=geometry,prompt=info,fusion=settings.to_record())


def raw(tensor):
    return dict(minimum=float(tensor.min()),maximum=float(tensor.max()),
        out_of_range_fraction=float(((tensor<-1)|(tensor>1)).float().mean()),mean_rgb=tensor.mean((0,2,3)).tolist())


@torch.no_grad()
def generate(row,directory,validation):
    from studies.all_prompts.runtime import ordinary_run
    from studies.gradient_blending.diagnostics import save_rgb
    load=time.perf_counter();backend,table=load_backend(row)
    pipeline,conditions,slots,config=setup(row,backend,table)
    expected=validation['groups'][row['prompt']+'/old89']
    assert config['provenance']['conditioning_sha256']==expected['conditioning_sha256']
    states,initialization=initialize(backend,'flux',pipeline.cameras,slots)
    initialization=safe(initialization)
    assert initialization['initial_local_sha256']==expected['initial_local_sha256']
    assert initialization['initial_local_sha256']==resolved(row)[-1]['initialization']['initial_local_sha256']
    assert digest(config['provenance']['prepared_schedule'])==expected['schedule_sha256']
    config.update(gradient_mode=MODE,selection='existing geometric-confidence edge selection',
        initialization=initialization,configuration=row['semantic']['configuration'],
        refinement=dict(last_fraction=0.,coupled_intervals=20,independent_intervals=0),
        camera_count=89,FOV_x=80,FOV_y=80,local_RGB_resolution=[1024,1024],local_native_resolution=[128,128],ERP_resolution=[2048,4096],
        camera_geometry_sha256=config['geometry']['angular_sha256'],schedule_sha256=digest(config['provenance']['prepared_schedule']))
    write(directory/'config.json',config)
    setup_seconds=time.perf_counter()-load;setup_peak=torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    def progress(step,total,record):
        assert record.get('phase')!='independent_refinement'
        fusion=pipeline.canvas.records[-1]
        assert fusion['mode']==MODE and fusion['reference_mode']==row['reference_mode']
        assert fusion.get('gradient_mix_alpha') is None and fusion['converged']
        current=dict(step=step,total=total,elapsed_seconds=time.perf_counter()-start,record=record,fusion=fusion)
        write(directory/'progress.json',safe(current));emit('OLD89_SELECT_PROGRESS',case=row['case_id'],**current)
    with count_calls(backend,'flux') as counts:
        holder=[states];del states
        native,erp,details=ordinary_run(pipeline,holder.pop(),conditions,table,progress)
    torch.cuda.synchronize();wall=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
    assert counts==COUNTS,counts
    assert len(pipeline.canvas.records)==21 and pipeline.canvas.records[-1]['terminal']
    for fusion in pipeline.canvas.records:
        assert fusion['mode']==MODE and fusion['reference_mode']==row['reference_mode']
        assert fusion['converged'] and fusion.get('gradient_mix_alpha') is None
    assert details['backward_calls']==details['replay_count']==details['original_noise_reinjections']==0
    assert erp.shape==(1,3,2048,4096) and erp.dtype==torch.float32 and bool(torch.isfinite(erp).all())
    staging=directory/('staging-'+os.environ['SLURM_JOB_ID']);staging.mkdir(exist_ok=True)
    save_rgb(staging/'final.png',erp);save_rgb(staging/'terminal-rgb-reference.png',pipeline.canvas.last_reference)
    torch.save(erp.detach().cpu().contiguous(),staging/'final-unclamped-fp32.pt')
    torch.save(pipeline.canvas.last_reference.detach().cpu().float().contiguous(),staging/'terminal-rgb-reference-fp32.pt')
    names=['final.png','terminal-rgb-reference.png','final-unclamped-fp32.pt','terminal-rgb-reference-fp32.pt']
    for name in names:os.replace(str(staging/name),str(directory/name))
    staging.rmdir();names.append('config.json')
    stage_seconds={}
    for record in pipeline.interval_records:
        for key,value in record['stage_seconds'].items():stage_seconds[key]=stage_seconds.get(key,0)+value
    stage_seconds.update(details['terminal_seconds'])
    return safe(dict(**config,counts=counts,details=details,intervals=pipeline.interval_records,fusions=pipeline.canvas.records,
        diagnostic_seconds=pipeline.canvas.diagnostic_seconds,terminal_settings=pipeline.canvas.records[-1],
        poisson_solves=sum(fusion.get('converged',False) for fusion in pipeline.canvas.records),generation_seconds=wall,setup_seconds=setup_seconds,
        solver_seconds=sum(fusion.get('solve_seconds',0) for fusion in pipeline.canvas.records),stage_seconds=stage_seconds,
        final_raw=raw(erp),terminal_reference_raw=raw(pipeline.canvas.last_reference),
        generation_peak_allocated_gib=peak/2**30,peak_allocated_gib=max(setup_peak,peak)/2**30,
        artifacts={name:sha(directory/name) for name in names},gpu=torch.cuda.get_device_name(),
        terminal_reference_label='Current completed weighted RGB ERP from THIS run terminal proposals',terminal_reference_is_independent_baseline=False))
