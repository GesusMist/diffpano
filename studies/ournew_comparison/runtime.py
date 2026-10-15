"""Suite bookkeeping around the existing backend, GWTF and ordinary_run paths."""
from dataclasses import asdict
import time
import torch
import numpy as np
from studies.ournew_comparison.common import *
from studies.ournew_comparison.configuration import resolved
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
    b=build_view_denoiser(c);_configure_denoiser(c,b)
    table=benchmark_schedule(b,c)
    return b,table


def setup(row,b,table):
    from diffpano.initialization import load_directional_prompts
    from diffpano.planar_pipeline import _negative_prompt
    from diffpano.dense_consensus import prepare_camera_conditioning
    c,cameras,owners,noise_slots,settings,geometry,old=resolved(row)
    assert safe(asdict(c))==row['semantic']['configuration']
    assert safe(geometry)==row['semantic']['geometry'] and settings.matches_record(row['semantic']['solver'])
    info,path=prompt(row['prompt']);assert info==row['semantic']['prompt']
    bank=b.prepare_prompt_conditioning(load_directional_prompts(str(path)),_negative_prompt(c))
    conditions,prompt_slots=prepare_camera_conditioning(b,bank,cameras,c.dense_consensus.prompt_assignment,1);del bank
    assert list(prompt_slots)==geometry['route']['indices']
    p=StudyPipeline(b,cameras,c,settings,owners)
    provenance,audit=runtime_audit(c,b,p,conditions,prompt_slots,old)
    allowed={'conditioning_sha256'}
    if row['layout']=='uniform89':allowed|={'camera_sha256','camera_geometry_sha256','prompt_indices'}
    assert set(audit['differences'])<=allowed,audit['differences']
    assert provenance['camera_geometry_sha256']==geometry['angular_sha256']
    assert safe(provenance['prepared_schedule'])==row['semantic']['schedule']
    assert len(table)==len(b.timesteps)==20 and all(not t.eligible for t in table)
    return p,conditions,noise_slots,dict(provenance=safe(provenance),audit=safe(audit),geometry=geometry,prompt=info,fusion=settings.to_record())


def initial(b,row,p,slots):
    states,record=initialize(b,'flux',p.cameras,slots)
    if row['layout']=='old89':
        old=resolved(row)[-1]
        assert record['initial_local_sha256']==old['initialization']['initial_local_sha256']
    return states,safe(record)


def raw(x):
    return dict(minimum=float(x.min()),maximum=float(x.max()),out_of_range_fraction=float(((x<-1)|(x>1)).float().mean()),mean_rgb=x.mean((0,2,3)).tolist())


@torch.no_grad()
def generate(row,directory,validation):
    from studies.all_prompts.runtime import ordinary_run
    from studies.gradient_blending.diagnostics import save_rgb
    load=time.perf_counter();b,table=load_backend(row);p,conditions,slots,config=setup(row,b,table)
    expected=validation['groups'][row['prompt']+'/'+row['layout']]
    assert config['provenance']['conditioning_sha256']==expected['conditioning_sha256']
    states,init=initial(b,row,p,slots)
    assert init['initial_local_sha256']==expected['initial_local_sha256']
    config.update(initialization=init,configuration=row['semantic']['configuration'],refinement=dict(last_fraction=0.,coupled_intervals=20,independent_intervals=0),
        camera_count=89,FOV_x=80,FOV_y=80,local_RGB_resolution=[1024,1024],local_native_resolution=[128,128],
        camera_geometry_sha256=config['geometry']['angular_sha256'],schedule_sha256=digest(config['provenance']['prepared_schedule']))
    write(directory/'config.json',config)
    setup_seconds=time.perf_counter()-load;setup_peak=torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    def progress(step,total,record):
        assert record.get('phase')!='independent_refinement'
        current=dict(step=step,total=total,elapsed_seconds=time.perf_counter()-start,record=record,fusion=p.canvas.records[-1])
        write(directory/'progress.json',safe(current));emit('OURNEW_PROGRESS',case=row['case_id'],**current)
    with count_calls(b,'flux') as counts:
        holder=[states];del states
        native,erp,details=ordinary_run(p,holder.pop(),conditions,table,progress)
    torch.cuda.synchronize();wall=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
    assert counts==dict(denoiser=1780,encode=3560,decode=1869,initialize=0),counts
    assert len(p.canvas.records)==21 and p.canvas.records[-1]['terminal']
    for f in p.canvas.records:
        assert f['mode']==row['gradient_mode'] and f['reference_mode']==config['fusion']['reference_mode']
        if row['gradient_mode']!='rgb':assert f['converged']
    assert details['backward_calls']==details['replay_count']==details['original_noise_reinjections']==0
    assert erp.shape==(1,3,2048,4096) and erp.dtype==torch.float32 and bool(torch.isfinite(erp).all())
    staging=directory/('staging-'+os.environ['SLURM_JOB_ID']);staging.mkdir(exist_ok=True)
    save_rgb(staging/'final.png',erp);save_rgb(staging/'terminal-rgb-reference.png',p.canvas.last_reference)
    torch.save(erp.detach().cpu().contiguous(),staging/'final-unclamped-fp32.pt')
    torch.save(p.canvas.last_reference.detach().cpu().float().contiguous(),staging/'terminal-rgb-reference-fp32.pt')
    for name in ('final.png','terminal-rgb-reference.png','final-unclamped-fp32.pt','terminal-rgb-reference-fp32.pt'):
        os.replace(str(staging/name),str(directory/name))
    staging.rmdir()
    names=['final.png','terminal-rgb-reference.png','final-unclamped-fp32.pt','terminal-rgb-reference-fp32.pt','config.json']
    stage_seconds={}
    for record in p.interval_records:
        for key,value in record['stage_seconds'].items():stage_seconds[key]=stage_seconds.get(key,0)+value
    stage_seconds.update(details['terminal_seconds'])
    return safe(dict(**config,counts=counts,details=details,intervals=p.interval_records,fusions=p.canvas.records,
        poisson_solves=0 if row['gradient_mode']=='rgb' else 21,generation_seconds=wall,setup_seconds=setup_seconds,
        solver_seconds=sum(r.get('solve_seconds',0) for r in p.canvas.records),stage_seconds=stage_seconds,
        final_raw=raw(erp),terminal_reference_raw=raw(p.canvas.last_reference),
        generation_peak_allocated_gib=peak/2**30,peak_allocated_gib=max(setup_peak,peak)/2**30,
        artifacts={n:sha(directory/n) for n in names},gpu=torch.cuda.get_device_name(),
        terminal_reference_label='Current completed weighted RGB ERP from THIS run terminal proposals',terminal_reference_is_independent_baseline=False))
