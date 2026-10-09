"""Five explicitly authorized tolerance-only reruns of the frozen study runner."""
import argparse,fcntl,os,time,traceback
from dataclasses import asdict
import numpy as np
import torch
from studies.gradient_refinement_retry.common import *
from diffpano.gradient_fusion import GradientSettings
from diffpano.refinement import RefinementConfig,resolved_refinement
from studies.gradient_blending.operator import StudyPipeline
from studies.gradient_blending.common import expected_counts
from studies.gradient_blending.diagnostics import save_rgb
from studies.gradient_blending.run import benchmark_schedule

@torch.no_grad()
def run(index,retry=False):
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available(),'GPU Slurm allocation required'
    assert index in INDICES, 'Only the five explicitly requested failed cases may use this runner'
    manifest=read(OUT/'manifest.json');row=manifest['rows'][index];sources=retry_source_hashes()
    plan=read(RETRY/'plan.json');check_scope(plan)
    assert row==next(r for r in plan['rows'] if r['index']==index)
    assert manifest['source_hashes']==original_source_hashes(),'Original implementation differs'
    for gate in ('validation.json','gpu-smoke.json'):
        record=read(OUT/gate);assert record['passed'] and record['source_hashes']==manifest['source_hashes'],gate+' is stale or failed'
    gate=read(RETRY/'validation.json')
    assert gate['passed'] and gate['source_hashes']==sources and gate['plan_sha256']==sha(RETRY/'plan.json')
    attempt=next(r for r in plan['failed_attempts'] if r['index']==index)
    for path,expected in attempt['archived_artifacts'].items():assert sha(path)==expected,'Archived failure changed'
    preserved();directory=folder(row);directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if case_complete(row):print('VALID_CACHE',row['key'],flush=True);return
    if (directory/'status.json').exists() and not retry:raise RuntimeError('Incomplete/failed case requires explicit --retry')
    job=os.environ['SLURM_JOB_ID']
    write(directory/'status.json',dict(state='running',job=job,case=row['key'],semantic_sha256=row['semantic_sha256']))
    try:
        from studies.original_spherediff.common import seed_global
        from studies.all_prompts.audit import configuration
        from studies.all_prompts.runtime import ordinary_run
        from studies.all_prompts.common import effective_prompt,emit,safe
        from studies.tt_cea.runtime import initialize
        from studies.gwtf_noise.run import runtime_audit,count_calls
        from diffpano.pipelines import build_view_denoiser
        from scripts.generate import _configure_denoiser
        from diffpano.initialization import load_directional_prompts
        from diffpano.planar_pipeline import _negative_prompt
        from diffpano.dense_consensus import prepare_camera_conditioning
        with effective_prompt(row['prompt']) as (prompt_info,prompt_path):
            c,_,cameras,ids,old=configuration('flux',str(prompt_path));c.global_pipeline.refinement=RefinementConfig(row['last_fraction'])
            assert asdict(c)==row['semantic']['configuration'] and prompt_info==row['semantic']['prompt']
            seed_global(torch,np);load=time.perf_counter();b=build_view_denoiser(c);_configure_denoiser(c,b)
            table=benchmark_schedule(b,c)
            bank=b.prepare_prompt_conditioning(load_directional_prompts(str(prompt_path)),_negative_prompt(c))
            conditions,slots=prepare_camera_conditioning(b,bank,cameras,c.dense_consensus.prompt_assignment,1);del bank
            settings=GradientSettings(**row['semantic']['fusion']);assert asdict(settings)==row['semantic']['fusion']
            p=StudyPipeline(b,cameras,c,settings,ids)
            provenance,audit=runtime_audit(c,b,p,conditions,slots,old)
            assert set(audit['differences'])<={'conditioning_sha256'},audit['differences']
            assert provenance['conditioning_sha256']==row['semantic']['conditioning_sha256']
            assert safe(provenance['prepared_schedule'])==row['semantic']['schedule']
            states,initialization,initialization_audit=initialize(b,'flux',dict(cameras=cameras,ids=ids,conditions=conditions,slots=slots),old)
            assert initialization['initial_local_sha256']==row['semantic']['initial_native_sha256']
            assert initialization_audit['calls']==dict(denoiser=0,encode=0,decode=0,initialize=89)
            plan=resolved_refinement(b.timesteps,c.global_pipeline.refinement)
            config_record=dict(case={k:v for k,v in row.items() if k!='semantic'},semantic_sha256=row['semantic_sha256'],configuration=asdict(c),
                backend='flux',prompt=prompt_info,fusion=asdict(settings),reference_mode=row['reference_mode'],refinement=plan,
                terminal_fusion_mode=row['gradient_mode'],terminal_reference_mode=row['reference_mode'],
                provenance=safe(provenance),initialization=safe(initialization),initialization_audit=safe(initialization_audit),
                source_hashes=sources,implementation_revision=manifest['implementation_revision'],
                tolerance_retry=dict(plan=str(RETRY/'plan.json'),plan_sha256=sha(RETRY/'plan.json'),previous_job=attempt['previous_job'],previous_relative_tolerance=1e-5,relative_tolerance=RTOL,absolute_tolerance=1e-7),
                historical_source_revision=old.get('source_revision'),camera_ids=ids,camera_count=len(cameras),FOV_x=80,FOV_y=80,
                camera_geometry_sha256=provenance['camera_geometry_sha256'],local_RGB_resolution=[1024,1024],local_native_resolution=[128,128],
                schedule_sha256=digest(safe(provenance['prepared_schedule'])),boundary='horizontal periodic, vertical open; pixel-unit differences; binary edge support')
            write(directory/'config.json',config_record)
            setup_seconds=time.perf_counter()-load;setup_peak=torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            def progress(step,total,record):
                independent=record.get('phase')=='independent_refinement'
                current=dict(step=step,total=total,phase='independent_refinement' if independent else 'coupled',
                    elapsed_seconds=time.perf_counter()-start,record=record,fusion=None if independent else p.canvas.records[-1])
                write(directory/'progress.json',current);emit('REFINEMENT_PROGRESS',case=row['key'],**current)
            with count_calls(b,'flux') as counts:
                holder=[states];del states
                native,erp,details=ordinary_run(p,holder.pop(),conditions,table,progress)
            torch.cuda.synchronize();wall=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
            assert counts==expected_counts('flux',20,89,row['last_fraction']),counts
            assert len(p.canvas.records)==plan['coupled_intervals']+1 and p.canvas.records[-1]['terminal']
            assert all(r['mode']==row['gradient_mode'] and r['reference_mode']==row['reference_mode'] and r['converged'] for r in p.canvas.records)
            assert erp.shape==(1,3,2048,4096) and erp.dtype==torch.float32 and bool(torch.isfinite(erp).all())
            save_rgb(directory/'final.png',erp);save_rgb(directory/'terminal-rgb-reference.png',p.canvas.last_reference)
            torch.save(erp.detach().cpu().contiguous(),directory/'final-unclamped-fp32.pt')
            torch.save(p.canvas.last_reference.detach().cpu().float().contiguous(),directory/'terminal-rgb-reference-fp32.pt')
            names=['final.png','terminal-rgb-reference.png','final-unclamped-fp32.pt','terminal-rgb-reference-fp32.pt']
            artifacts={name:sha(directory/name) for name in names};stage_seconds={}
            for record in p.interval_records:
                for key,val in record['stage_seconds'].items():stage_seconds[key]=stage_seconds.get(key,0)+val
            stage_seconds.update(details['terminal_seconds'])
            assert sources==retry_source_hashes(),'Implementation changed during generation';preserved()
            def raw(x):return dict(minimum=float(x.min()),maximum=float(x.max()),out_of_range_fraction=float(((x<-1)|(x>1)).float().mean()),mean_rgb=x.mean((0,2,3)).tolist())
            metadata=dict(**config_record,counts=counts,details=details,intervals=p.interval_records,fusions=p.canvas.records,
                generation_seconds=wall,setup_seconds=setup_seconds,solver_seconds=sum(r.get('solve_seconds',0) for r in p.canvas.records),
                stage_seconds=stage_seconds,final_raw=raw(erp),terminal_reference_raw=raw(p.canvas.last_reference),
                generation_peak_allocated_gib=peak/2**30,peak_allocated_gib=max(setup_peak,peak)/2**30,
                artifacts=artifacts,job=job,gpu=torch.cuda.get_device_name(),reused=False,
                terminal_reference_label='Ordinary weighted RGB assembly reference from THIS run terminal proposals',terminal_reference_is_independent_baseline=False)
            write(directory/'metadata.json',safe(metadata))
            write(directory/'status.json',dict(state='complete',job=job,semantic_sha256=row['semantic_sha256'],image_sha256=artifacts['final.png'],metadata_sha256=sha(directory/'metadata.json')))
            print('CASE_COMPLETE',row['key'],counts,'seconds',wall,flush=True)
    except BaseException as error:
        write(directory/'status.json',dict(state='failed',job=job,semantic_sha256=row['semantic_sha256'],error=traceback.format_exc(),solver=getattr(error,'diagnostics',None)))
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=None);p.add_argument('--retry',action='store_true');a=p.parse_args()
    run(int(os.environ['SLURM_ARRAY_TASK_ID']) if a.index is None else a.index,a.retry)
