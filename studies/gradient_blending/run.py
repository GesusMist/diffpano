"""Backend-neutral bounded runner using the unchanged ordinary benchmark loop."""
import argparse
from dataclasses import asdict,replace
import fcntl
import os
import time
import traceback
import numpy as np
import torch
from studies.gradient_blending.common import *
from diffpano.gradient_fusion import GradientSettings,GradientSolveError
from studies.gradient_blending.operator import StudyPipeline
from studies.gradient_blending.diagnostics import Capture,save_rgb

def benchmark_schedule(backend,config):
    from studies.tt_cea.runtime import prepare_schedule
    steps=config.generation.num_inference_steps
    table=tuple(replace(v,eligible=False) for v in prepare_schedule(backend,config,steps))
    assert len(table)==steps,'Prepared interval count differs from benchmark configuration'
    return table

@torch.no_grad()
def run(prompt,mode,instrument=False):
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available(),'GPU Slurm allocation required'
    assert prompt in PROMPTS and mode in MODES
    gate=read(GATES/'validation.json');assert gate['passed'] and gate['core_hashes']==core_hashes()
    smoke=read(GATES/'gpu-smoke.json');assert smoke['passed'] and smoke['core_hashes']==core_hashes()
    if mode!='rgb':
        replay=read(GATES/'offline/replay.json')
        assert replay['passed'] and replay['pilot_lambda_color']==.1 and replay['core_hashes']==core_hashes()
    if instrument:assert (prompt,mode)==('ruins','rgb')
    preserved();sources=source_hashes();baseline=read(OUT/'baseline.json')
    if BACKEND in ('pixeldit','sd35'):
        extension=read(BASE_OUT/'extension-validation.json');backend_gate=read(OUT/'backend-validation.json')
        assert extension['passed'] and extension['source_hashes']==sources
        assert backend_gate['passed'] and backend_gate['source_hashes']==sources
        assert backend_gate['baseline_sha256']==sha(OUT/'baseline.json')
    if SUITE=='flux-scenes20':
        scene_gate=read(OUT/'validation.json')
        assert scene_gate['passed'] and scene_gate['source_hashes']==sources
        assert scene_gate['baseline_sha256']==sha(OUT/'baseline.json')
        assert tuple(scene_gate['prompts'])==PROMPTS
    folder=OUT/'cases'/prompt/mode;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'status.json').exists() and read(folder/'status.json')['state']=='complete':
        assert sha(folder/'final.png')==read(folder/'metadata.json')['artifacts']['final.png'];return
    write(folder/'status.json',dict(state='running',job=os.environ['SLURM_JOB_ID'],mode=mode,prompt=prompt))
    try:
        from studies.original_spherediff.common import seed_global
        from studies.all_prompts.audit import configuration
        from studies.all_prompts.runtime import ordinary_run
        from studies.all_prompts.common import effective_prompt,emit
        from studies.tt_cea.runtime import initialize
        from studies.gwtf_noise.run import runtime_audit,count_calls
        from diffpano.pipelines import build_view_denoiser
        from scripts.generate import _configure_denoiser
        from diffpano.initialization import load_directional_prompts
        from diffpano.planar_pipeline import _negative_prompt
        from diffpano.dense_consensus import prepare_camera_conditioning
        with effective_prompt(prompt) as (prompt_info,prompt_path):
            c,_,cameras,ids,old=configuration(BACKEND,str(prompt_path))
            seed_global(torch,np);load=time.perf_counter()
            b=build_view_denoiser(c);_configure_denoiser(c,b)
            table=benchmark_schedule(b,c)
            bank=b.prepare_prompt_conditioning(load_directional_prompts(str(prompt_path)),_negative_prompt(c))
            conditions,slots=prepare_camera_conditioning(b,bank,cameras,c.dense_consensus.prompt_assignment,1);del bank
            observer=Capture(folder,cameras,sufficient_statistics=instrument)
            settings=GradientSettings(mode=mode)
            p=StudyPipeline(b,cameras,c,settings,ids,observer=observer)
            observer.canvas=p.canvas
            provenance,audit=runtime_audit(c,b,p,conditions,slots,old)
            assert set(audit['differences'])<= {'conditioning_sha256'},audit['differences']
            assert digest(provenance['prepared_schedule'])==baseline['schedule_sha256']
            cached=baseline['cached_cases'][prompt]
            expected_conditioning=None
            for evidence in cached['source_provenance']:
                path=Path(evidence)
                if path.suffix=='.json':
                    record=read(path)
                    if 'conditioning_sha256' in record:expected_conditioning=record['conditioning_sha256'];break
                elif path.suffix=='.out':
                    for line in path.read_text().splitlines():
                        if line.startswith('RUNTIME_PROVENANCE '):
                            record=json.loads(line.split(' ',1)[1])
                            if record.get('row',{}).get('prompt')==prompt and record['row'].get('backend')==BACKEND:
                                expected_conditioning=record['provenance']['conditioning_sha256'];break
                    if expected_conditioning is not None:break
            assert expected_conditioning is not None,'Missing matched conditioning evidence'
            assert provenance['conditioning_sha256']==expected_conditioning,'Matched prompt conditioning changed'
            group=dict(cameras=cameras,ids=ids,conditions=conditions,slots=slots)
            states,initialization,initialization_audit=initialize(b,BACKEND,group,old)
            assert initialization['initial_local_sha256']==baseline['initialization']['initial_local_sha256']
            config_record=dict(backend=BACKEND,suite=SUITE,offline_replay_backend='sana',configuration=c.to_dict(),fusion=asdict(settings),terminal_fusion_mode=mode,
                               prompt=prompt_info,provenance=provenance,initialization=initialization,
                               initialization_audit=initialization_audit,source_hashes=sources,source_revision=baseline['source_revision'],
                               camera_ids=ids,camera_geometry_sha256=provenance['camera_geometry_sha256'],
                               schedule_sha256=digest(provenance['prepared_schedule']),boundary='horizontal periodic, vertical open; pixel-unit differences; binary edge support')
            write(folder/'config.json',config_record)
            setup_seconds=time.perf_counter()-load;setup_peak=torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            def progress(step,total,record):
                emit('GRADIENT_PROGRESS',backend=BACKEND,prompt=prompt,mode=mode,step=step,total=total,
                     elapsed_seconds=time.perf_counter()-start,record=record,fusion=p.canvas.records[-1])
                write(folder/'progress.json',dict(step=step,total=total,elapsed_seconds=time.perf_counter()-start,
                     record=record,fusion=p.canvas.records[-1]))
            with count_calls(b,BACKEND) as counts:
                holder=[states];del states
                native,erp,details=ordinary_run(p,holder.pop(),conditions,table,progress)
            torch.cuda.synchronize();wall_seconds=time.perf_counter()-start
            generation_seconds=wall_seconds-p.canvas.diagnostic_seconds
            peak_allocated=torch.cuda.max_memory_allocated()
            assert counts==expected_counts(BACKEND,c.generation.num_inference_steps,len(cameras)),counts
            assert len(p.canvas.records)==c.generation.num_inference_steps+1 and p.canvas.records[-1]['terminal']
            assert all(r['mode']==mode for r in p.canvas.records)
            if mode!='rgb':assert all(r['converged'] for r in p.canvas.records)
            assert erp.shape==(1,3,2048,4096) and erp.dtype==torch.float32 and bool(torch.isfinite(erp).all())
            save_rgb(folder/'final.png',erp)
            save_rgb(folder/'terminal-rgb-reference.png',p.canvas.last_reference)
            artifacts={name:sha(folder/name) for name in ['final.png','terminal-rgb-reference.png']}
            stage_seconds={}
            for record in p.interval_records:
                for key,value in record['stage_seconds'].items():stage_seconds[key]=stage_seconds.get(key,0)+value
            stage_seconds.update(details['terminal_seconds'])
            preserved();assert sources==source_hashes(),'Study code changed during generation'
            def raw_summary(value):
                return dict(minimum=float(value.min()),maximum=float(value.max()),
                    out_of_range_fraction=float(((value < -1)|(value > 1)).float().mean()),
                    mean_rgb=value.mean((0,2,3)).tolist())
            metadata=dict(**config_record,counts=counts,details=details,intervals=p.interval_records,fusions=p.canvas.records,
                generation_seconds=generation_seconds,instrumented_wall_seconds=wall_seconds,
                final_raw=raw_summary(erp),terminal_reference_raw=raw_summary(p.canvas.last_reference),
                diagnostic_seconds=p.canvas.diagnostic_seconds,stage_seconds=stage_seconds,
                solver_seconds=sum(r.get('solve_seconds',0) for r in p.canvas.records),
                generation_peak_allocated_gib=peak_allocated/2**30,peak_allocated_gib=max(setup_peak,peak_allocated)/2**30,
                setup_seconds=setup_seconds,job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),
                terminal_reference_label='RGB terminal assembly of gradient-run terminal states' if mode!='rgb' else 'RGB terminal assembly of RGB-run terminal states',
                terminal_reference_is_independent_baseline=False,artifacts=artifacts,reused=False,
                instrumentation=instrument,diagnostic_timing_note='Generation excludes bounded diagnostic callbacks and instrument-only statistics; stage timing includes callbacks and lists their total separately.')
            write(folder/'metadata.json',metadata)
            write(folder/'status.json',dict(state='complete',job=os.environ['SLURM_JOB_ID'],image_sha256=artifacts['final.png'],metadata_sha256=sha(folder/'metadata.json')))
            print('CASE_COMPLETE',prompt,mode,generation_seconds,flush=True)
    except BaseException as error:
        write(folder/'status.json',dict(state='failed',job=os.environ['SLURM_JOB_ID'],error=traceback.format_exc(),
                                       solver=getattr(error,'diagnostics',None)))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prompt',choices=PROMPTS,required=True)
    parser.add_argument('--mode',choices=MODES,required=True);parser.add_argument('--instrument',action='store_true')
    a=parser.parse_args();run(a.prompt,a.mode,a.instrument)
