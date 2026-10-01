"""Ordinary validated RGB-consensus intervals; no time-travel execution path."""
import time
import torch
from diffpano.erp_noise_initialization import states_digest
from diffpano.trajectory import conditioning_digest
from studies.all_prompts.common import emit

@torch.no_grad()
def ordinary_run(pipeline,states,conditions,intervals,progress=None):
    b=pipeline.backend;schedule=b.timesteps.clone()
    condition_hashes=[conditioning_digest(x) for x in conditions]
    initial_hash=states_digest(states);calls_before=getattr(b,'guided_prediction_count',0)
    # Rebind and release each previous noisy state list. No initial noise bank.
    for interval in intervals:
        states,summary=pipeline.advance_interval(states,conditions,interval,pass_kind='initial')
        assert summary['pass_kind']=='initial'
        if progress:progress(interval.k+1,len(intervals),summary)
    assert torch.equal(schedule,b.timesteps)
    assert condition_hashes==[conditioning_digest(x) for x in conditions]
    assert b.guided_prediction_count-calls_before==len(pipeline.cameras)*len(intervals)
    terminal_hash=states_digest(states)
    native,erp,terminal=pipeline.terminal(states)
    return native,erp,dict(initial_state_sha256=initial_hash,terminal_state_sha256=terminal_hash,
        guided_predictions=b.guided_prediction_count-calls_before,time_travel_enabled=False,
        backward_calls=0,replay_count=0,original_noise_reinjections=0,additional_denoising_cycles=0,
        original_noise_bank_retained=False,terminal_seconds=terminal,
        active_consensus_projection=pipeline.canvas.spec.projection,
        terminal_semantics='decode terminal local states; fuse active RGB canvas; CEA exports once to ERP')

def diffpano(row,prompt_path):
    import numpy as np
    from dataclasses import replace
    from studies.original_spherediff.common import seed_global
    from studies.all_prompts.audit import configuration
    from studies.tt_cea.pipeline import ExperimentalPipeline
    from studies.tt_cea.runtime import prepare_schedule,initialize
    from studies.gwtf_noise.run import count_calls,runtime_audit
    from diffpano.pipelines import build_view_denoiser
    from scripts.generate import _configure_denoiser
    from diffpano.initialization import load_directional_prompts
    from diffpano.planar_pipeline import _negative_prompt
    from diffpano.dense_consensus import prepare_camera_conditioning
    from scripts.dense_erp_experiment import canonical_schedule,schedule_metadata
    from diffpano.diagnostics import tensor_to_pil
    from studies.all_prompts.common import safe

    n=row['backend'];c,_,cams,ids,old=configuration(n,str(prompt_path))
    seed_global(torch,np)
    load=time.perf_counter();b=build_view_denoiser(c);_configure_denoiser(c,b)
    table=prepare_schedule(b,c,row['steps'])
    # The eligibility marker is irrelevant to the ordinary loop; explicitly disable it.
    table=tuple(replace(x,eligible=False) for x in table)
    bank=b.prepare_prompt_conditioning(load_directional_prompts(str(prompt_path)),_negative_prompt(c))
    conditions,slots=prepare_camera_conditioning(b,bank,cams,c.dense_consensus.prompt_assignment,1);del bank
    p=ExperimentalPipeline(b,cams,c,projection=row['projection'])
    provenance,audit=runtime_audit(c,b,p,conditions,slots,old)
    assert set(audit['differences'])<= {'conditioning_sha256'},audit['differences']
    assert safe(canonical_schedule(schedule_metadata(b)))==safe(old['prepared_schedule'])
    emit('RUNTIME_PROVENANCE',row=row,configuration=c.to_dict(),provenance=provenance,
        checkpoint_provenance=audit['checkpoint_provenance'],model_loading_and_prompt_seconds=time.perf_counter()-load,
        time_travel_enabled=False,old89=True,
        pixeldit_effective_cfg=getattr(b,'cfg_scale',None),pixeldit_interval_guidance=getattr(b,'interval_guidance',None),
        pixeldit_flow_shift=getattr(getattr(b,'solver',None),'flow_shift',None))
    group=dict(cameras=cams,ids=ids,conditions=conditions,slots=slots)
    states,initialization,initialization_audit=initialize(b,n,group,old)
    assert initialization['initial_local_sha256']==old['initialization']['initial_local_sha256']
    assert initialization_audit['calls']==dict(denoiser=0,encode=0,decode=0,initialize=89)
    emit('INITIALIZATION',row=row,record=initialization,audit=initialization_audit,
        ERP_CEA_expected_identical_sha256=old['initialization']['initial_local_sha256'])
    setup_peak=torch.cuda.max_memory_allocated();setup_reserved=torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    def progress(step,total,record):emit('PROGRESS',index=row['index'],backend=n,projection=row['projection'],
        prompt=row['prompt'],step=step,total=total,elapsed_seconds=time.perf_counter()-start,diagnostic=record)
    with count_calls(b,n) as counts:
        # Transfer ownership of states into the ordinary loop: no persistent original-noise copy.
        holder=[states];del states
        native,erp,details=ordinary_run(p,holder.pop(),conditions,table,progress)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    expected=89*row['steps']
    assert details['guided_predictions']==counts['denoiser']==expected
    assert counts==dict(denoiser=expected,encode=0 if n=='pixeldit' else 2*expected,
        decode=0 if n=='pixeldit' else expected+89,initialize=0)
    assert details['replay_count']==details['backward_calls']==details['original_noise_reinjections']==details['additional_denoising_cycles']==0
    assert tuple(erp.shape)==(1,3,2048,4096) and bool(torch.isfinite(erp).all())
    assert native.rgb.dtype==torch.float32
    image=tensor_to_pil(erp[0].detach().cpu())
    emit('GENERATION_VALIDATED',row=row,counts=counts,details=details,generation_seconds=seconds,
        peak_allocated_gib=max(setup_peak,torch.cuda.max_memory_allocated())/2**30,
        peak_reserved_gib=max(setup_reserved,torch.cuda.max_memory_reserved())/2**30,
        active_projection_each_interval=p.canvas.spec.projection,
        final_cea_export_count=1 if row['projection']=='cea' else 0)
    return image

def original(row,prompt_path):
    import numpy as np
    from studies.original_spherediff.common import (import_official,official_hashes,seed_global,fresh_generator,
        specs,static_solver,scheduler_record,ForwardLog,defaults,CACHE,COMMIT)
    from studies.all_prompts.common import read,ORIGINAL
    n=row['backend'];classes,paths=import_official();plan=read(ORIGINAL/'manifest.json')
    assert paths==plan['import_paths'] and official_hashes()==plan['official_source_hashes']
    spec=specs()[n];assert spec['call']['num_inference_steps']==row['steps']
    emit('ORIGINAL_IMPORT_BEFORE_WEIGHTS',paths=paths,commit=COMMIT,source_hashes=official_hashes(),spec=spec)
    seed_global(torch,np);load=time.perf_counter()
    pipe=classes[n].from_pretrained(spec['model_source'],revision=spec['revision'],variant=spec['variant'],
        torch_dtype=torch.bfloat16,local_files_only=True,cache_dir=str(CACHE))
    pipe.to(torch.device('cuda'),dtype=torch.bfloat16);order=static_solver(pipe.scheduler)
    assert not getattr(pipe.vae,'use_tiling',False)
    generator=fresh_generator(torch);forward=pipe.transformer.forward;counter=ForwardLog(forward);pipe.transformer.forward=counter
    def progress(pipe,step,timestep,state):emit('PROGRESS',row=row,step=step+1,total=row['steps']);return state
    call=dict(spec['call'],prompt_txt_path=str(prompt_path),generator=generator,callback_on_step_end=progress)
    emit('ORIGINAL_ARGUMENTS',row=row,explicit=spec['call'],official_defaults=defaults(classes[n]),
        prompt_txt_path=str(prompt_path),generator_device=str(generator.device),generator_seed=0,
        model_loading_seconds=time.perf_counter()-load,
        global_seeding_order='random,numpy,torch,cuda_all before loading; fresh CUDA generator after loading')
    setup_peak=torch.cuda.max_memory_allocated();setup_reserved=torch.cuda.max_memory_reserved()
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    try:
        with torch.no_grad():result=pipe(**call)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
    finally:pipe.transformer.forward=forward
    schedule=scheduler_record(pipe.scheduler,order)
    expected=next(r['expected_scheduler'] for r in plan['rows'] if r['backend']==n)
    for k in ('class_name','config','timesteps','sigmas','effective_numerical_order'):assert schedule[k]==expected[k],k
    assert counter.calls==89*row['steps'] and sum(counter.shapes.values())==counter.calls
    image=result.images[0];assert image.size==(4096,2048)
    assert np.isfinite(np.asarray(image)).all()
    emit('GENERATION_VALIDATED',row=row,scheduler=schedule,transformer_calls=counter.calls,
        model_input_shapes=dict(counter.shapes),generation_seconds=seconds,
        peak_allocated_gib=max(setup_peak,torch.cuda.max_memory_allocated())/2**30,
        peak_reserved_gib=max(setup_reserved,torch.cuda.max_memory_reserved())/2**30)
    return image
