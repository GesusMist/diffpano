"""Audited official loader adapter; official pipeline and numerical code are untouched.

Copied only the existing loader/instrumentation from flux_step_controls.runtime.original.
The sole adapter is the new suite's fresh preflight gate; the old gate is not modified.
"""
import collections,importlib,time
from studies.ournew_comparison.common import read,digest,gate
from studies.all_prompts.common import emit,safe
from studies.flux_step_controls.runtime import original_spec,tensor_hash,PreflightStop
from studies.flux_step_controls.common import schedule_valid

def preflight_gate():
    return gate('official.json')

def original(row,prompt_path,preflight=False):
    import torch
    import numpy as np
    from studies.original_spherediff.common import (import_official,official_hashes,seed_global,fresh_generator,
        static_solver,scheduler_record,defaults,CACHE,COMMIT,ROOT as HIST_ROOT)
    classes,paths=import_official();plan=read(HIST_ROOT/'manifest.json')
    assert paths==plan['import_paths'] and official_hashes()==plan['official_source_hashes']
    spec,delta=original_spec();assert row['steps']==20
    seed_global(torch,np)
    pipe=classes['flux'].from_pretrained(spec['model_source'],revision=spec['revision'],variant=spec['variant'],
        torch_dtype=torch.bfloat16,local_files_only=True,cache_dir=str(CACHE))
    pipe.to(torch.device('cuda'),dtype=torch.bfloat16);order=static_solver(pipe.scheduler)
    assert not getattr(pipe.vae,'use_tiling',False)
    module=importlib.import_module(classes['flux'].__module__)
    reference=next(r['expected_scheduler'] for r in plan['rows'] if r['backend']=='flux')
    noise_records=[];original_randn=module.randn_tensor
    def randn(*a,**kw):
        x=original_randn(*a,**kw)
        noise_records.append(dict(shape=list(x.shape),dtype=str(x.dtype),sha256=tensor_hash(x)))
        return x
    # Wrapper observes the actual official spherical draw; it adds no random draws.
    module.randn_tensor=randn
    count=dict(denoiser=0,encode=0,decode=0);shapes=collections.Counter();saved=[]
    prior_init=None
    def capture(obj,method,key,stop_before=False):
        fn=getattr(obj,method);saved.append((obj,method,fn))
        def invoke(*a,**kw):
            if stop_before:raise PreflightStop()
            count[key]+=1
            if key=='denoiser':shapes[str(list(kw['hidden_states'].shape))]+=1
            return fn(*a,**kw)
        setattr(obj,method,invoke)
    try:
        if preflight:
            # Official 28-step call constructs its full scheduler and initial state, then stops BEFORE inference.
            capture(pipe.transformer,'forward','denoiser',stop_before=True)
            try:
                with torch.no_grad():pipe(**dict(spec['call'],num_inference_steps=28,prompt_txt_path=str(prompt_path),generator=fresh_generator(torch)))
            except PreflightStop:pass
            else:raise AssertionError('Preflight failed to stop before inference')
            prior_init=noise_records[-1]['sha256']
            prior=scheduler_record(pipe.scheduler,order)
            for k in ('class_name','config','timesteps','sigmas','effective_numerical_order'):assert prior[k]==reference[k],k
            for obj,method,fn in reversed(saved):setattr(obj,method,fn)
            saved.clear();noise_records.clear()
        capture(pipe.transformer,'forward','denoiser')
        capture(pipe.vae,'encode','encode');capture(pipe.vae,'decode','decode')
        def progress(pipe,step,timestep,state):
            emit('PROGRESS',row=row,step=step+1,total=20)
            if preflight and step==0:
                assert bool(torch.isfinite(state['latents']).all());raise PreflightStop()
            return state
        call=dict(spec['call'],prompt_txt_path=str(prompt_path),generator=fresh_generator(torch),callback_on_step_end=progress)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
        try:
            with torch.no_grad():result=pipe(**call)
        except PreflightStop:
            assert preflight;result=None
        torch.cuda.synchronize();seconds=time.perf_counter()-started
    finally:
        module.randn_tensor=original_randn
        for obj,method,fn in reversed(saved):setattr(obj,method,fn)
    schedule=scheduler_record(pipe.scheduler,order);schedule_valid(schedule,20)
    for k in ('class_name','config','effective_numerical_order'):assert schedule[k]==reference[k],k
    assert schedule['timesteps']!=reference['timesteps']
    assert len(noise_records)==1 and noise_records[0]['shape'][-1]==26500
    if preflight:assert prior_init==noise_records[0]['sha256']
    else:
        gate=preflight_gate()
        assert noise_records[0]['sha256']==gate['record']['initialization']['initial_state_sha256']
        prior_init=gate['record']['historical_initialization_sha256']
    assert count['denoiser']==89*(1 if preflight else 20) and sum(shapes.values())==count['denoiser']
    image=None if preflight else result.images[0]
    if image is not None:assert image.size==(4096,2048) and np.isfinite(np.asarray(image)).all()
    config=dict(configuration=spec,configuration_differences=delta,official_defaults=defaults(classes['flux']),
        model=dict(id=spec['model_source'],revision=spec['revision']),
        official_source_commit=COMMIT,official_import_paths=paths,official_source_hashes=official_hashes(),
        schedule=schedule,schedule_sha256=digest(schedule),
        initialization=dict(initial_state_sha256=noise_records[0]['sha256'],draws=noise_records,
          convention='official prepare_latents CUDA draw, then spherical randn_tensor; fresh CUDA generator seed 0'),
        historical_initialization_sha256=prior_init,initialization_match=prior_init==noise_records[0]['sha256'],
        execution=dict(counts=count,model_input_shapes=dict(shapes),generation_seconds=seconds,
          peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/2**30),
        projection='spherical',vae_tiling=False,model_cpu_offload=False)
    return image,config
