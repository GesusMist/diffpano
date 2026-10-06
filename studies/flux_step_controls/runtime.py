"""Only step count and bookkeeping differ from the audited historical runtimes."""
import collections
import contextlib
import hashlib
import importlib
import time
from dataclasses import replace
from .common import *


def diff_configuration(prompt_path):
    from studies.all_prompts.audit import configuration
    from studies.gwtf_erp4k.common import differences
    old_c,_,cams,ids,old=configuration('flux',str(prompt_path))
    c=replace(old_c,generation=replace(old_c.generation,num_inference_steps=28))
    delta=differences(old_c.to_dict(),c.to_dict())
    assert {d['path'] for d in delta}=={'generation.num_inference_steps'}
    return c,cams,ids,old,delta


def original_spec():
    from studies.original_spherediff.common import specs
    from studies.gwtf_erp4k.common import differences
    old=specs()['flux'];new=dict(old,call=dict(old['call'],num_inference_steps=20))
    delta=differences(old,new)
    assert {d['path'] for d in delta}=={'call.num_inference_steps'}
    return new,delta


def tensor_hash(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


@contextlib.contextmanager
def shape_counter(module):
    original=module.forward;shapes=collections.Counter()
    def call(*a,**kw):
        x=kw.get('hidden_states',a[0] if a else None)
        if x is not None:shapes[str(list(x.shape))]+=1
        return original(*a,**kw)
    module.forward=call
    try:yield shapes
    finally:module.forward=original


def diffpano(row,prompt_path,preflight=False):
    import torch
    import numpy as np
    from studies.original_spherediff.common import seed_global, scheduler_record
    from studies.all_prompts.runtime import ordinary_run
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
    c,cams,ids,old,delta=diff_configuration(prompt_path)
    assert row['steps']==28 and row['projection']=='erp'
    seed_global(torch,np);b=build_view_denoiser(c);_configure_denoiser(c,b)
    table=tuple(replace(x,eligible=False) for x in prepare_schedule(b,c,28))
    schedule=safe(canonical_schedule(schedule_metadata(b)))
    original=safe(old['prepared_schedule'])
    for k in ('class_name','config'):
        if k in original:assert schedule[k]==original[k],k
    schedule_valid(schedule,28)
    assert schedule['timesteps']!=original['timesteps'] and len(original['timesteps'])==20
    bank=b.prepare_prompt_conditioning(load_directional_prompts(str(prompt_path)),_negative_prompt(c))
    conditions,slots=prepare_camera_conditioning(b,bank,cams,c.dense_consensus.prompt_assignment,1);del bank
    p=ExperimentalPipeline(b,cams,c,projection='erp')
    provenance,audit=runtime_audit(c,b,p,conditions,slots,old)
    assert set(audit['differences'])<= {'prepared_schedule','conditioning_sha256'},audit['differences']
    group=dict(cameras=cams,ids=ids,conditions=conditions,slots=slots)
    states,init,init_audit=initialize(b,'flux',group,old)
    assert init['initial_local_sha256']==old['initialization']['initial_local_sha256']
    reference_init=init['initial_local_sha256']
    if preflight:
        # Rebuild the prior schedule on the same backend and verify exact initialization.
        prepare_schedule(b,c,20)
        assert safe(canonical_schedule(schedule_metadata(b)))==original
        replay,control_init,_=initialize(b,'flux',group,old);del replay
        assert control_init['initial_local_sha256']==reference_init
        table=tuple(replace(x,eligible=False) for x in prepare_schedule(b,c,28))
        assert safe(canonical_schedule(schedule_metadata(b)))==schedule
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    with torch.no_grad(),count_calls(b,'flux') as counts,shape_counter(b.pipeline.transformer) as shapes:
        if preflight:
            next_states,details=p.advance_interval(states,conditions,table[0],pass_kind='initial')
            assert all(torch.isfinite(x).all() for x in next_states)
            assert details['min_contributors']>0 and details['current_state_error_max']<1e-4
            image=None
        else:
            holder=[states];del states
            native,erp,details=ordinary_run(p,holder.pop(),conditions,table,
                lambda k,n,s:emit('PROGRESS',row=row,step=k,total=n))
            assert erp.shape==(1,3,2048,4096) and bool(torch.isfinite(erp).all())
            image=tensor_to_pil(erp[0].cpu())
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    n=1 if preflight else 28
    assert counts==dict(denoiser=89*n,encode=2*89*n,decode=89*n+(0 if preflight else 89),initialize=0)
    config=dict(configuration=c.to_dict(),configuration_differences=delta,
        model=dict(id=c.model.id,revision=c.model.revision,checkpoint=audit['checkpoint_provenance']),
        schedule=schedule,schedule_sha256=digest(schedule),initialization=init,
        historical_initialization_sha256=reference_init,initialization_match=True,
        camera=dict(strategy='old89',num_cameras=89,fov_x=80.,fov_y=80.,angular_geometry_sha256=provenance['camera_geometry_sha256']),
        runtime_provenance=provenance,execution=dict(counts=counts,model_input_shapes=dict(shapes),
        generation_seconds=seconds,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/2**30),
        time_travel_enabled=False,bridge=p.bridge_mode,projection='erp')
    return image,config


class PreflightStop(Exception): pass


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
        gate=preflight_gate('spherediff')
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
