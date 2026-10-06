"""N-view execution through unchanged GWTFlow, bridge, and ordinary intervals."""
import collections
import contextlib
import resource
import time
import torch
from .common import *
from .cameras import restore
from studies.camera_patching.runtime import setup,initialize
from studies.camera_patching.cameras import routing,angular_hash
from studies.all_prompts.runtime import ordinary_run
from studies.tt_cea.pipeline import ExperimentalPipeline
from studies.gwtf_noise.run import runtime_audit,count_calls

@contextlib.contextmanager
def shape_counter(module):
    fn=module.forward;hist=collections.Counter()
    def forward(*a,**kw):
        x=kw.get('hidden_states',a[0] if a else None)
        if x is not None:hist[str(list(x.shape))]+=1
        return fn(*a,**kw)
    module.forward=forward
    try:yield hist
    finally:module.forward=fn

def sync():torch.cuda.synchronize()
@torch.no_grad()
def generate(row,prompt_path,preflight=False):
    from diffpano.dense_consensus import prepare_camera_conditioning
    from diffpano.erp_noise_initialization import native_cameras,primary_noise_size
    from diffpano.diagnostics import tensor_to_pil
    from diffpano.planar_pipeline import _negative_prompt
    layout=load_layout(row);start=time.perf_counter()
    c,b,old,table,bank=setup(row['backend'],prompt_path);sync()
    load_seconds=time.perf_counter()-start
    cams=restore(layout,c.view);ids=layout['camera_ids'];n=len(cams)
    assert n==row['num_cameras']
    conditions,slots=prepare_camera_conditioning(b,bank,cams,c.dense_consensus.prompt_assignment,1);del bank
    route=routing(cams,prompt_record(prompt_path)[0]['effective_lines'])
    assert slots==route['indices']
    p=ExperimentalPipeline(b,cams,c,projection='erp')
    provenance,audit=runtime_audit(c,b,p,conditions,slots,old)
    allowed={'camera_sha256','camera_geometry_sha256','conditioning_sha256','prompt_indices'}
    assert set(audit['differences'])<=allowed,audit['differences']
    assert primary_noise_size(native_cameras(b,cams))==tuple(old['initialization']['source_shape'][-2:])
    assert safe(provenance['prepared_schedule'])==safe(old['prepared_schedule'])
    assert angular_hash(cams)==layout['angular_geometry_sha256']
    initstart=time.perf_counter();states,init=initialize(b,row['backend'],cams,ids);sync()
    init_seconds=time.perf_counter()-initstart
    assert init['source_shape']==old['initialization']['source_shape']
    assert init['source_sha256']==old['initialization']['source_sha256']
    if not preflight:
        assert init['initial_local_sha256']==preflight_gate(row)['record']['initialization']['initial_state_sha256']
    if preflight and row['strategy']=='old' and n==70:
        replay,replayed=initialize(b,row['backend'],cams,ids,reverse=True)
        assert replayed['initial_local_sha256']==init['initial_local_sha256'];del replay
    stage=collections.Counter();intervals=[];module=b.model if row['backend']=='pixeldit' else b.pipeline.transformer
    def progress(step,total,r):
        assert r['min_contributors']>0 and r['minimum_weight_sum']>0
        assert r['current_state_error_max']<1e-4
        intervals.append(r)
        stage.update(r['stage_seconds'])
        emit('PROGRESS',index=row['index'],step=step,total=total,
             minimum_weight_sum=r['minimum_weight_sum'],min_contributors=r['min_contributors'])
    sync();runstart=time.perf_counter()
    with shape_counter(module) as shapes,count_calls(b,row['backend']) as counts:
        if preflight:
            states,r=p.advance_interval(states,conditions,table[0],pass_kind='initial')
            progress(1,len(table),r)
            assert all(torch.isfinite(s).all() for s in states)
            details={};image=None
        else:
            holder=[states];del states
            final,erp,details=ordinary_run(p,holder.pop(),conditions,table,progress)
            assert erp.shape==(1,3,2048,4096) and torch.isfinite(erp).all()
            image=tensor_to_pil(erp[0].cpu())
    sync();runtime=time.perf_counter()-runstart
    assert counts==expected_counts(row,steps=1 if preflight else None,terminal=not preflight)
    assert sum(shapes.values())==counts['denoiser']
    actual_guidance=float(b.cfg_scale if row['backend']=='pixeldit' else b.guidance_scale)
    expected_guidance=c.pixeldit.cfg_scale if row['backend']=='pixeldit' else c.generation.guidance_scale
    assert actual_guidance==expected_guidance
    record=dict(configuration=c.to_dict(),configuration_scope='historical backend settings; actual frozen camera tuple injected explicitly',
        runtime_audit=provenance,checkpoint=audit['checkpoint_provenance'],
        backend=dict(name=row['backend'],precision=str(b.dtype),native_arithmetic='float32',
            local_rgb=[c.view.height,c.view.width],local_native=provenance['local_native_resolution'],
            native_channels=b.native_channels,vae_tiling=bool(getattr(getattr(getattr(b,'pipeline',None),'vae',None),'use_tiling',False)),
            cpu_offload=c.model.cpu_offload,model_input_shapes=dict(shapes)),
        generation=dict(seed=0,steps=len(table),guidance=actual_guidance,true_cfg_scale=c.generation.true_cfg_scale,
            batch_size=1,negative_prompt=c.pixeldit.negative_prompt if row['backend']=='pixeldit' else _negative_prompt(c),
            pixeldit_interval_guidance=list(b.interval_guidance) if row['backend']=='pixeldit' else None),
        schedule=provenance['prepared_schedule'],schedule_sha256=digest(safe(provenance['prepared_schedule'])),
        routing=dict(indices=slots,sha256=digest(slots),semantic_band_counts=route['semantic_band_counts']),
        initialization=dict(method='gwtflow',initial_state_sha256=init['initial_local_sha256'],source_shape=init['source_shape'],
            source_sha256=init['source_sha256'],source_seed=0,source_density=init['source_density'],rng=init['rng'],
            config=init['config'],scaling=init['scaling'],scaling_applications=1,
            native_initialization_calls=n,denoiser_calls=0,vae_calls=0,source_released=True,transport_buffers_released=True,
            geometry_seconds=init['geometry_seconds'],sampling_and_scaling_seconds=init['sampling_and_scaling_seconds']),
        aggregation=dict(mode='weighted_average',weight_mode='spherediff_center',temperature=.1,
            persistent_representation='local native states',consensus_projection='erp',warp='standard',
            perspective_to_erp='bilinear',erp_to_perspective='nearest'),
        bridge=p.bridge_mode,transition='preserve_current_state',
        trajectory=dict(time_travel=False,replay_count=0,backward_count=0,original_noise_reinjection_count=0),
        execution=dict(counts=counts,guided_predictions=counts['denoiser'],physical_model_calls=counts['denoiser'],
            model_input_shapes=dict(shapes),model_loading_and_conditioning_seconds=load_seconds,
            initialization_seconds=init_seconds,denoising_and_fusion_seconds=runtime,stage_seconds=dict(stage),
            terminal_seconds=details.get('terminal_seconds',{}),total_seconds=time.perf_counter()-start,
            peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20),
        interval_checks=intervals)
    return image,record
