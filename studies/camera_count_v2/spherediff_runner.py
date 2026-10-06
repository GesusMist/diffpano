"""Original spherical pipelines; only frozen ring cameras and FLUX20 override."""
import collections
import contextlib
import importlib
import resource
import time
import numpy as np
import torch
from .common import *
from .cameras import restore
from .spherediff_adapter import override,directions,tensor_hash
from studies.original_spherediff.common import (import_official,official_hashes,COMMIT,CACHE,specs,
    seed_global,fresh_generator,static_solver,scheduler_record)

class StopPreflight(Exception):pass

def synchronize():torch.cuda.synchronize()

@contextlib.contextmanager
def instrumentation(pipe,module,sf,cameras,row,preflight):
    """Read-only observers always call the original operations unchanged."""
    n=len(cameras);points=2600 if row['backend']=='sana' else 26500
    center_first=row['backend']=='sana'
    saved=[];counter=dict(denoiser=0,encode=0,decode=0,initialize=0)
    shapes=collections.Counter();timing=collections.Counter()
    selected={};initial={};routing={};sample_count=0;paste_count=0
    point_weights=torch.zeros(points,device='cuda',dtype=torch.bfloat16)
    diagnostic_weights=torch.zeros(points,device='cuda')
    decoded_union=torch.zeros(points,device='cuda',dtype=torch.bool)
    def wrap(obj,name,fn):
        original=obj.__dict__.get(name) if isinstance(obj,type) else getattr(obj,name)
        saved.append((obj,name,original))
        setattr(obj,name,staticmethod(fn) if isinstance(obj,type) else fn)
    randn_original=module.randn_tensor
    def randn(*a,**kw):
        synchronize();start=time.perf_counter();x=randn_original(*a,**kw);synchronize()
        initial.update(initial_state_sha256=tensor_hash(x),shape=list(x.shape),dtype=str(x.dtype))
        timing['spherical_noise_draw_seconds']+=time.perf_counter()-start
        assert x.shape[-1]==points
        return x
    wrap(module,'randn_tensor',randn)
    route_original=sf.get_prompt_indices
    def route(view_dir,prompt_dir,prompt_fovs):
        assert len(view_dir)==n
        expected=directions(cameras).to(view_dir)
        assert torch.equal(expected,view_dir)
        out=route_original(view_dir,prompt_dir,prompt_fovs)
        routing.update(indices=out[0].tolist(),sha256=digest(out[0].tolist()),
            semantic_band_counts={str(k):sum(i//4==k for i in out[0].tolist()) for k in range(5)})
        assert all(tuple(f)==(80,80) for f in out[1])
        return out
    wrap(sf,'get_prompt_indices',route)
    sample_original=sf.dynamic_laetent_sampling
    def sample(spherical_points,cur_view_dir,num_points_on_sphere,_fov,temperature,center_first=True):
        nonlocal sample_count
        assert num_points_on_sphere==points
        assert center_first==(row['backend']=='sana') and tuple(_fov)==(80,80) and temperature==.1
        i=sample_count%n
        assert torch.equal(cur_view_dir[0],directions(cameras)[i].to(cur_view_dir))
        synchronize();start=time.perf_counter()
        ix,weight=sample_original(spherical_points,cur_view_dir,num_points_on_sphere,_fov,temperature,center_first=center_first)
        synchronize();timing['geometry_sampling_seconds']+=time.perf_counter()-start
        assert int(ix.min())>=0 and int(ix.max())<points and bool(torch.isfinite(weight).all())
        signature=dict(index_sha256=tensor_hash(ix),weight_sha256=tensor_hash(weight),selected_count=ix.numel())
        if sample_count<n:
            selected[i]=signature
            point_weights[ix]+=weight.flatten().to(point_weights)
            diagnostic_weights[ix]+=weight.flatten().float()
        else:assert signature==selected[i],('Changed sampling across steps/terminal',i)
        if sample_count>=n*row['steps']:
            assert bool((point_weights[ix]>0).all()), 'Final decode reads latent indices never updated'
            decoded_union[ix]=True
        sample_count+=1
        return ix,weight
    wrap(sf,'dynamic_laetent_sampling',sample)
    paste_original=sf.paste_perspective_to_erp_rectangle
    def paste(*a,**kw):
        nonlocal paste_count
        i=paste_count
        actual=a[2];expected=directions(cameras)[i].to(dtype=torch.bfloat16,device=actual.device).to(actual)
        assert torch.equal(actual[0],expected)
        synchronize();start=time.perf_counter();out=paste_original(*a,**kw);synchronize()
        timing['terminal_erp_assembly_seconds']+=time.perf_counter()-start;paste_count+=1
        if paste_count==n:
            # Observe the genuine denominator BEFORE the official zero handling.
            assert bool((out[1]>0).all()) and bool(torch.isfinite(out[1]).all())
            initial['final_rgb_minimum_denominator']=float(out[1].min())
        return out
    wrap(sf,'paste_perspective_to_erp_rectangle',paste)
    def counted(obj,name,key):
        fn=getattr(obj,name)
        def invoke(*a,**kw):
            x=kw.get('hidden_states',a[0] if a else None)
            if isinstance(x,torch.Tensor):assert bool(torch.isfinite(x).all()), key+' input nonfinite'
            if key=='denoiser' and preflight and counter[key]==n:raise StopPreflight()
            if key=='denoiser':
                shapes[str(list(x.shape))]+=1
            synchronize();start=time.perf_counter();out=fn(*a,**kw);synchronize()
            timing[key+'_seconds']+=time.perf_counter()-start;counter[key]+=1
            value=out[0] if isinstance(out,tuple) else out
            if isinstance(value,torch.Tensor):assert bool(torch.isfinite(value).all()), key+' output nonfinite'
            if key=='denoiser' and counter[key]%n==0:
                emit('SPHEREDIFF_FORWARDS',row=row,count=counter[key],total=n*row['steps'])
            return out
        wrap(obj,name,invoke)
    counted(pipe.transformer,'forward','denoiser')
    counted(pipe.vae,'encode','encode');counted(pipe.vae,'decode','decode')
    result=dict(counts=counter,shapes=shapes,timing=timing,initialization=initial,routing=routing)
    try:yield result
    finally:
        for obj,name,fn in reversed(saved):setattr(obj,name,fn)
        selected_union=point_weights>0
        result['latent_utilization']=dict(
            spherical_points=points,selected_index_union_count=int(selected_union.sum()),
            unused_spherical_points=int((~selected_union).sum()),selected_index_union_sha256=tensor_hash(selected_union),
            point_weight_sum_sha256=tensor_hash(point_weights),
            point_weight_sum_dtype='BF16, original per-view accumulation order',
            point_weight_min=float(point_weights.min()),point_weight_max=float(point_weights.max()),
            positive_weight_min=float(point_weights[selected_union].min()) if bool(selected_union.any()) else None,
            diagnostic_fp32_weight_min=float(diagnostic_weights.min()),diagnostic_fp32_weight_max=float(diagnostic_weights.max()),
            decoded_union_count=int(decoded_union.sum()),all_decode_indices_updated=bool((~decoded_union|selected_union).all()),
            camera_selection_hashes=[selected[k] for k in sorted(selected)],sampler_calls=sample_count,paste_calls=paste_count,
            center_first=center_first)

def baseline_initialization(pipe,module,spec,prompt_path):
    fn=module.randn_tensor;captured={}
    def capture(*a,**kw):
        x=fn(*a,**kw)
        captured.update(sha256=tensor_hash(x),shape=list(x.shape))
        raise StopPreflight()
    module.randn_tensor=capture
    try:
        try:pipe(**dict(spec['call'],prompt_txt_path=str(prompt_path),generator=fresh_generator(torch)))
        except StopPreflight:pass
        else:raise AssertionError('Initialization observer did not stop')
    finally:module.randn_tensor=fn
    assert captured
    return captured

@torch.no_grad()
def generate(row,prompt_path,preflight=False):
    assert row['family']=='spherediff_camera_override' and row['strategy']=='old'
    layout=load_layout(row);cameras=restore(layout);n=len(cameras)
    classes,paths=import_official();audit=read(ROOT/'audit.json');assert paths==audit['official_import_paths']
    assert official_hashes()==audit['protected_sources']['official']
    spec=audit['spherediff'][row['backend']]
    assert spec['call']['num_inference_steps']==20==row['steps']
    seed_global(torch,np);start=time.perf_counter()
    pipe=classes[row['backend']].from_pretrained(spec['model_source'],revision=spec['revision'],variant=spec['variant'],
        torch_dtype=torch.bfloat16,local_files_only=True,cache_dir=str(CACHE))
    pipe.to('cuda',dtype=torch.bfloat16);order=static_solver(pipe.scheduler)
    pipe.set_progress_bar_config(disable=True)
    assert not getattr(pipe.vae,'use_tiling',False)
    module=importlib.import_module(classes[row['backend']].__module__);sf=module.SphericalFunctions
    synchronize();load_seconds=time.perf_counter()-start
    baseline=baseline_initialization(pipe,module,spec,prompt_path) if preflight else None
    synchronize();began=time.perf_counter()
    with override(cameras,sf=sf) as camera_audit,instrumentation(pipe,module,sf,cameras,row,preflight) as observed:
        try:
            output=pipe(**dict(spec['call'],prompt_txt_path=str(prompt_path),generator=fresh_generator(torch)))
        except StopPreflight:
            assert preflight;output=None
        assert camera_audit['factory_calls']==1
    synchronize();elapsed=time.perf_counter()-began
    assert observed['counts']==expected_counts(row,steps=1 if preflight else None,terminal=not preflight)
    initial=observed['initialization']['initial_state_sha256']
    if preflight:
        assert initial==baseline['sha256']
        immutable(ROOT/'preflight'/('original-initialization-'+row['backend']+'.json'),
            dict(backend=row['backend'],steps=20,shape=baseline['shape'],sha256=initial))
    else:assert initial==preflight_gate(row)['record']['initialization']['initial_state_sha256']
    schedule=scheduler_record(pipe.scheduler,order)
    if row['backend']=='flux':
        reference=read(REPO/'outputs/flux-step-controls-seed0/spherediff/flux/steps20/ruins/config.json')
        expected=reference['schedule']
        assert initial==reference['initialization']['initial_state_sha256']
    else:
        plan=read(REPO/'outputs/original-spherediff/ruins-underwater-seed0-v1/manifest.json')
        expected=next(r['expected_scheduler'] for r in plan['rows'] if r['backend']=='sana')
    for key in ('class_name','config','timesteps','sigmas','effective_numerical_order'):
        if key in expected:assert schedule[key]==expected[key],key
    assert len(schedule['timesteps'])==20
    if preflight:
        # Same original discrete sampler for the would-be terminal views;
        # no model, VAE, or scientific image is generated here.
        points=sf.fibonacci_sphere(N=spec['call']['n_spherical_points']).to('cuda',dtype=torch.bfloat16)[None,None]
        dirs=directions(cameras).to('cuda',dtype=torch.bfloat16)
        for i,d in enumerate(dirs):
            ix,weight=sf.dynamic_laetent_sampling(points,d[None],points.shape[-2],(80,80),.1,
                                                 center_first=row['backend']=='sana')
            expected_selection=observed['latent_utilization']['camera_selection_hashes'][i]
            assert tensor_hash(ix)==expected_selection['index_sha256']
            assert tensor_hash(weight)==expected_selection['weight_sha256']
        observed['latent_utilization']['terminal_geometry_verified_without_decoding']=True
    else:
        assert observed['latent_utilization']['sampler_calls']==n*(row['steps']+1)
        assert observed['latent_utilization']['paste_calls']==n
        assert output.images[0].size==(4096,2048)
    geom_route=layout['coverage']['spherediff']['adapter']
    assert observed['routing']['indices']==geom_route['spherediff_prompt_slots']
    observed['routing']['semantic_band_disagreements_with_diffpano']=geom_route['semantic_band_disagreements']
    observed['routing']['tie_policy']=geom_route['tie_policy']
    record=dict(label='SphereDiff — reduced ring cameras',configuration=spec,
        backend=dict(name=row['backend'],precision='bfloat16',vae_tiling=False,cpu_offload=False,
                     spherical_points=spec['call']['n_spherical_points'],model_input_shapes=dict(observed['shapes']),
                     persistent_native_shape=observed['initialization']['shape'],
                     local_rgb=layout['coverage']['spherediff']['backends'][row['backend']]['local_rgb_size'],
                     local_native_sample_side_counts=sorted(set(round(v['selected_count']**.5) for v in observed['latent_utilization']['camera_selection_hashes']))),
        official_source_commit=COMMIT,official_import_paths=paths,official_source_hashes=official_hashes(),
        camera_override=camera_audit,adapter_sha256=sha(STUDY/'spherediff_adapter.py'),
        generation=dict(seed=0,steps=20,batch_size=1,guidance=spec['call']['guidance_scale'],
            true_cfg_scale=spec['call'].get('true_cfg_scale'),negative_prompt='',
            step_override='User explicitly requested SphereDiff FLUX 20; SANA keeps 20'),
        schedule=schedule,schedule_sha256=digest(schedule),routing=observed['routing'],
        initialization=dict(method='original spherical Gaussian',initial_state_sha256=initial,
            shape=observed['initialization']['shape'],seed=0,spherical_points=spec['call']['n_spherical_points'],
            original89_initialization_match=True,preliminary_flux_prepare_latents_preserved=True),
        aggregation=dict(persistent_representation='spherical native latents',method='original weighted latent aggregation',
            temperature=.1,center_first=row['backend']=='sana',final_export='original perspective RGB to ERP'),
        bridge='not applicable',transition='original scheduler and latent aggregation',
        latent_utilization=observed['latent_utilization'],
        trajectory=dict(time_travel=False,replay_count=0,backward_count=0,original_noise_reinjection_count=0),
        execution=dict(counts=observed['counts'],guided_predictions=observed['counts']['denoiser'],
            physical_model_calls=observed['counts']['denoiser'],model_input_shapes=dict(observed['shapes']),
            model_loading_seconds=load_seconds,pipeline_seconds=elapsed,stage_seconds=dict(observed['timing']),
            total_seconds=time.perf_counter()-start,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))
    return (None if preflight else output.images[0]),record
