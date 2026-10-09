"""Backend-independent final native denoising, with no consensus work in the tail.

Callers prepare local noisy states exactly once using their existing representation,
then call NativeRefinement.step at the ORIGINAL scheduler indices. Terminal assembly
belongs to the caller. This module has no dependency on a study or fusion method.
"""
from dataclasses import dataclass
import hashlib
import math
import time
import torch


@dataclass
class RefinementConfig:
    last_fraction: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.last_fraction) or not 0 <= self.last_fraction <= 1:
            raise ValueError('global_pipeline.refinement.last_fraction must be finite in [0,1]')


def independent_count(total, fraction):
    if not isinstance(total, int) or isinstance(total, bool) or total < 1:
        raise ValueError('A positive number of original intervals is required')
    RefinementConfig(fraction)
    return 0 if fraction == 0 else min(total, max(1, math.ceil(fraction*total)))


def fraction_of(config):
    return 0.0 if config is None else config.last_fraction


def cutoff(total, config=None):
    return total - independent_count(total, fraction_of(config))


def resolved_refinement(timesteps, config=None):
    total=len(timesteps);n=independent_count(total,fraction_of(config));coupled=total-n
    return dict(last_fraction=fraction_of(config),total_intervals=total,coupled_intervals=coupled,
                independent_intervals=n,transition_step=coupled+1 if n else None,
                transition_timestep=float(timesteps[coupled]) if n else None)


def state_hash(states):
    digest=hashlib.sha256()
    for value in states:digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class NativeRefinement:
    """Freeze geometry/identity/conditioning and advance each native state exactly once."""
    def __init__(self, backend, geometry, conditions, start_index, *, identities=None, order=None):
        from diffpano.pipelines.base import ensure_first_order_scheduler
        from diffpano.trajectory import conditioning_digest
        self.backend=backend;self.geometry=tuple(geometry)
        self.geometry_identity=repr(self.geometry)
        self.identities=tuple(range(len(geometry))) if identities is None else tuple(identities)
        if len(self.identities)!=len(geometry) or len(set(self.identities))!=len(geometry):raise ValueError('Unique stable local identities required')
        self.order=tuple(range(len(geometry))) if order is None else tuple(order)
        if sorted(self.order)!=list(range(len(geometry))):raise ValueError('Invalid local processing order')
        if len(conditions)!=len(geometry):raise ValueError('One conditioning per local state required')
        self.conditions=tuple(conditions);self.condition_hashes=tuple(conditioning_digest(c) for c in conditions)
        self.schedule=backend.timesteps.clone();self.next_index=start_index
        if not 0 <= start_index < len(self.schedule):raise ValueError('Invalid original transition index')
        scheduler=getattr(getattr(backend,'pipeline',None),'scheduler',None)
        if scheduler is not None:ensure_first_order_scheduler(scheduler)
        if not callable(getattr(backend,'denoise_native_step',None)):raise TypeError('Native sampler required for refinement')
        self.transition=dict(original_index=start_index,step=start_index+1,timestep=float(self.schedule[start_index]),
                             identities=list(self.identities),geometry_sha256=hashlib.sha256(self.geometry_identity.encode()).hexdigest(),
                             conditioning_hashes=self.condition_hashes)
        self.records=[]

    @torch.no_grad()
    def step(self, states, index, *, geometry=None, conditions=None):
        from diffpano.pipelines.base import reset_scheduler_step_state
        from diffpano.trajectory import conditioning_digest
        b=self.backend
        if index!=self.next_index:raise ValueError('Independent intervals must follow the original schedule in order')
        if not torch.equal(self.schedule,b.timesteps):raise AssertionError('Prepared schedule mutated')
        if geometry is not None and repr(tuple(geometry))!=self.geometry_identity:raise AssertionError('Frozen geometry changed')
        conditions=self.conditions if conditions is None else tuple(conditions)
        if tuple(conditioning_digest(c) for c in conditions)!=self.condition_hashes:raise AssertionError('Frozen conditioning changed')
        if len(states)!=len(self.geometry):raise ValueError('Local state count changed')
        if not self.records:self.transition['native_state_sha256']=state_hash(states)
        if b.device.type=='cuda':torch.cuda.synchronize(b.device)
        started=time.perf_counter();out=[None]*len(states)
        before=getattr(b,'guided_prediction_count',None)
        scheduler=getattr(getattr(b,'pipeline',None),'scheduler',None)
        for i in self.order:
            if scheduler is not None:reset_scheduler_step_state(scheduler)
            current=states[i].to(b.device).float()
            value=b.denoise_native_step(current.clone(),b.timesteps[index],conditions[i])
            if value.shape!=current.shape or not bool(torch.isfinite(value).all()):raise ValueError('Invalid independent native update')
            out[i]=value.detach().float().to(states[i].device)
            for name in ('last_model_prediction','last_clean_prediction'):
                if hasattr(b,name):setattr(b,name,None)
        if before is not None and b.guided_prediction_count-before!=len(states):raise AssertionError('Unexpected native sampler call count')
        if b.device.type=='cuda':torch.cuda.synchronize(b.device)
        record=dict(k=index,pass_kind='initial',phase='independent_refinement',fusion=None,
                    model_timestep=float(self.schedule[index]),guided_predictions=len(states),
                    stage_seconds={'independent_native':time.perf_counter()-started},
                    native_min=min(float(x.min()) for x in out),native_max=max(float(x.max()) for x in out))
        self.records.append(record);self.next_index+=1
        return out,record


def continue_native(backend,states,geometry,conditions,start_index,*,identities=None,order=None):
    tail=NativeRefinement(backend,geometry,conditions,start_index,identities=identities,order=order)
    for index in range(start_index,len(backend.timesteps)):
        states,_=tail.step(states,index)
    return states,tail


def encode_rgb_state(backend,rgb,pixel_native=False):
    return rgb.float() if pixel_native else backend.encode_clean(rgb).float()


def decode_terminal(backend,state,pixel_native=False):
    return state.to(backend.device).float() if pixel_native else backend.decode_native_canvas(state.to(backend.device)).float()


def tail_step_records(tail, *, planar=False):
    # No coverage/fusion measurement is fabricated for independent intervals.
    from diffpano.erp_pipeline import StepDiagnostics
    from diffpano.planar_pipeline import PlanarStepDiagnostics
    cls=PlanarStepDiagnostics if planar else StepDiagnostics
    return [cls(r['k'],r['model_timestep'],len(tail.geometry),float('nan'),float('nan'),
                float('nan'),float('nan'),float('nan'),r['stage_seconds'],
                {'phase':'independent_refinement','fusion':None,'native_min':r['native_min'],'native_max':r['native_max']})
            for r in tail.records]


def is_pixel_native(backend):
    # Adapters expose a protocol, but legacy mocks need no model dependencies.
    from diffpano.pipelines.pixeldit import PixelDiTViewDenoiser
    return isinstance(backend,PixelDiTViewDenoiser) or bool(getattr(backend,'pixel_native',False))


def erp_canvas_tail(pipeline,source,bank,cameras,index,*,noise_bank=None,batch_size=None,size=None):
    """Prepare the ordinary next native inputs once; RGB and x0 share the tail."""
    from diffpano.fusion import RGBFusionAccumulator
    b=getattr(pipeline,'backend',getattr(pipeline,'view_denoiser',None));pixel=is_pixel_native(b)
    if source is None:
        source=torch.zeros(batch_size,3,*size,device=b.device,dtype=torch.float32);bootstrap=True
    else:batch_size=source.shape[0];bootstrap=False
    states=[];conditions=[]
    for i,camera in enumerate(cameras):
        if bootstrap:state=b.make_initial_noisy_state(noise_bank.get([i],device=b.device),b.timesteps[index])
        else:
            rgb=pipeline.warp_operator.erp_to_perspective(source,camera)
            state=encode_rgb_state(b,rgb,pixel)
            if noise_bank is not None:state=b.add_fixed_noise(state,noise_bank.get([i],device=b.device),b.timesteps[index])
        states.append(state.detach().cpu().float())
        conditions.append(b.conditioning_for_cameras(bank,[camera],batch_size=batch_size))
    states,tail=continue_native(b,states,cameras,conditions,index)
    factory=getattr(pipeline.warp_operator,'create_fusion_accumulator',None)
    acc=factory(source) if factory else None;joint=acc is not None
    if acc is None:acc=RGBFusionAccumulator(source,pipeline.fusion_config)
    for state,camera in zip(states,cameras):
        rgb=decode_terminal(b,state,pixel)
        if joint:acc.accumulate(rgb,camera)
        else:acc.accumulate(pipeline.warp_operator.perspective_to_erp(rgb,camera,source.shape[-2:]))
    final=acc.finalize();pipeline.refinement_tail=tail
    return final.erp_rgb,tail_step_records(tail)


def planar_canvas_tail(pipeline,source,bank,layout,index,*,noise_bank=None,batch_size=None):
    from diffpano.planar import PlanarFusionAccumulator,extract_planar_patch,planar_prompt_indices
    b=pipeline.backend;pixel=is_pixel_native(b);bootstrap=source is None
    if bootstrap:source=torch.zeros(batch_size,3,layout.canvas_height,layout.canvas_width,device=b.device,dtype=torch.float32)
    else:batch_size=source.shape[0]
    patches=pipeline._ordered_patches(layout,pipeline.patch_order);states=[];conditions=[]
    for patch in patches:
        if bootstrap:state=b.make_initial_noisy_state(noise_bank.get([patch.index],device=b.device),b.timesteps[index])
        else:
            state=encode_rgb_state(b,extract_planar_patch(source,patch),pixel)
            if noise_bank is not None:state=b.add_fixed_noise(state,noise_bank.get([patch.index],device=b.device),b.timesteps[index])
        states.append(state.detach().cpu().float())
        slots=planar_prompt_indices(layout,[patch],pipeline.planar_config.prompt_assignment)
        conditions.append(b.conditioning_for_prompt_indices(bank,slots,batch_size=batch_size))
    states,tail=continue_native(b,states,patches,conditions,index,identities=[p.index for p in patches])
    acc=PlanarFusionAccumulator(source,pipeline.fusion_config)
    for state,patch in zip(states,patches):acc.accumulate(decode_terminal(b,state,pixel),patch)
    final=acc.finalize();pipeline.refinement_tail=tail
    return final.canvas_rgb,tail_step_records(tail,planar=True)
