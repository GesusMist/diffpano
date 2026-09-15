"""K: persistent local native states, transient standard-warp ERP RGB consensus.

No ERP latent field, VAE residual, or fixed-noise reconstruction is used.
"""
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field

import torch

from diffpano.camera import build_camera_sampler
from diffpano.current_state_transition import interpolate_from_current_state, transition_diagnostics
from diffpano.erp_pipeline import StepDiagnostics
from diffpano.fusion import RGBFusionAccumulator
from diffpano.native_multidiffusion import prepare_native_backend
from diffpano.pipelines.base import reset_scheduler_step_state
from diffpano.trajectory import conditioning_digest
from diffpano.warp import StandardWarpOperator


def camera_digest(cameras):
    return hashlib.sha256(json.dumps([asdict(c) for c in cameras],sort_keys=True).encode()).hexdigest()


def native_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def coverage_statistics(counts):
    return dict(coverage_percent=float((counts>0).float().mean()*100),
        multi_contributor_percent=float((counts>1).float().mean()*100),
        minimum_contributors=float(counts.min()),maximum_contributors=float(counts.max()),
        mean_contributors=float(counts.mean()))


def projected_disagreement(contributions):
    records=[]
    for i,a in enumerate(contributions):
        for j in range(i+1,len(contributions)):
            b=contributions[j];mask=(a.valid_mask>0)&(b.valid_mask>0)
            count=int(mask.sum())
            if not count:continue
            difference=(a.rgb-b.rgb)*mask
            records.append(dict(view_i=i,view_j=j,overlap_pixels=count,
                mae=float(difference.abs().sum()/(3*count)),
                rmse=float((difference.square().sum()/(3*count)).sqrt())))
    return records


def geometry_preflight(cameras,erp_size,operator,device=torch.device('cpu')):
    previous=torch.zeros(1,3,*erp_size,device=device)
    accumulator=RGBFusionAccumulator(previous,operator.fusion_config)
    for camera in cameras:
        rgb=torch.ones(1,3,camera.height,camera.width,device=device)
        accumulator.accumulate(operator.perspective_to_erp(rgb,camera,erp_size))
    fused=accumulator.finalize();stats=coverage_statistics(fused.contributor_count)
    if stats['coverage_percent'] != 100.:
        raise ValueError('K fixed cover has gaps; inspect geometry before model loading')
    torch.testing.assert_close(fused.erp_rgb,torch.ones_like(fused.erp_rgb),atol=1e-6,rtol=1e-6)
    return stats,fused.contributor_count


@dataclass
class ERPLocalConsensusResult:
    erp_rgb: torch.Tensor
    steps: list
    local_states: list
    final_views: list
    final_consensus_erp: torch.Tensor
    coverage_counts: torch.Tensor
    final_pairwise: list
    transition_records: list
    audit: dict = field(default_factory=dict)


class ERPLocalCurrentStatePipeline:
    def __init__(self,*,backend,cameras,erp_size,warp_operator,flow_transition=True,view_order=None):
        if type(warp_operator) is not StandardWarpOperator:
            raise ValueError('K requires the existing StandardWarpOperator exactly')
        f=warp_operator.fusion_config
        if f.mode!='average' or f.weight_mode!='uniform' or warp_operator.warp_config.mode!='standard':
            raise ValueError('K requires standard warp and average/uniform RGB fusion')
        self.backend=backend;self.cameras=tuple(cameras);self.erp_size=tuple(erp_size)
        if not self.cameras:raise ValueError('K needs fixed camera slots')
        self.camera_sha256=camera_digest(self.cameras)
        self.operator=warp_operator;self.flow_transition=flow_transition
        self.view_order=list(range(len(cameras))) if view_order is None else list(view_order)
        if sorted(self.view_order)!=list(range(len(cameras))):raise ValueError('view_order must be a permutation')

    def initialize_local_states(self,seed,batch_size=1):
        generator=torch.Generator(device='cpu').manual_seed(seed)
        return [self.backend.sample_initial_native_state(batch_size=batch_size,
            native_height=self.backend.native_spatial_shape_for_rgb(c.height,c.width)[0],
            native_width=self.backend.native_spatial_shape_for_rgb(c.height,c.width)[1],generator=generator)
            for c in self.cameras]

    def _timed(self,timings,key,operation):
        if self.backend.device.type=='cuda':torch.cuda.synchronize(self.backend.device)
        started=time.perf_counter();value=operation()
        if self.backend.device.type=='cuda':torch.cuda.synchronize(self.backend.device)
        timings[key]=timings.get(key,0.)+time.perf_counter()-started
        return value

    def _project_and_fuse(self,views,timings):
        first=views[0]
        accumulator=RGBFusionAccumulator(first.new_zeros(first.shape[0],3,*self.erp_size),self.operator.fusion_config)
        contributions=[]
        for camera,rgb in zip(self.cameras,views):
            contribution=self._timed(timings,'view_to_erp',lambda:self.operator.perspective_to_erp(rgb,camera,self.erp_size))
            accumulator.accumulate(contribution);contributions.append(contribution)
        fused=accumulator.finalize()
        if not bool(fused.coverage_mask.all()):raise ValueError('Unexpected K coverage gap')
        pairs=projected_disagreement(contributions)
        return fused,pairs

    @torch.no_grad()
    def run(self,local_states,conditioning,snapshot_callback=None):
        if camera_digest(self.cameras)!=self.camera_sha256:raise AssertionError('Camera slots changed')
        states=[s.clone().float() for s in local_states];count=len(self.cameras)
        if len(states)!=count:raise ValueError('Wrong local state count')
        for camera,state in zip(self.cameras,states):
            shape=(states[0].shape[0],self.backend.native_channels,*self.backend.native_spatial_shape_for_rgb(camera.height,camera.width))
            if tuple(state.shape)!=shape:raise ValueError('K state is not a local native camera tensor')
        timesteps=self.backend.timesteps.clone();condition_hash=conditioning_digest(conditioning)
        before=self.backend.guided_prediction_count if hasattr(self.backend,'guided_prediction_count') else 0
        records=[];patch_records=[];camera_hashes=[]
        scheduler=getattr(getattr(self.backend,'pipeline',None),'scheduler',None)
        for index,timestep in enumerate(timesteps):
            camera_hashes.append(camera_digest(self.cameras))
            if camera_hashes[-1]!=self.camera_sha256:raise AssertionError('Camera slots changed during trajectory')
            timings={};pairs=[None]*count;views=[None]*count
            # Freeze all source states until every next state is constructed.
            for i in self.view_order:
                if scheduler is not None:reset_scheduler_step_state(scheduler)
                prior=getattr(self.backend,'guided_prediction_count',0)
                pairs[i]=self._timed(timings,'model',lambda:self.backend.predict_clean_and_endpoint(states[i].clone(),timestep,conditioning))
                if self.backend.guided_prediction_count-prior!=1:raise AssertionError('Exactly one prediction per view is required')
                views[i]=self._timed(timings,'vae_decode',lambda:self.backend.decode_clean(pairs[i].clean))
                if views[i].shape[1:]!=(3,self.cameras[i].height,self.cameras[i].width):raise ValueError('Wrong decoded RGB view shape')
            fused,overlaps=self._project_and_fuse(views,timings)
            coverage=coverage_statistics(fused.contributor_count)
            # Finish ALL geometry/encode operations before any transition.
            cleans=[];rgb_deltas=[]
            for camera,rgb in zip(self.cameras,views):
                crop=self._timed(timings,'erp_to_view',lambda:self.operator.erp_to_perspective(fused.erp_rgb,camera))
                rgb_deltas.append(float((crop-rgb).abs().mean()))
                cleans.append(self._timed(timings,'vae_encode',lambda:self.backend.encode_clean(crop)))
            next_states=[];diagnostics=[]
            for i,(x,clean,pair) in enumerate(zip(states,cleans,pairs)):
                nxt=interpolate_from_current_state(x,clean,pair.alpha,pair.sigma,pair.next_alpha,pair.next_sigma,flow=self.flow_transition)
                d=transition_diagnostics(x,clean,pair,nxt)
                d.update(rgb_consensus_delta_mae=rgb_deltas[i],latent_consensus_delta_mae=d['clean_consensus_delta_mae'],
                    current_state_interpolation_reconstruction_error=d['i_current_state_mismatch'],
                    next_state_update_mae=float((nxt-x).abs().mean()))
                diagnostics.append(d);patch_records.append(dict(step=index,timestep=float(timestep),camera=i,**d));next_states.append(nxt)
            if any(not bool(torch.isfinite(s).all()) for s in next_states) or not bool(torch.isfinite(fused.erp_rgb).all()):
                raise ValueError('Non-finite K state')
            stats=pairs[0].schedule_stats()
            for key in diagnostics[0]:
                values=[d[key] for d in diagnostics];stats[key+'_mean']=sum(values)/count;stats[key+'_max']=max(values)
            stats.update(guided_predictions=count,pre_fusion_rgb_overlap_mae_mean=sum(p['mae'] for p in overlaps)/max(len(overlaps),1),
                pre_fusion_rgb_overlap_mae_max=max((p['mae'] for p in overlaps),default=0.),overlap_pair_count=len(overlaps))
            records.append(StepDiagnostics(index,float(timestep),count,coverage['coverage_percent'],coverage['multi_contributor_percent'],
                float(fused.accumulated_weight.min()),float(fused.accumulated_weight.max()),float(fused.accumulated_weight.mean()),timings,stats))
            if snapshot_callback is not None:snapshot_callback(index,fused.erp_rgb)
            if index==len(timesteps)-1:final_consensus=fused.erp_rgb
            states=next_states
            del pairs,views,cleans,fused,overlaps,pair
            for name in ('last_model_prediction','last_clean_prediction'):
                if hasattr(self.backend,name):setattr(self.backend,name,None)
        if camera_digest(self.cameras)!=self.camera_sha256 or not torch.equal(timesteps,self.backend.timesteps) or conditioning_digest(conditioning)!=condition_hash:
            raise AssertionError('Fixed cameras, schedule, or conditioning mutated')
        final_views=[self.backend.decode_native_canvas(s) for s in states]
        final,pairwise=self._project_and_fuse(final_views,{})
        if not bool(torch.isfinite(final.erp_rgb).all()):raise ValueError('Non-finite terminal ERP')
        actual=self.backend.guided_prediction_count-before
        if actual!=count*len(timesteps):raise AssertionError('Unexpected K denoiser count')
        return ERPLocalConsensusResult(final.erp_rgb,records,states,final_views,final_consensus,final.contributor_count,pairwise,patch_records,
            dict(experiment='K',guided_predictions=actual,expected_guided_predictions=count*len(timesteps),extra_denoiser_calls=0,
                synchronous=True,camera_sha256=self.camera_sha256,camera_hashes_per_step=camera_hashes,
                conditioning_sha256=condition_hash,persistent_state='local native noisy tensors only',
                erp_state='transient clean RGB consensus only',global_native_state_persisted=False,
                vae_residual_correction=False,fixed_initial_noise_renoising=False,warp='standard',fusion='average/uniform',
                consensus_transition='preserve_current_state'))


def generate_erp_local_current_state(config,backend,**kwargs):
    config.validate();prepared=prepare_native_backend(config,backend)
    cameras=build_camera_sampler(config.sampling,config.view,config.experiment.seed).sample(0,config.generation.num_inference_steps)
    operator=StandardWarpOperator(config.warp,config.fusion)
    geometry_preflight(cameras,(config.erp.height,config.erp.width),operator,backend.device)
    pipeline=ERPLocalCurrentStatePipeline(backend=backend,cameras=cameras,erp_size=(config.erp.height,config.erp.width),
        warp_operator=operator,flow_transition=config.model.pipeline!='sd2')
    local=pipeline.initialize_local_states(config.experiment.seed,config.generation.batch_size)
    cond=backend.conditioning_for_prompt_indices(prepared,[8],batch_size=config.generation.batch_size)
    return pipeline.run(local,cond)
