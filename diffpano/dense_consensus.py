"""Bounded-GPU-memory execution of K's algorithm, shared by L and M.

Only local native noisy states persist. Clean views/proposals are temporarily
hosted on CPU for diagnostics; no predictions or per-view artifacts are saved.
"""
import hashlib
import json
import time
from dataclasses import dataclass
import torch
from diffpano.conditioning import camera_prompt_indices
from diffpano.current_state_transition import interpolate_from_current_state
from diffpano.erp_local_consensus import ERPLocalCurrentStatePipeline, camera_digest, native_digest
from diffpano.fusion import RGBFusionAccumulator
from diffpano.pipelines.base import reset_scheduler_step_state
from diffpano.projection import erp_to_perspective_grid
from diffpano.trajectory import conditioning_digest


class ScalarAggregate:
    def __init__(self):
        self.total = 0.; self.count = 0; self.maximum = 0.
    def add(self, mean, maximum=None):
        mean = float(mean); self.total += mean; self.count += 1
        self.maximum = max(self.maximum, mean if maximum is None else float(maximum))
    def values(self):
        return dict(mean=self.total/max(self.count,1), max=self.maximum)


def compact_series(steps):
    result = {}
    for key in steps[0]:
        values = [s[key] for s in steps]
        result[key] = dict(mean=sum(values)/len(values), max=max(values),
            first=values[0], midpoint=values[len(values)//2], last=values[-1])
    return result


def prepare_camera_conditioning(backend, prepared, cameras, policy, batch_size=1):
    if policy == 'spherediff_directional':
        indices = camera_prompt_indices(cameras, prepared.prompt_directions).tolist()
    elif policy == 'original_k_global_slot_8':
        indices = [8]*len(cameras)
    else:
        raise ValueError('Unknown dense prompt policy')
    # At most 20 small conditioning objects, reused throughout the trajectory.
    cache = {}
    for i, slot in enumerate(indices):
        if slot not in cache:
            cache[slot] = (backend.conditioning_for_cameras(prepared,[cameras[i]],batch_size=batch_size)
                if policy == 'spherediff_directional' else
                backend.conditioning_for_prompt_indices(prepared,[slot],batch_size=batch_size))
    return [cache[slot] for slot in indices], indices


@dataclass
class DenseConsensusResult:
    erp_rgb: torch.Tensor
    metrics: dict
    stage_seconds: dict
    audit: dict


class DenseERPLocalCurrentStatePipeline(ERPLocalCurrentStatePipeline):
    """The same two-pass Jacobi algorithm for either fixed-camera/prompt policy."""
    def initialize_local_states(self, seed, batch_size=1):
        generator = torch.Generator(device='cpu').manual_seed(seed)
        states = []
        for camera in self.cameras:
            h,w = self.backend.native_spatial_shape_for_rgb(camera.height,camera.width)
            state = self.backend.sample_initial_native_state(batch_size=batch_size,native_height=h,native_width=w,generator=generator)
            states.append(state.cpu())
        return states

    @torch.no_grad()
    def precompute_geometry(self, minimum=1):
        count = torch.zeros(1,1,*self.erp_size,device=self.backend.device)
        for camera in self.cameras:
            contribution = self.operator.perspective_to_erp(torch.ones(1,3,camera.height,camera.width,device=self.backend.device),camera,self.erp_size)
            count.add_((contribution.valid_mask > 0).float())
            erp_to_perspective_grid(camera,*self.erp_size,device=self.backend.device,cache=self.operator.cache)
        if int(count.min()) < minimum:
            raise AssertionError('Actual GPU projection masks fail the required coverage')
        return count

    def _accumulator(self, batch_size):
        return RGBFusionAccumulator(torch.zeros(batch_size,3,*self.erp_size,device=self.backend.device), self.operator.fusion_config)

    @torch.no_grad()
    def run_dense(self, local_states, conditionings, *, required_minimum=1, progress=None):
        if camera_digest(self.cameras) != self.camera_sha256:
            raise AssertionError('Camera slots changed')
        count = len(self.cameras)
        if len(local_states) != count or len(conditionings) != count:
            raise ValueError('One native state and conditioning object required per camera')
        states = [s.detach().cpu().clone().float() for s in local_states]
        batch = states[0].shape[0]
        for c,s in zip(self.cameras,states):
            expected = (batch,self.backend.native_channels,*self.backend.native_spatial_shape_for_rgb(c.height,c.width))
            if tuple(s.shape) != expected:
                raise ValueError('Persistent states must be local perspective-native tensors')
        initial_hash = hashlib.sha256(''.join(native_digest(s) for s in states).encode()).hexdigest()
        condition_hashes = [conditioning_digest(c) for c in conditionings]
        timesteps = self.backend.timesteps.clone()
        scheduler = getattr(getattr(self.backend,'pipeline',None),'scheduler',None)
        before = getattr(self.backend,'guided_prediction_count',0)
        totals = {}; step_summaries = []
        for step,timestep in enumerate(timesteps):
            if camera_digest(self.cameras) != self.camera_sha256:
                raise AssertionError('Camera slots changed during trajectory')
            timings = {}; acc = self._accumulator(batch)
            originals = [None]*count; native_cleans = [None]*count; coefficients = None
            # All denoiser predictions see frozen states; no next state yet exists.
            for i in self.view_order:
                if scheduler is not None:
                    reset_scheduler_step_state(scheduler)
                x = states[i].to(self.backend.device)
                calls = self.backend.guided_prediction_count if hasattr(self.backend,'guided_prediction_count') else 0
                pair = self._timed(timings,'model',lambda:self.backend.predict_clean_and_endpoint(x.clone(),timestep,conditionings[i]))
                if self.backend.guided_prediction_count-calls != 1:
                    raise AssertionError('Exactly one prediction per camera/timestep')
                coeff = tuple(v.detach().clone() if isinstance(v,torch.Tensor) else v for v in (pair.alpha,pair.sigma,pair.next_alpha,pair.next_sigma))
                if coefficients is None:
                    coefficients = coeff
                elif any(float(a) != float(b) for a,b in zip(coefficients,coeff)):
                    raise AssertionError('View-specific schedule mutation')
                native_cleans[i] = pair.clean.detach().cpu()
                rgb = self._timed(timings,'decode',lambda:self.backend.decode_clean(pair.clean))
                if tuple(rgb.shape) != (batch,3,self.cameras[i].height,self.cameras[i].width):
                    raise ValueError('Wrong clean RGB view dimensions')
                originals[i] = rgb.detach().cpu()
                contribution = self._timed(timings,'view_to_erp',lambda:self.operator.perspective_to_erp(rgb,self.cameras[i],self.erp_size))
                acc.accumulate(contribution)
                del pair,rgb,contribution,x
            fused = acc.finalize()
            if int(fused.contributor_count.min()) < required_minimum:
                raise AssertionError('Per-step coverage below requirement')
            overlap_mask = fused.contributor_count > 1
            statistics = {k:ScalarAggregate() for k in ('rgb_consensus_delta','native_consensus_delta','current_state_error','next_state_update','overlap_view_to_consensus')}
            next_states = [None]*count
            for i in range(count):
                x = states[i].to(self.backend.device)
                original = originals[i].to(self.backend.device)
                crop = self._timed(timings,'erp_to_view',lambda:self.operator.erp_to_perspective(fused.erp_rgb,self.cameras[i]))
                delta = (crop-original).abs()
                statistics['rgb_consensus_delta'].add(delta.mean(),delta.max())
                # O(N) view-to-fused-consensus reprojection diagnostic. No N^2 list.
                contribution = self._timed(timings,'diagnostic_reprojection',lambda:self.operator.perspective_to_erp(original,self.cameras[i],self.erp_size))
                valid = (contribution.valid_mask > 0) & overlap_mask
                pixels = int(valid.sum())
                if pixels:
                    error = (contribution.rgb-fused.erp_rgb).abs()*valid
                    statistics['overlap_view_to_consensus'].add(error.sum()/(3*pixels),error.max())
                clean = self._timed(timings,'encode',lambda:self.backend.encode_clean(crop))
                d = (clean-native_cleans[i].to(self.backend.device)).abs()
                statistics['native_consensus_delta'].add(d.mean(),d.max())
                a,s,an,sn = coefficients
                nxt = interpolate_from_current_state(x,clean,a,s,an,sn,flow=self.flow_transition)
                reconstructed = a*clean+s*((x-a*clean)/s) if float(s) else a*clean
                error = (reconstructed-x).abs()
                torch.testing.assert_close(reconstructed,x,atol=2e-6,rtol=2e-6)
                statistics['current_state_error'].add(error.mean(),error.max())
                update = (nxt-x).abs()
                statistics['next_state_update'].add(update.mean(),update.max())
                if not bool(torch.isfinite(nxt).all()):
                    raise ValueError('Non-finite next state')
                next_states[i] = nxt.cpu()
                originals[i] = native_cleans[i] = None
                del x,original,crop,contribution,clean,nxt,d,error,reconstructed,update
            if not bool(torch.isfinite(fused.erp_rgb).all()):
                raise ValueError('Non-finite clean ERP')
            step_summaries.append({key+'_'+name:value for key,metric in statistics.items() for name,value in metric.values().items()})
            for key,value in timings.items():
                totals[key] = totals.get(key,0.)+value
            states = next_states  # Jacobi commit after all transitions.
            del fused,acc,originals,native_cleans
            for name in ('last_model_prediction','last_clean_prediction'):
                if hasattr(self.backend,name):setattr(self.backend,name,None)
            if progress is not None:progress(step,len(timesteps),step_summaries[-1])
        # Terminal local states, rather than the last clean consensus, define output.
        acc = self._accumulator(batch)
        for i,camera in enumerate(self.cameras):
            rgb = self.backend.decode_native_canvas(states[i].to(self.backend.device))
            acc.accumulate(self.operator.perspective_to_erp(rgb,camera,self.erp_size))
            del rgb
        final = acc.finalize()
        if not bool(torch.isfinite(final.erp_rgb).all()):raise ValueError('Non-finite terminal ERP')
        if camera_digest(self.cameras) != self.camera_sha256 or not torch.equal(timesteps,self.backend.timesteps):
            raise AssertionError('Camera or schedule mutation')
        if condition_hashes != [conditioning_digest(c) for c in conditionings]:
            raise AssertionError('Conditioning mutated')
        calls = self.backend.guided_prediction_count-before
        if calls != count*len(timesteps):raise AssertionError('Unexpected denoiser count')
        return DenseConsensusResult(final.erp_rgb,compact_series(step_summaries),totals,
            dict(guided_predictions=calls,expected_guided_predictions=count*len(timesteps),extra_denoiser_calls=0,
                 camera_sha256=self.camera_sha256,cameras_fixed=True,conditioning_fixed=True,
                 conditioning_sha256=hashlib.sha256(''.join(condition_hashes).encode()).hexdigest(),
                 initial_local_sha256=initial_hash,initialization='one CPU Gaussian stream in fixed camera order',
                 synchronous=True,persistent_state='local native noisy',erp_state='transient clean RGB',
                 transition='preserve_current_state',vae_residual=False,fixed_noise_renoising=False,
                 spherical_latent=False,erp_latent=False,warp='standard',fusion='average',
                 diagnostics='O(N) view-to-fused-ERP overlap; raw RGB; mean across view means, max pixel error'))
