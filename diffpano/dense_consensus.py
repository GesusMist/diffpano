"""Bounded-GPU-memory execution of K's algorithm, shared by L, M, N and O.

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
from diffpano.projection import erp_to_perspective_grid, perspective_to_erp_grid
from diffpano.warp import StandardWarpOperator, LaplacianPyramidWarpOperator, _level_camera, _level_size
from dataclasses import replace
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


def snapshot_schedule(num_steps):
    """Integer ceil(percent * steps / 100), grouping duplicate milestones."""
    if num_steps < 1:
        raise ValueError('A snapshot schedule needs positive step count')
    mapping = {}
    for percentage in range(10, 100, 10):
        completed = (percentage * num_steps + 99) // 100
        mapping.setdefault(completed, []).append(percentage)
    return mapping


class DenseERPLocalCurrentStatePipeline(ERPLocalCurrentStatePipeline):
    """The same two-pass Jacobi algorithm for either fixed-camera/prompt policy."""
    def __init__(self, *, backend, cameras, erp_size, warp_operator, flow_transition=True, view_order=None):
        # K's constructor and its strict guards remain unchanged. L/M still use it.
        if warp_operator.fusion_config.mode == 'average':
            super().__init__(backend=backend,cameras=cameras,erp_size=erp_size,
                warp_operator=warp_operator,flow_transition=flow_transition,view_order=view_order)
        else:
            f = warp_operator.fusion_config
            if (f.mode, f.weight_mode) not in {('detail_preserving_average', 'uniform'), ('weighted_average', 'spherediff_center')}:
                raise ValueError('Dense DPA requires uniform weights')
            expected = LaplacianPyramidWarpOperator if warp_operator.warp_config.mode == 'lpw' else StandardWarpOperator
            if type(warp_operator) is not expected:
                raise ValueError('Dense warp type does not match its config')
            self.backend=backend;self.cameras=tuple(cameras);self.erp_size=tuple(erp_size)
            if not self.cameras:raise ValueError('Dense consensus requires cameras')
            self.camera_sha256=camera_digest(self.cameras)
            self.operator=warp_operator;self.flow_transition=flow_transition
            self.view_order=list(range(len(cameras))) if view_order is None else list(view_order)
            if sorted(self.view_order)!=list(range(len(cameras))):raise ValueError('Invalid view order')
        self.diagnostic_operator = StandardWarpOperator(
            replace(warp_operator.warp_config,mode='standard'),warp_operator.fusion_config,warp_operator.cache)

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
            contribution = self.diagnostic_operator.perspective_to_erp(torch.ones(1,3,camera.height,camera.width,device=self.backend.device),camera,self.erp_size)
            count.add_((contribution.valid_mask > 0).float())
            erp_to_perspective_grid(camera,*self.erp_size,device=self.backend.device,cache=self.operator.cache)
            if isinstance(self.operator,LaplacianPyramidWarpOperator):
                for level in range(1,self.operator.warp_config.lpw.levels):
                    c = _level_camera(camera,level)
                    h,w = (_level_size(v,level) for v in self.erp_size)
                    perspective_to_erp_grid(c,h,w,device=self.backend.device,cache=self.operator.cache)
                    erp_to_perspective_grid(c,h,w,device=self.backend.device,cache=self.operator.cache)
        if int(count.min()) < minimum:
            raise AssertionError('Actual GPU projection masks fail the required coverage')
        return count

    def _accumulator(self, batch_size):
        previous = torch.zeros(batch_size,3,*self.erp_size,device=self.backend.device)
        dedicated = self.operator.create_fusion_accumulator(previous)
        return dedicated if dedicated is not None else RGBFusionAccumulator(previous,self.operator.fusion_config)

    def _accumulate(self, accumulator, rgb, camera, timings):
        if isinstance(self.operator,LaplacianPyramidWarpOperator):
            accumulator.accumulate(rgb,camera,timer=lambda key,fn:self._timed(timings,key,fn))
        else:
            contribution = self._timed(timings,'view_to_erp',lambda:self.operator.perspective_to_erp(rgb,camera,self.erp_size))
            self._timed(timings,'rgb_fusion',lambda:accumulator.accumulate(contribution))

    def _finalize(self, accumulator, timings):
        return self._timed(timings,'erp_reconstruction' if isinstance(self.operator,LaplacianPyramidWarpOperator) else 'rgb_finalize',accumulator.finalize)

    def _local_clean_residual(self, clean, decoded_rgb, timings):
        """Optional local bridge hook; historical trajectories have no residual."""
        return None

    def _consensus_native_clean(self, rgb, local_residual, timings):
        return self._timed(timings, 'encode', lambda: self.backend.encode_clean(rgb))

    @torch.no_grad()
    def run_dense(self, local_states, conditionings, *, required_minimum=1, progress=None, snapshot_callback=None, expected_initial_sha256=None, stage_audit=None):
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
        if expected_initial_sha256 is not None and initial_hash != expected_initial_sha256:
            raise AssertionError('Initial local states differ from L')
        milestones = snapshot_schedule(len(self.backend.timesteps))
        condition_hashes = [conditioning_digest(c) for c in conditionings]
        timesteps = self.backend.timesteps.clone()
        scheduler = getattr(getattr(self.backend,'pipeline',None),'scheduler',None)
        before = getattr(self.backend,'guided_prediction_count',0)
        totals = {}; step_summaries = []
        last_clean = None; audit_seconds = 0.
        for step,timestep in enumerate(timesteps):
            if camera_digest(self.cameras) != self.camera_sha256:
                raise AssertionError('Camera slots changed during trajectory')
            timings = {}; acc = self._accumulator(batch)
            originals = [None]*count; native_cleans = [None]*count; local_residuals = [None]*count; coefficients = None
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
                local_residuals[i] = self._local_clean_residual(pair.clean, rgb, timings)
                originals[i] = rgb.detach().cpu()
                self._accumulate(acc,rgb,self.cameras[i],timings)
                del pair,rgb,x
            fused = self._finalize(acc,timings)
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
                contribution = self._timed(timings,'diagnostic_reprojection',lambda:self.diagnostic_operator.perspective_to_erp(original,self.cameras[i],self.erp_size))
                valid = (contribution.valid_mask > 0) & overlap_mask
                pixels = int(valid.sum())
                if pixels:
                    error = (contribution.rgb-fused.erp_rgb).abs()*valid
                    statistics['overlap_view_to_consensus'].add(error.sum()/(3*pixels),error.max())
                clean = self._consensus_native_clean(crop, local_residuals[i], timings)
                if stage_audit is not None and stage_audit.wants(step+1, i):
                    started_audit = time.perf_counter()
                    devices = [self.backend.device.index or 0] if self.backend.device.type == 'cuda' else []
                    with torch.random.fork_rng(devices=devices):
                        if stage_audit.backend_name == 'flux':
                            roundtrip = self.backend.decode_clean(clean.detach().clone())
                            stage_audit.extra_decodes += 1
                        else:
                            roundtrip = crop.detach().clone()
                        stage_audit.observe(step+1, i, original.detach().clone(), crop.detach().clone(), roundtrip)
                        del roundtrip
                    audit_seconds += time.perf_counter()-started_audit
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
                originals[i] = native_cleans[i] = local_residuals[i] = None
                del x,original,crop,contribution,clean,nxt,d,error,reconstructed,update
            if not bool(torch.isfinite(fused.erp_rgb).all()):
                raise ValueError('Non-finite clean ERP')
            step_summaries.append({key+'_'+name:value for key,metric in statistics.items() for name,value in metric.values().items()})
            for key,value in timings.items():
                totals[key] = totals.get(key,0.)+value
            states = next_states  # Jacobi commit after all transitions.
            if snapshot_callback is not None and step+1 in milestones:
                # An isolated copy prevents a callback from modifying the trajectory.
                snapshot_callback(dict(requested_percentages=milestones[step+1],completed_step=step+1,
                    scheduler_timestep=float(timestep),alpha=float(coefficients[0]),sigma=float(coefficients[1]),
                    next_alpha=float(coefficients[2]),next_sigma=float(coefficients[3])),fused.erp_rgb.detach().clone())
            if stage_audit is not None and step+1 == len(timesteps):
                last_clean = fused.erp_rgb.detach().clone()
            del fused,acc,originals,native_cleans,local_residuals
            for name in ('last_model_prediction','last_clean_prediction'):
                if hasattr(self.backend,name):setattr(self.backend,name,None)
            if progress is not None:progress(step,len(timesteps),step_summaries[-1])
        # Terminal local states, rather than the last clean consensus, define output.
        acc = self._accumulator(batch)
        for i,camera in enumerate(self.cameras):
            rgb = self.backend.decode_native_canvas(states[i].to(self.backend.device))
            self._accumulate(acc,rgb,camera,{})
            del rgb
        final = self._finalize(acc,{})
        if not bool(torch.isfinite(final.erp_rgb).all()):raise ValueError('Non-finite terminal ERP')
        if camera_digest(self.cameras) != self.camera_sha256 or not torch.equal(timesteps,self.backend.timesteps):
            raise AssertionError('Camera or schedule mutation')
        if condition_hashes != [conditioning_digest(c) for c in conditionings]:
            raise AssertionError('Conditioning mutated')
        calls = self.backend.guided_prediction_count-before
        if calls != count*len(timesteps):raise AssertionError('Unexpected denoiser count')
        stage_record = {}
        if stage_audit is not None:
            started_audit = time.perf_counter()
            stage_record = stage_audit.finish(last_clean, final.erp_rgb, self.cameras, self.operator, coefficients)
            audit_seconds += time.perf_counter()-started_audit
            stage_record['diagnostic_seconds'] = audit_seconds
        return DenseConsensusResult(final.erp_rgb,compact_series(step_summaries),totals,
            dict(stage_audit=stage_record, guided_predictions=calls,expected_guided_predictions=count*len(timesteps),extra_denoiser_calls=0,
                 camera_sha256=self.camera_sha256,cameras_fixed=True,conditioning_fixed=True,
                 conditioning_sha256=hashlib.sha256(''.join(condition_hashes).encode()).hexdigest(),
                 initial_local_sha256=initial_hash,initialization='one CPU Gaussian stream in fixed camera order',
                 synchronous=True,persistent_state='local native noisy',erp_state='transient clean RGB',
                 transition='preserve_current_state',vae_residual=getattr(self, 'uses_local_vae_bridge', False),fixed_noise_renoising=False,
                 spherical_latent=False,erp_latent=False,warp=self.operator.warp_config.mode,fusion=self.operator.fusion_config.mode,
                 diagnostics='O(N) view-to-fused-ERP overlap; raw RGB; mean across view means, max pixel error'))
