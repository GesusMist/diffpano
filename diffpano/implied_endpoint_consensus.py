"""Planar Jacobi consensus in clean RGB with step-local native endpoints."""
import time
from dataclasses import asdict, dataclass, field
from copy import deepcopy

import torch

from diffpano.config import FusionConfig
from diffpano.native_multidiffusion import NativePlanarFusionAccumulator, prepare_native_backend
from diffpano.overlap import OverlapDisagreement
from diffpano.pipelines.base import reset_scheduler_step_state
from diffpano.pipelines.endpoints import EndpointBackend
from diffpano.planar import PlanarFusionAccumulator, build_planar_patch_layout, extract_planar_patch
from diffpano.planar_pipeline import PlanarGenerationResult, PlanarStepDiagnostics
from diffpano.trajectory import conditioning_digest
from diffpano.vae_residual import vae_residual, fuse_native_residuals


@dataclass
class ImpliedConsensusResult(PlanarGenerationResult):
    local_states: list = field(default_factory=list)
    fused_clean_rgb: torch.Tensor = None
    audit: dict = field(default_factory=dict)


class PlanarImpliedEndpointConsensusPipeline:
    """Only local noisy states persist; clean RGB and endpoints are step-local.

    Patch batching is deliberately unsupported. Accumulation uses canonical
    geometric order, independently of model processing order. Final output is
    the selected RGB fusion of decoded terminal local states, after the complete
    final reconstruction (including DDIM's actual terminal alpha).
    """
    def __init__(self, *, native_config, backend, patch_order=None, pixel_native=False, bridge=None, residual_correction=False, fusion_config=None):
        if not isinstance(backend, EndpointBackend):
            raise TypeError('Backend must implement validated endpoint prediction')
        self.backend = backend
        self.pixel_native = pixel_native
        self.bridge = bridge
        self.residual_correction = residual_correction
        self.fusion_config = deepcopy(fusion_config) if fusion_config is not None else FusionConfig(mode='average', weight_mode='uniform')
        if self.fusion_config.mode not in {'average', 'detail_preserving_average'} or self.fusion_config.weight_mode != 'uniform':
            raise ValueError('Implied endpoint consensus requires average or detail_preserving_average with uniform spatial weights')
        if residual_correction and (pixel_native or bridge is not None):
            raise ValueError("Training-free residual correction requires a latent backend and no learned bridge")
        if pixel_native and bridge is not None:
            raise ValueError("Pixel-native consensus does not use a VAE bridge")
        n = native_config
        factor = backend.native_spatial_factor
        if not isinstance(factor, int) or factor < 1:
            raise ValueError('Native-to-RGB spatial factor must be a positive integer')
        self.native_layout = build_planar_patch_layout(n.canvas_height, n.canvas_width, n.patch_size, n.stride)
        self.rgb_layout = build_planar_patch_layout(n.canvas_height*factor, n.canvas_width*factor,
                                                   n.patch_size*factor, n.stride*factor)
        for a, b in zip(self.native_layout.patches, self.rgb_layout.patches):
            if (b.y, b.x, b.size) != (a.y*factor, a.x*factor, a.size*factor):
                raise AssertionError('Native/RGB mapping is not exact')
        order = list(range(self.native_layout.num_patches)) if patch_order is None else list(patch_order)
        if sorted(order) != list(range(self.native_layout.num_patches)):
            raise ValueError('patch_order must be a permutation')
        self.patch_order = order

    def initialize_local_states(self, initial_native):
        n = self.native_layout
        if initial_native.ndim != 4 or tuple(initial_native.shape[1:]) != (self.backend.native_channels, n.canvas_height, n.canvas_width):
            raise ValueError('Initial global field does not match native geometry')
        locals_ = [extract_planar_patch(initial_native, p).clone() for p in n.patches]
        if not all(torch.equal(s, extract_planar_patch(initial_native, p)) for s, p in zip(locals_, n.patches)):
            raise AssertionError('Local initialization is not bit-identical to global crops')
        return locals_

    def fuse_rgb(self, proposals):
        first = proposals[0]
        layout = self.rgb_layout
        canvas = first.new_zeros(first.shape[0], 3, layout.canvas_height, layout.canvas_width)
        if self.fusion_config.mode == 'detail_preserving_average':
            accumulator = PlanarFusionAccumulator(canvas, self.fusion_config)
            for patch, proposal in zip(layout.patches, proposals):
                accumulator.accumulate(proposal, patch)
            result = accumulator.finalize()
            return result.canvas_rgb, result.accumulated_weight
        accumulator = NativePlanarFusionAccumulator(canvas)
        for patch, proposal in zip(layout.patches, proposals):
            accumulator.accumulate(proposal, patch)
        return accumulator.finalize(), accumulator.denominator

    @torch.no_grad()
    def run(self, local_states, prepared_conditioning):
        backend = self.backend
        count = self.native_layout.num_patches
        if len(local_states) != count or len(backend.timesteps) < 1:
            raise ValueError('Wrong local state count or empty schedule')
        states = [state.clone().float() for state in local_states]
        expected = (states[0].shape[0], backend.native_channels, self.native_layout.patch_size, self.native_layout.patch_size)
        if any(tuple(s.shape) != expected for s in states):
            raise ValueError('Invalid local native state shape')
        conditioning = backend.conditioning_for_prompt_indices(prepared_conditioning, [8], batch_size=states[0].shape[0])
        condition_hash = conditioning_digest(conditioning)
        timesteps = backend.timesteps.clone()
        records, evaluations = [], 0
        scheduler = getattr(getattr(backend, 'pipeline', None), 'scheduler', None)
        for index, timestep in enumerate(timesteps):
            started = time.perf_counter()
            endpoints, clean_rgb = [None]*count, [None]*count
            # Predict all patches from frozen source states, before any fusion.
            for i in self.patch_order:
                if scheduler is not None:
                    reset_scheduler_step_state(scheduler)
                before = getattr(backend, 'guided_prediction_count', 0)
                endpoints[i] = backend.predict_clean_and_endpoint(states[i].clone(), timestep, conditioning)
                delta = getattr(backend, 'guided_prediction_count', 0) - before
                if delta != 1:
                    raise AssertionError('Expected one guided prediction per patch per timestep')
                evaluations += delta
                clean_rgb[i] = endpoints[i].clean if self.pixel_native else backend.decode_clean(endpoints[i].clean)
            residuals, fused_residual = None, None
            if self.residual_correction:
                # All residuals use the SAME decoded clean proposals used by RGB fusion.
                # Canonical geometric order makes accumulation independent of model order.
                residuals = [vae_residual(pair.clean, backend.encode_clean(rgb))
                             for pair, rgb in zip(endpoints, clean_rgb)]
                fused_residual = fuse_native_residuals(self.native_layout, residuals)
            overlap = OverlapDisagreement(self.rgb_layout)
            for patch, rgb in zip(self.rgb_layout.patches, clean_rgb):
                overlap.add(patch, rgb)
            fused, weights = self.fuse_rgb(clean_rgb)
            corrections, roundtrips, next_states, bridge_corrections = [], [], [], []
            residual_corrections = []
            for i, patch in enumerate(self.rgb_layout.patches):
                crop = extract_planar_patch(fused, patch)
                corrections.append(float((crop-clean_rgb[i]).abs().mean()))
                clean = crop if self.pixel_native else backend.encode_clean(crop)
                if not self.pixel_native:
                    # Diagnostic-only extra VAE encode; never an extra denoiser call.
                    if residuals is None:
                        own_roundtrip = backend.encode_clean(clean_rgb[i])
                        roundtrips.append(float((own_roundtrip-endpoints[i].clean).abs().mean()))
                    else:
                        roundtrips.append(float(residuals[i].abs().mean()))
                if fused_residual is not None:
                    correction = extract_planar_patch(fused_residual, self.native_layout.patches[i])
                    residual_corrections.append(float(correction.abs().mean()))
                    clean = clean + correction
                if self.bridge is not None:
                    corrected = self.bridge(clean.clone(), timestep)
                    bridge_corrections.append(float((corrected-clean).abs().mean()))
                    clean = corrected
                next_states.append(endpoints[i].reconstruct_next(clean=clean))
            if not all(bool(torch.isfinite(s).all()) for s in next_states) or not bool(torch.isfinite(fused).all()):
                raise ValueError('Non-finite consensus state at step {}'.format(index))
            stats = endpoints[0].schedule_stats()
            stats.update({'pre_fusion_rgb_'+k: v for k, v in overlap.values().items()})
            stats['consensus_correction_mae_mean'] = sum(corrections)/count
            stats['guided_predictions'] = count
            if residual_corrections:
                stats.update(local_vae_residual_mae_mean=sum(roundtrips)/count,
                             local_vae_residual_mae_max=max(roundtrips),
                             fused_residual_mae_mean=float(fused_residual.abs().mean()),
                             fused_residual_std=float(fused_residual.std(unbiased=False)),
                             residual_correction_magnitude=sum(residual_corrections)/count)
            if bridge_corrections:
                stats['bridge_correction_mae_mean'] = sum(bridge_corrections)/count
            if roundtrips:
                stats['clean_native_roundtrip_mae'] = sum(roundtrips)/count
            records.append(PlanarStepDiagnostics(index, float(timestep), count, 100.,
                float((weights>1).float().mean()*100), float(weights.min()), float(weights.max()), float(weights.mean()),
                {'consensus_step': time.perf_counter()-started}, stats))
            states = next_states
            # No endpoint is retained for a later timestep or fused globally.
            del endpoints, clean_rgb, overlap, residuals, fused_residual
            if self.residual_correction: del correction
            for name in ('last_model_prediction', 'last_clean_prediction'):
                if hasattr(backend, name): setattr(backend, name, None)
        if not torch.equal(timesteps, backend.timesteps) or condition_hash != conditioning_digest(conditioning):
            raise AssertionError('Schedule or shared conditioning mutated')
        final_rgb, _ = self.fuse_rgb([s if self.pixel_native else backend.decode_native_canvas(s) for s in states])
        if not bool(torch.isfinite(final_rgb).all()):
            raise ValueError('Non-finite decoded terminal image')
        return ImpliedConsensusResult(canvas_rgb=final_rgb, steps=records, local_states=states, fused_clean_rgb=fused,
            audit=dict(guided_predictions=evaluations, expected_guided_predictions=count*len(timesteps),
                       diagnostic_extra_denoiser_evaluations=0, synchronous=True, endpoint_lifetime='one timestep',
                       global_native_state_persisted=False, conditioning_sha256=condition_hash, bridge_enabled=self.bridge is not None,
                       training_free_residual_correction=self.residual_correction,
                       residual_fusion='temporary uniform native-coordinate canvas' if self.residual_correction else None,
                       rgb_fusion=asdict(self.fusion_config),
                       output=self.fusion_config.mode+' RGB fusion of decoded terminal local states', patch_batch_size=1))


def generate_planar_implied_endpoint_consensus(config, backend, *, diagnostics_writer=None):
    config.validate()
    prepared = prepare_native_backend(config, backend)
    n = config.native_multidiffusion
    initial = backend.sample_initial_native_state(batch_size=config.generation.batch_size,
        native_height=n.canvas_height, native_width=n.canvas_width,
        generator=torch.Generator(device=backend.device).manual_seed(config.experiment.seed))
    pipeline = PlanarImpliedEndpointConsensusPipeline(native_config=n, backend=backend,
        pixel_native=config.model.pipeline == 'pixeldit', fusion_config=config.fusion)
    local_states = pipeline.initialize_local_states(initial)
    del initial
    result = pipeline.run(local_states, prepared)
    result.audit['initial_local_states_equal_global_native_crops'] = True
    return result
