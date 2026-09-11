"""Exact native-state planar MultiDiffusion with ordinary first-order sampling."""

import time
from dataclasses import dataclass

import torch

from diffpano.initialization import load_directional_prompts
from diffpano.overlap import OverlapDisagreement
from diffpano.pipelines.native_state import NativeStateBackend
from diffpano.planar import build_planar_patch_layout, extract_planar_patch
from diffpano.planar_pipeline import PlanarGenerationResult, PlanarStepDiagnostics, _negative_prompt


class NativePlanarFusionAccumulator:
    """Uniform arithmetic averaging of arbitrary [B,C,H,W] native proposals."""

    def __init__(self, previous):
        if previous.ndim != 4:
            raise ValueError("Native canvas must have shape [B,C,H,W]")
        self.numerator = torch.zeros_like(previous, dtype=torch.float32)
        self.denominator = torch.zeros_like(self.numerator[:, :1])

    def accumulate(self, proposal, patch):
        target = extract_planar_patch(self.numerator, patch)
        if target.shape != proposal.shape:
            raise ValueError("Native proposal shape does not match exact placement")
        target.add_(proposal.to(target))
        extract_planar_patch(self.denominator, patch).add_(1)

    def finalize(self):
        if not bool((self.denominator > 0).all()):
            raise ValueError("Native layout left uncovered cells")
        return self.numerator / self.denominator


@dataclass
class NativeGenerationResult(PlanarGenerationResult):
    native_canvas: torch.Tensor = None


def prepare_native_backend(config, backend):
    """Prepare scheduler at the actual local model resolution and one prompt."""
    if not isinstance(backend, NativeStateBackend):
        raise TypeError("Backend does not implement native diffusion operations")
    p = config.native_multidiffusion.patch_size
    height, width = backend.rgb_spatial_shape_for_native(p, p)
    backend.prepare(num_steps=config.generation.num_inference_steps,
                    view_height=height, view_width=width)
    # Existing prompt banks have 20 slots. Repeat the global (equatorial) prompt
    # so all slots, including those used by trajectory controls, are identical.
    prompt = load_directional_prompts(config.prompt.path)[2]
    return backend.prepare_prompt_conditioning([prompt] * 5, _negative_prompt(config))


class NativeMultiDiffusionPipeline:
    def __init__(self, *, native_config, backend, overlap_disagreement=False, patch_order=None, clean_rgb_overlap_disagreement=False):
        self.config = native_config
        self.backend = backend
        self.overlap_disagreement = overlap_disagreement
        self.clean_rgb_overlap_disagreement = clean_rgb_overlap_disagreement
        self.layout = build_planar_patch_layout(native_config.canvas_height, native_config.canvas_width,
                                                native_config.patch_size, native_config.stride)
        order = list(range(self.layout.num_patches)) if patch_order is None else list(patch_order)
        if sorted(order) != list(range(self.layout.num_patches)):
            raise ValueError("patch_order must be a permutation of all patches")
        self.patches = [self.layout.patches[i] for i in order]

    @torch.no_grad()
    def run(self, initial_native, prepared_conditioning):
        expected = (self.backend.native_channels, self.layout.canvas_height, self.layout.canvas_width)
        if initial_native.ndim != 4 or tuple(initial_native.shape[1:]) != expected:
            raise ValueError("Native initial canvas dimensions/channels differ from configured layout")
        if len(self.backend.timesteps) < 1:
            raise ValueError("Native sampling requires at least one timestep")
        state = initial_native.float()
        records = []
        conditioning = self.backend.conditioning_for_prompt_indices(
            prepared_conditioning, [8], batch_size=state.shape[0])
        for index, timestep in enumerate(self.backend.timesteps):
            started = time.perf_counter()
            accumulator = NativePlanarFusionAccumulator(state)
            overlap = OverlapDisagreement(self.layout) if self.overlap_disagreement else None
            rgb_overlap = None
            if self.clean_rgb_overlap_disagreement:
                factor = self.backend.native_spatial_factor
                rgb_layout = build_planar_patch_layout(self.layout.canvas_height*factor,
                    self.layout.canvas_width*factor, self.layout.patch_size*factor, self.config.stride*factor)
                rgb_overlap = OverlapDisagreement(rgb_layout)
            for patch in self.patches:
                # Clone protects the persistent Jacobi source from in-place backends.
                local = extract_planar_patch(state, patch).clone()
                if rgb_overlap is None:
                    proposal = self.backend.denoise_native_step(local, timestep, conditioning)
                else:
                    # Reuse this ordinary native step's prediction; decoding is diagnostic only.
                    proposal, endpoints = self.backend.native_step_with_endpoints(local, timestep, conditioning)
                    rgb_overlap.add(rgb_layout.patches[patch.index], self.backend.decode_clean(endpoints.clean))
                    del endpoints

                if overlap is not None:
                    overlap.add(patch, proposal)
                accumulator.accumulate(proposal, patch)
            state = accumulator.finalize()
            if not bool(torch.isfinite(state).all()):
                raise ValueError(f"Non-finite native canvas at step {index}")
            weights = accumulator.denominator
            records.append(PlanarStepDiagnostics(
                index, float(timestep), self.layout.num_patches, 100.0,
                float((weights > 1).float().mean() * 100),
                float(weights.min()), float(weights.max()), float(weights.mean()),
                {"native_step": time.perf_counter() - started},
                {**(overlap.values() if overlap is not None else {}),
                 **({'pre_fusion_rgb_'+k: v for k, v in rgb_overlap.values().items()} if rgb_overlap is not None else {})},
            ))
        # Exactly one global decode, after every native timestep has completed.
        rgb = self.backend.decode_native_canvas(state)
        expected_rgb = (state.shape[0], 3, *self.backend.rgb_spatial_shape_for_native(*state.shape[-2:]))
        if tuple(rgb.shape) != expected_rgb:
            raise ValueError(f"Final native decode returned {tuple(rgb.shape)}, expected {expected_rgb}")
        return NativeGenerationResult(canvas_rgb=rgb, steps=records, native_canvas=state)


def generate_planar_native_multidiffusion(config, backend, *, diagnostics_writer=None):
    config.validate()
    prepared = prepare_native_backend(config, backend)
    geometry = config.native_multidiffusion
    generator = torch.Generator(device=backend.device).manual_seed(config.experiment.seed)
    initial = backend.sample_initial_native_state(
        batch_size=config.generation.batch_size, native_height=geometry.canvas_height,
        native_width=geometry.canvas_width, generator=generator)
    return NativeMultiDiffusionPipeline(native_config=geometry, backend=backend,
        overlap_disagreement=config.debug.overlap_disagreement).run(initial, prepared)
