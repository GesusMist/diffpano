"""Synchronous diffusion over a rectangular RGB canvas and exact square crops."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from diffpano.config import ExperimentConfig, FusionConfig, PlanarConfig
from diffpano.diagnostics import TensorStatisticsAccumulator, tensor_state_statistics
from diffpano.initialization import load_directional_prompts
from diffpano.noise import FixedPatchNoiseBank
from diffpano.pipelines.base import ViewDenoiser
from diffpano.pipelines.clean_prediction import CleanPredictionBackend
from diffpano.planar import (
    PlanarFusionAccumulator,
    PlanarFusionResult,
    PlanarPatch,
    PlanarPatchLayout,
    build_planar_patch_layout_for_step,
    extract_planar_patch,
    planar_prompt_indices,
)


@dataclass
class PlanarStepDiagnostics:
    step_index: int
    scheduler_timestep: float
    num_patches: int
    coverage_percent: float
    multi_contributor_percent: float
    weight_min: float
    weight_max: float
    weight_mean: float
    timings_seconds: Dict[str, float] = field(default_factory=dict)
    state_statistics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PlanarGenerationResult:
    canvas_rgb: torch.Tensor
    steps: List[PlanarStepDiagnostics]
    fixed_noise_identities: Dict[int, str] = field(default_factory=dict)


class _TimedPlanarPipeline:
    backend: Any
    measure_performance: bool

    def _sync(self) -> None:
        if self.measure_performance and self.backend.device.type == "cuda":
            torch.cuda.synchronize(self.backend.device)

    def _timed(self, timings: Dict[str, float], name: str, operation):
        self._sync()
        started = time.perf_counter()
        output = operation()
        self._sync()
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - started
        return output

    @staticmethod
    def _ordered_patches(
        layout: PlanarPatchLayout, patch_order: Optional[Sequence[int]]
    ) -> Sequence[PlanarPatch]:
        if patch_order is None:
            return layout.patches
        if sorted(patch_order) != list(range(layout.num_patches)):
            raise ValueError("patch_order must be a permutation of all layout indices")
        return tuple(layout.patches[index] for index in patch_order)


class PlanarRGBPipeline(_TimedPlanarPipeline):
    """Jacobi-style RGB-state updates using only exact slicing and placement."""

    def __init__(
        self,
        *,
        planar_config: PlanarConfig,
        fusion_config: FusionConfig,
        backend: ViewDenoiser,
        patch_batch_size: int = 1,
        diagnostics_writer: Optional[Any] = None,
        measure_performance: bool = False,
        patch_order: Optional[Sequence[int]] = None,
    ):
        if patch_batch_size < 1:
            raise ValueError("patch_batch_size must be positive")
        self.planar_config = planar_config
        self.fusion_config = fusion_config
        self.backend = backend
        self.patch_batch_size = patch_batch_size
        self.diagnostics_writer = diagnostics_writer
        self.measure_performance = measure_performance
        self.patch_order = patch_order

    @torch.no_grad()
    def run(
        self, canvas_rgb: torch.Tensor, prepared_conditioning: Any
    ) -> PlanarGenerationResult:
        expected_size = (self.planar_config.height, self.planar_config.width)
        if (
            canvas_rgb.ndim != 4
            or canvas_rgb.shape[1] != 3
            or canvas_rgb.shape[-2:] != expected_size
        ):
            raise ValueError(
                "Persistent planar state must be RGB [B,3,planar.height,planar.width]"
            )
        canvas_rgb = canvas_rgb.float()
        batch_size = canvas_rgb.shape[0]
        records: List[PlanarStepDiagnostics] = []

        for step_index, timestep in enumerate(self.backend.timesteps):
            timings: Dict[str, float] = {}
            source = canvas_rgb
            layout = build_planar_patch_layout_for_step(
                self.planar_config, step_index
            )
            patches = self._ordered_patches(layout, self.patch_order)
            accumulator = PlanarFusionAccumulator(source, self.fusion_config)
            collect = bool(
                getattr(self.backend, "state_diagnostics_enabled", False)
                or (
                    self.diagnostics_writer is not None
                    and getattr(self.diagnostics_writer.config, "enabled", False)
                )
            )
            statistics: Dict[str, float] = {}
            source_patch_stats = TensorStatisticsAccumulator() if collect else None
            prediction_stats = TensorStatisticsAccumulator() if collect else None
            proposal_stats = TensorStatisticsAccumulator() if collect else None
            if collect:
                statistics.update(
                    tensor_state_statistics(
                        source, "canvas_source", spatial_correlation=True
                    )
                )

            for start in range(0, len(patches), self.patch_batch_size):
                patch_chunk = patches[start : start + self.patch_batch_size]
                source_patches = self._timed(
                    timings,
                    "exact_patch_crop",
                    lambda: torch.cat(
                        [extract_planar_patch(source, patch) for patch in patch_chunk],
                        dim=0,
                    ),
                )
                if collect:
                    source_patch_stats.add(source_patches)
                prompt_slots = planar_prompt_indices(
                    layout, patch_chunk, self.planar_config.prompt_assignment
                )
                conditioning = self.backend.conditioning_for_prompt_indices(
                    prepared_conditioning,
                    prompt_slots,
                    batch_size=batch_size,
                )
                proposals = self._timed(
                    timings,
                    "local_rgb_denoise",
                    lambda: self.backend.denoise_step(
                        source_patches, timestep, conditioning
                    ),
                )
                for name, value in getattr(self.backend, "last_timings", {}).items():
                    timings[name] = timings.get(name, 0.0) + value
                if proposals.shape != source_patches.shape:
                    raise ValueError(
                        f"ViewDenoiser returned {tuple(proposals.shape)}, expected {tuple(source_patches.shape)}"
                    )
                if collect:
                    proposal_stats.add(proposals)
                    model_prediction = getattr(
                        self.backend, "last_model_prediction", None
                    )
                    if model_prediction is not None:
                        prediction_stats.add(model_prediction)
                        self.backend.last_model_prediction = None
                for offset, (patch, before, proposal) in enumerate(
                    zip(
                        patch_chunk,
                        source_patches.split(batch_size, dim=0),
                        proposals.split(batch_size, dim=0),
                    )
                ):
                    self._timed(
                        timings,
                        "exact_patch_accumulation",
                        lambda patch=patch, proposal=proposal: accumulator.accumulate(
                            proposal, patch
                        ),
                    )
                    if self.diagnostics_writer is not None:
                        method = getattr(
                            self.diagnostics_writer, "on_planar_patch", None
                        )
                        if method is not None:
                            method(step_index, patch.index, before, proposal)

            fused: PlanarFusionResult = self._timed(
                timings, "planar_rgb_fusion", accumulator.finalize
            )
            canvas_rgb = fused.canvas_rgb
            if collect:
                statistics.update(source_patch_stats.values("extracted_patch"))
                statistics.update(prediction_stats.values("model_prediction"))
                statistics.update(proposal_stats.values("patch_proposal"))
                statistics.update(
                    tensor_state_statistics(
                        canvas_rgb, "fused_canvas", spatial_correlation=True
                    )
                )
                source_std = statistics["canvas_source_std"]
                statistics["fusion_variance_ratio"] = (
                    statistics["fused_canvas_std"] / source_std
                    if source_std > 0
                    else 0.0
                )
            record = _planar_step_record(
                step_index, timestep, layout, fused, timings, statistics
            )
            records.append(record)
            if self.diagnostics_writer is not None:
                method = getattr(self.diagnostics_writer, "on_planar_step", None)
                if method is not None:
                    method(step_index, source, fused, record, clean=False)
        return PlanarGenerationResult(canvas_rgb=canvas_rgb, steps=records)


class PlanarX0ConsensusPipeline(_TimedPlanarPipeline):
    """Persist fused predicted-clean RGB and reuse native noise per patch slot."""

    def __init__(
        self,
        *,
        planar_config: PlanarConfig,
        fusion_config: FusionConfig,
        consensus_config: Any,
        backend: CleanPredictionBackend,
        patch_batch_size: int = 1,
        diagnostics_writer: Optional[Any] = None,
        measure_performance: bool = False,
        seed: int = 0,
        patch_order: Optional[Sequence[int]] = None,
    ):
        if patch_batch_size < 1:
            raise ValueError("patch_batch_size must be positive")
        if not isinstance(backend, CleanPredictionBackend):
            raise TypeError("Backend does not implement the clean-prediction protocol")
        if not callable(getattr(backend, "conditioning_for_prompt_indices", None)):
            raise TypeError("Backend does not support direct prompt-index conditioning")
        self.planar_config = planar_config
        self.fusion_config = fusion_config
        self.consensus_config = consensus_config
        self.backend = backend
        self.patch_batch_size = patch_batch_size
        self.diagnostics_writer = diagnostics_writer
        self.measure_performance = measure_performance
        self.seed = seed
        self.patch_order = patch_order

    @torch.no_grad()
    def _step(
        self,
        *,
        step_index: int,
        timestep: Any,
        layout: PlanarPatchLayout,
        noise_bank: FixedPatchNoiseBank,
        prepared_conditioning: Any,
        batch_size: int,
        clean_source: Optional[torch.Tensor],
    ):
        bootstrap = clean_source is None
        previous = (
            torch.zeros(
                batch_size,
                3,
                layout.canvas_height,
                layout.canvas_width,
                device=self.backend.device,
                dtype=torch.float32,
            )
            if bootstrap
            else clean_source
        )
        timings: Dict[str, float] = {}
        accumulator = PlanarFusionAccumulator(previous, self.fusion_config)
        patches = self._ordered_patches(layout, self.patch_order)
        collect = bool(
            getattr(self.backend, "state_diagnostics_enabled", False)
            or (
                self.diagnostics_writer is not None
                and getattr(self.diagnostics_writer.config, "enabled", False)
            )
        )
        statistics: Dict[str, float] = {}
        clean_patch_stats = TensorStatisticsAccumulator() if collect else None
        noise_stats = TensorStatisticsAccumulator() if collect else None
        noisy_stats = TensorStatisticsAccumulator() if collect else None
        prediction_stats = TensorStatisticsAccumulator() if collect else None
        native_stats = TensorStatisticsAccumulator() if collect else None
        rgb_stats = TensorStatisticsAccumulator() if collect else None
        if collect and clean_source is not None:
            statistics.update(
                tensor_state_statistics(clean_source, "canvas_source")
            )

        for start in range(0, len(patches), self.patch_batch_size):
            patch_chunk = patches[start : start + self.patch_batch_size]
            indices = [patch.index for patch in patch_chunk]
            if bootstrap:
                clean_patches = None
                clean_native = None
            else:
                clean_patches = self._timed(
                    timings,
                    "exact_clean_patch_crop",
                    lambda: torch.cat(
                        [
                            extract_planar_patch(clean_source, patch)
                            for patch in patch_chunk
                        ],
                        dim=0,
                    ),
                )
                clean_native = self._timed(
                    timings,
                    "encode_clean",
                    lambda: self.backend.encode_clean(clean_patches),
                )
                if collect:
                    clean_patch_stats.add(clean_patches)
            fixed_noise = noise_bank.get(indices, device=self.backend.device)
            if collect:
                noise_stats.add(fixed_noise)
            if bootstrap:
                construct_noisy = lambda: self.backend.make_initial_noisy_state(
                    fixed_noise, timestep
                )
            else:
                construct_noisy = lambda: self.backend.add_fixed_noise(
                    clean_native, fixed_noise, timestep
                )
            noisy_native = self._timed(
                timings, "construct_noisy_native", construct_noisy
            )
            if collect:
                noisy_stats.add(noisy_native)
            prompt_slots = planar_prompt_indices(
                layout, patch_chunk, self.planar_config.prompt_assignment
            )
            conditioning = self.backend.conditioning_for_prompt_indices(
                prepared_conditioning,
                prompt_slots,
                batch_size=batch_size,
            )
            predicted_clean_native = self._timed(
                timings,
                "predict_clean_native",
                lambda: self.backend.predict_clean_native(
                    noisy_native, timestep, conditioning
                ),
            )
            predicted_clean_rgb = self._timed(
                timings,
                "decode_predicted_clean",
                lambda: self.backend.decode_clean(predicted_clean_native).float(),
            )
            expected = (
                len(patch_chunk) * batch_size,
                3,
                layout.patch_size,
                layout.patch_size,
            )
            if predicted_clean_rgb.shape != expected:
                raise ValueError(
                    f"Clean backend returned {tuple(predicted_clean_rgb.shape)}, expected {expected}"
                )
            for name, value in getattr(self.backend, "last_timings", {}).items():
                timings[name] = timings.get(name, 0.0) + value
            if collect:
                model_prediction = getattr(
                    self.backend, "last_model_prediction", None
                )
                if model_prediction is not None:
                    prediction_stats.add(model_prediction)
                    self.backend.last_model_prediction = None
                native_stats.add(predicted_clean_native)
                rgb_stats.add(predicted_clean_rgb)
            split_source = (
                [None] * len(patch_chunk)
                if clean_patches is None
                else clean_patches.split(batch_size, dim=0)
            )
            for patch, before, predicted in zip(
                patch_chunk,
                split_source,
                predicted_clean_rgb.split(batch_size, dim=0),
            ):
                self._timed(
                    timings,
                    "exact_clean_patch_accumulation",
                    lambda patch=patch, predicted=predicted: accumulator.accumulate(
                        predicted, patch
                    ),
                )
                if self.diagnostics_writer is not None:
                    method = getattr(
                        self.diagnostics_writer, "on_planar_patch", None
                    )
                    if method is not None:
                        method(step_index, patch.index, before, predicted)

        fused = self._timed(
            timings, "planar_clean_rgb_fusion", accumulator.finalize
        )
        if collect:
            statistics.update(clean_patch_stats.values("extracted_clean_patch"))
            statistics.update(noise_stats.values("fixed_noise"))
            statistics.update(noisy_stats.values("noisy_native_state"))
            statistics.update(prediction_stats.values("model_prediction"))
            statistics.update(native_stats.values("predicted_clean_native"))
            statistics.update(rgb_stats.values("predicted_clean_rgb"))
            statistics.update(
                tensor_state_statistics(fused.canvas_rgb, "fused_canvas")
            )
            if clean_source is not None:
                source_std = statistics["canvas_source_std"]
                statistics["fusion_variance_ratio"] = (
                    statistics["fused_canvas_std"] / source_std
                    if source_std > 0
                    else 0.0
                )
        record = _planar_step_record(
            step_index, timestep, layout, fused, timings, statistics
        )
        if self.diagnostics_writer is not None:
            method = getattr(self.diagnostics_writer, "on_planar_step", None)
            if method is not None:
                method(step_index, clean_source, fused, record, clean=True)
        return fused.canvas_rgb, record

    @torch.no_grad()
    def run(
        self, prepared_conditioning: Any, *, batch_size: int
    ) -> PlanarGenerationResult:
        timesteps = self.backend.timesteps
        if len(timesteps) < 1:
            raise ValueError("Clean consensus requires at least one timestep")
        layout = build_planar_patch_layout_for_step(self.planar_config, 0)
        noise_bank = FixedPatchNoiseBank(
            self.consensus_config,
            backend=self.backend,
            num_cameras=layout.num_patches,
            batch_size=batch_size,
            height=layout.patch_size,
            width=layout.patch_size,
            seed=self.seed,
        )
        records: List[PlanarStepDiagnostics] = []
        clean_canvas: Optional[torch.Tensor] = None
        for step_index, timestep in enumerate(timesteps):
            step_layout = build_planar_patch_layout_for_step(
                self.planar_config, step_index
            )
            if step_layout != layout:
                raise ValueError(
                    "fixed per-patch noise requires an unchanged planar layout"
                )
            clean_canvas, record = self._step(
                step_index=step_index,
                timestep=timestep,
                layout=step_layout,
                noise_bank=noise_bank,
                prepared_conditioning=prepared_conditioning,
                batch_size=batch_size,
                clean_source=clean_canvas,
            )
            records.append(record)
        return PlanarGenerationResult(
            canvas_rgb=clean_canvas,
            steps=records,
            fixed_noise_identities=dict(noise_bank.identities),
        )


def _planar_step_record(
    step_index: int,
    timestep: Any,
    layout: PlanarPatchLayout,
    fused: PlanarFusionResult,
    timings: Dict[str, float],
    statistics: Dict[str, float],
) -> PlanarStepDiagnostics:
    return PlanarStepDiagnostics(
        step_index=step_index,
        scheduler_timestep=float(torch.as_tensor(timestep).detach().cpu()),
        num_patches=layout.num_patches,
        coverage_percent=float(fused.coverage_mask.float().mean().mul(100).cpu()),
        multi_contributor_percent=float(
            (fused.contributor_count > 1).float().mean().mul(100).cpu()
        ),
        weight_min=float(fused.accumulated_weight.min().cpu()),
        weight_max=float(fused.accumulated_weight.max().cpu()),
        weight_mean=float(fused.accumulated_weight.mean().cpu()),
        timings_seconds=timings,
        state_statistics=statistics,
    )


def _negative_prompt(config: ExperimentConfig) -> str:
    if not config.prompt.negative_path:
        return ""
    with open(config.prompt.negative_path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def _prepare_planar_backend(config: ExperimentConfig, backend: Any) -> Any:
    backend.prepare(
        num_steps=config.generation.num_inference_steps,
        view_height=config.planar.patch_size,
        view_width=config.planar.patch_size,
    )
    prompts = load_directional_prompts(config.prompt.path)
    return backend.prepare_prompt_conditioning(prompts, _negative_prompt(config))


def initialize_planar_canvas(
    config: ExperimentConfig,
    backend: ViewDenoiser,
    generator: torch.Generator,
) -> torch.Tensor:
    initialization = config.initialization
    shape = (
        config.generation.batch_size,
        3,
        config.planar.height,
        config.planar.width,
    )
    if initialization.mode in {
        "erp_rgb_noise",
        "canvas_rgb_noise",
        "pixel_gaussian",
    }:
        canvas = torch.randn(
            *shape,
            device=backend.device,
            dtype=torch.float32,
            generator=generator,
        )
        canvas = canvas * initialization.std + initialization.mean
        if initialization.clamp:
            canvas = canvas.clamp(
                initialization.clamp_min, initialization.clamp_max
            )
        return canvas
    if initialization.mode != "latent_native_bootstrap":
        raise ValueError(f"Unsupported initialization mode {initialization.mode!r}")
    previous = torch.zeros(*shape, device=backend.device, dtype=torch.float32)
    layout = build_planar_patch_layout_for_step(config.planar, 0)
    accumulator = PlanarFusionAccumulator(previous, config.fusion)
    for patch in layout.patches:
        rgb = backend.sample_native_rgb(
            batch_size=config.generation.batch_size,
            height=patch.size,
            width=patch.size,
            generator=generator,
        )
        accumulator.accumulate(rgb, patch)
    return accumulator.finalize().canvas_rgb


def _generator_for(device: torch.device, seed: int) -> torch.Generator:
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except RuntimeError:
        return torch.Generator(device=device.type).manual_seed(seed)


def generate_planar_rgb(
    config: ExperimentConfig,
    backend: ViewDenoiser,
    *,
    diagnostics_writer: Optional[Any] = None,
) -> PlanarGenerationResult:
    config.validate()
    prepared = _prepare_planar_backend(config, backend)
    initial = initialize_planar_canvas(
        config,
        backend,
        _generator_for(backend.device, config.experiment.seed),
    )
    pipeline = PlanarRGBPipeline(
        planar_config=config.planar,
        fusion_config=config.fusion,
        backend=backend,
        patch_batch_size=config.performance.view_batch_size,
        diagnostics_writer=diagnostics_writer,
        measure_performance=config.debug.measure_performance,
    )
    return pipeline.run(initial, prepared)


def generate_planar_x0_consensus(
    config: ExperimentConfig,
    backend: CleanPredictionBackend,
    *,
    diagnostics_writer: Optional[Any] = None,
) -> PlanarGenerationResult:
    config.validate()
    prepared = _prepare_planar_backend(config, backend)
    pipeline = PlanarX0ConsensusPipeline(
        planar_config=config.planar,
        fusion_config=config.fusion,
        consensus_config=config.global_pipeline.clean_consensus,
        backend=backend,
        patch_batch_size=config.performance.view_batch_size,
        diagnostics_writer=diagnostics_writer,
        measure_performance=config.debug.measure_performance,
        seed=config.experiment.seed,
    )
    return pipeline.run(prepared, batch_size=config.generation.batch_size)
