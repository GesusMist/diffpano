"""Synchronous global denoising whose only persistent state is ERP RGB."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from diffpano.camera import CameraSampler, build_camera_sampler
from diffpano.config import ExperimentConfig, FusionConfig
from diffpano.diagnostics import TensorStatisticsAccumulator, tensor_state_statistics
from diffpano.fusion import FusionResult, RGBFusionAccumulator
from diffpano.initialization import initialize_erp_canvas, load_directional_prompts
from diffpano.pipelines.base import ViewDenoiser
from diffpano.projection import ProjectionCache
from diffpano.warp import WarpOperator, build_warp_operator


@dataclass
class StepDiagnostics:
    step_index: int
    scheduler_timestep: float
    num_cameras: int
    coverage_percent: float
    multi_contributor_percent: float
    weight_min: float
    weight_max: float
    weight_mean: float
    timings_seconds: Dict[str, float] = field(default_factory=dict)
    state_statistics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ERPGenerationResult:
    erp_rgb: torch.Tensor
    steps: List[StepDiagnostics]


class ERPRGBPipeline:
    """Jacobi-style orchestration over a frozen ERP RGB source each timestep."""

    def __init__(
        self,
        *,
        camera_sampler: CameraSampler,
        warp_operator: WarpOperator,
        fusion_config: FusionConfig,
        view_denoiser: ViewDenoiser,
        view_batch_size: int = 1,
        diagnostics_writer: Optional[Any] = None,
        measure_performance: bool = False,
    ):
        if view_batch_size < 1:
            raise ValueError("view_batch_size must be positive")
        self.camera_sampler = camera_sampler
        self.warp_operator = warp_operator
        self.fusion_config = fusion_config
        self.view_denoiser = view_denoiser
        self.view_batch_size = view_batch_size
        self.diagnostics_writer = diagnostics_writer
        self.measure_performance = measure_performance

    def _sync(self) -> None:
        if self.measure_performance and self.view_denoiser.device.type == "cuda":
            torch.cuda.synchronize(self.view_denoiser.device)

    def _timed(self, timings: Dict[str, float], name: str, operation):
        self._sync()
        started = time.perf_counter()
        output = operation()
        self._sync()
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - started
        return output

    @torch.no_grad()
    def run(self, erp_rgb: torch.Tensor, prepared_conditioning: Any) -> ERPGenerationResult:
        if erp_rgb.ndim != 4 or erp_rgb.shape[1] != 3:
            raise ValueError("Persistent state must be ERP RGB [B,3,H,W]")
        if erp_rgb.dtype != torch.float32:
            erp_rgb = erp_rgb.float()
        batch_size, _, erp_height, erp_width = erp_rgb.shape
        timesteps = self.view_denoiser.timesteps
        step_records: List[StepDiagnostics] = []

        for step_index, timestep in enumerate(timesteps):
            timings: Dict[str, float] = {}
            # This reference is frozen for the complete camera cover.  No operation
            # below writes into it; only the accumulator is mutable.
            erp_source = erp_rgb
            collect_state = bool(getattr(self.view_denoiser, "state_diagnostics_enabled", False))
            state_statistics: Dict[str, float] = {}
            view_source_stats = TensorStatisticsAccumulator() if collect_state else None
            model_prediction_stats = TensorStatisticsAccumulator() if collect_state else None
            view_next_stats = TensorStatisticsAccumulator() if collect_state else None
            if collect_state:
                state_statistics.update(
                    tensor_state_statistics(erp_source, "erp_source", spatial_correlation=True)
                )
            cameras = self._timed(
                timings,
                "camera_projection_preparation",
                lambda: self.camera_sampler.sample(step_index, len(timesteps)),
            )
            if self.diagnostics_writer is not None:
                self.diagnostics_writer.on_cameras(step_index, cameras)
            accumulator = RGBFusionAccumulator(erp_source, self.fusion_config)

            for start in range(0, len(cameras), self.view_batch_size):
                camera_chunk = cameras[start:start + self.view_batch_size]
                views = self._timed(
                    timings,
                    "erp_to_perspective_warp",
                    lambda: torch.cat(
                        [self.warp_operator.erp_to_perspective(erp_source, camera) for camera in camera_chunk],
                        dim=0,
                    ),
                )
                if collect_state:
                    view_source_stats.add(views)
                conditioning = self.view_denoiser.conditioning_for_cameras(
                    prepared_conditioning, camera_chunk, batch_size=batch_size
                )
                proposals = self._timed(
                    timings,
                    "local_rgb_denoise",
                    lambda: self.view_denoiser.denoise_step(views, timestep, conditioning),
                )
                for name, value in getattr(self.view_denoiser, "last_timings", {}).items():
                    timings[name] = timings.get(name, 0.0) + value
                if proposals.shape != views.shape:
                    raise ValueError(
                        f"ViewDenoiser returned {tuple(proposals.shape)}, expected {tuple(views.shape)}"
                    )
                if collect_state:
                    view_next_stats.add(proposals)
                    model_prediction = getattr(self.view_denoiser, "last_model_prediction", None)
                    if model_prediction is not None:
                        model_prediction_stats.add(model_prediction)
                        self.view_denoiser.last_model_prediction = None
                split_views = views.split(batch_size, dim=0)
                split_proposals = proposals.split(batch_size, dim=0)
                for chunk_index, (camera, source_view, proposal) in enumerate(
                    zip(camera_chunk, split_views, split_proposals)
                ):
                    contribution = self._timed(
                        timings,
                        "perspective_to_erp_warp",
                        lambda camera=camera, proposal=proposal: self.warp_operator.perspective_to_erp(
                            proposal, camera, (erp_height, erp_width)
                        ),
                    )
                    if self.diagnostics_writer is not None:
                        self.diagnostics_writer.on_view(
                            step_index,
                            start + chunk_index,
                            source_view,
                            proposal,
                            contribution,
                        )
                    self._timed(timings, "rgb_accumulation", lambda: accumulator.accumulate(contribution))

            fused: FusionResult = self._timed(timings, "rgb_fusion", accumulator.finalize)
            erp_rgb = fused.erp_rgb
            covered = fused.coverage_mask
            if collect_state:
                state_statistics.update(view_source_stats.values("view_source"))
                state_statistics.update(model_prediction_stats.values("model_prediction"))
                state_statistics.update(view_next_stats.values("view_next"))
                state_statistics.update(
                    tensor_state_statistics(erp_rgb, "erp_next", spatial_correlation=True)
                )
                source_std = state_statistics["erp_source_std"]
                state_statistics["fusion_variance_ratio"] = (
                    state_statistics["erp_next_std"] / source_std if source_std > 0 else 0.0
                )
            record = StepDiagnostics(
                step_index=step_index,
                scheduler_timestep=float(torch.as_tensor(timestep).detach().cpu()),
                num_cameras=len(cameras),
                coverage_percent=float(covered.float().mean().mul(100).cpu()),
                multi_contributor_percent=float((fused.contributor_count > 1).float().mean().mul(100).cpu()),
                weight_min=float(fused.accumulated_weight.min().cpu()),
                weight_max=float(fused.accumulated_weight.max().cpu()),
                weight_mean=float(fused.accumulated_weight.mean().cpu()),
                timings_seconds=timings,
                state_statistics=state_statistics,
            )
            step_records.append(record)
            if self.diagnostics_writer is not None:
                self.diagnostics_writer.on_step(step_index, erp_source, fused, record)

        return ERPGenerationResult(erp_rgb=erp_rgb, steps=step_records)


def generate_erp_rgb(
    config: ExperimentConfig,
    view_denoiser: ViewDenoiser,
    *,
    diagnostics_writer: Optional[Any] = None,
) -> ERPGenerationResult:
    """Prepare all fixed state once, then run the new main DiffPano algorithm."""

    config.validate()
    sampler = build_camera_sampler(config.sampling, config.view, config.experiment.seed)
    warp = build_warp_operator(
        config.warp,
        config.fusion,
        ProjectionCache(
            max_entries=config.performance.projection_cache_max_entries,
            cpu_fallback=config.performance.projection_cache_cpu_fallback,
        ),
    )
    view_denoiser.prepare(
        num_steps=config.generation.num_inference_steps,
        view_height=config.view.height,
        view_width=config.view.width,
    )
    prompts = load_directional_prompts(config.prompt.path)
    negative = ""
    if config.prompt.negative_path:
        with open(config.prompt.negative_path, "r", encoding="utf-8") as handle:
            negative = handle.read().strip()
    prepared = view_denoiser.prepare_prompt_conditioning(prompts, negative)
    try:
        generator = torch.Generator(device=view_denoiser.device).manual_seed(config.experiment.seed)
    except RuntimeError:
        generator = torch.Generator(device=view_denoiser.device.type).manual_seed(config.experiment.seed)
    initial = initialize_erp_canvas(
        config.initialization,
        batch_size=config.generation.batch_size,
        height=config.erp.height,
        width=config.erp.width,
        device=view_denoiser.device,
        generator=generator,
        camera_sampler=sampler,
        warp_operator=warp,
        fusion_config=config.fusion,
        view_denoiser=view_denoiser,
    )
    pipeline = ERPRGBPipeline(
        camera_sampler=sampler,
        warp_operator=warp,
        fusion_config=config.fusion,
        view_denoiser=view_denoiser,
        view_batch_size=config.performance.view_batch_size,
        diagnostics_writer=diagnostics_writer,
        measure_performance=config.debug.measure_performance,
    )
    return pipeline.run(initial, prepared)
