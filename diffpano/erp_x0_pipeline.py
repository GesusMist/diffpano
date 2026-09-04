"""Synchronous panorama generation with a persistent predicted-clean ERP canvas."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from diffpano.camera import CameraSampler, PerspectiveCamera, build_camera_sampler
from diffpano.config import CleanConsensusConfig, ExperimentConfig, FusionConfig
from diffpano.diagnostics import TensorStatisticsAccumulator, tensor_state_statistics
from diffpano.erp_pipeline import StepDiagnostics
from diffpano.fusion import FusionResult, RGBFusionAccumulator
from diffpano.initialization import load_directional_prompts
from diffpano.noise import FixedPatchNoiseBank
from diffpano.pipelines.clean_prediction import CleanPredictionBackend
from diffpano.projection import ProjectionCache
from diffpano.warp import WarpOperator, build_warp_operator


@dataclass
class ERPX0GenerationResult:
    erp_rgb: torch.Tensor
    steps: List[StepDiagnostics]
    fixed_noise_identities: Dict[int, str] = field(default_factory=dict)


class ERPX0ConsensusPipeline:
    """Persist only fused predicted-clean RGB; all diffusion noise stays local."""

    def __init__(
        self,
        *,
        camera_sampler: CameraSampler,
        warp_operator: WarpOperator,
        fusion_config: FusionConfig,
        consensus_config: CleanConsensusConfig,
        backend: CleanPredictionBackend,
        view_batch_size: int = 1,
        diagnostics_writer: Optional[Any] = None,
        measure_performance: bool = False,
        seed: int = 0,
    ):
        if view_batch_size < 1:
            raise ValueError("view_batch_size must be positive")
        if not isinstance(backend, CleanPredictionBackend):
            raise TypeError("Backend does not implement the clean-prediction protocol")
        self.camera_sampler = camera_sampler
        self.warp_operator = warp_operator
        self.fusion_config = fusion_config
        self.consensus_config = consensus_config
        self.backend = backend
        self.view_batch_size = view_batch_size
        self.diagnostics_writer = diagnostics_writer
        self.measure_performance = measure_performance
        self.seed = seed

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

    @torch.no_grad()
    def _step(
        self,
        *,
        step_index: int,
        timestep: Any,
        cameras: Sequence[PerspectiveCamera],
        noise_bank: FixedPatchNoiseBank,
        prepared_conditioning: Any,
        batch_size: int,
        erp_size: Sequence[int],
        clean_source: Optional[torch.Tensor],
    ):
        bootstrap = clean_source is None
        previous = (
            torch.zeros(
                batch_size,
                3,
                erp_size[0],
                erp_size[1],
                device=self.backend.device,
                dtype=torch.float32,
            )
            if bootstrap
            else clean_source
        )
        timings: Dict[str, float] = {}
        accumulator = RGBFusionAccumulator(previous, self.fusion_config)
        collect = bool(
            getattr(self.backend, "state_diagnostics_enabled", False)
            or (
                self.diagnostics_writer is not None
                and getattr(self.diagnostics_writer.config, "enabled", False)
            )
        )
        clean_view_stats = TensorStatisticsAccumulator() if collect else None
        noise_stats = TensorStatisticsAccumulator() if collect else None
        noisy_stats = TensorStatisticsAccumulator() if collect else None
        model_stats = TensorStatisticsAccumulator() if collect else None
        clean_native_stats = TensorStatisticsAccumulator() if collect else None
        clean_rgb_stats = TensorStatisticsAccumulator() if collect else None

        for start in range(0, len(cameras), self.view_batch_size):
            camera_chunk = cameras[start:start + self.view_batch_size]
            indices = list(range(start, start + len(camera_chunk)))
            if bootstrap:
                clean_views = None
                clean_native = None
            else:
                clean_views = self._timed(
                    timings,
                    "clean_erp_to_perspective",
                    lambda: torch.cat(
                        [
                            self.warp_operator.erp_to_perspective(
                                clean_source, camera
                            )
                            for camera in camera_chunk
                        ],
                        dim=0,
                    ),
                )
                clean_native = self._timed(
                    timings, "encode_clean", lambda: self.backend.encode_clean(clean_views)
                )
                if collect:
                    clean_view_stats.add(clean_views)
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
            conditioning = self.backend.conditioning_for_cameras(
                prepared_conditioning, camera_chunk, batch_size=batch_size
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
            if predicted_clean_rgb.shape != (
                len(camera_chunk) * batch_size,
                3,
                camera_chunk[0].height,
                camera_chunk[0].width,
            ):
                raise ValueError("Clean backend returned an unexpected RGB view shape")
            if collect:
                model_prediction = getattr(
                    self.backend, "last_model_prediction", None
                )
                if model_prediction is not None:
                    model_stats.add(model_prediction)
                clean_native_stats.add(predicted_clean_native)
                clean_rgb_stats.add(predicted_clean_rgb)
            split_clean = (
                [None] * len(camera_chunk)
                if clean_views is None
                else clean_views.split(batch_size, dim=0)
            )
            split_predictions = predicted_clean_rgb.split(batch_size, dim=0)
            for offset, (camera, clean_view, predicted) in enumerate(
                zip(camera_chunk, split_clean, split_predictions)
            ):
                contribution = self._timed(
                    timings,
                    "predicted_clean_to_erp",
                    lambda camera=camera, predicted=predicted: self.warp_operator.perspective_to_erp(
                        predicted, camera, (erp_size[0], erp_size[1])
                    ),
                )
                self._timed(
                    timings,
                    "clean_rgb_accumulation",
                    lambda contribution=contribution: accumulator.accumulate(
                        contribution
                    ),
                )
                if self.diagnostics_writer is not None:
                    method = getattr(
                        self.diagnostics_writer, "on_clean_consensus_view", None
                    )
                    if method is not None:
                        method(
                            step_index,
                            start + offset,
                            clean_view,
                            predicted,
                            contribution,
                        )

        fused: FusionResult = self._timed(
            timings, "clean_rgb_fusion", accumulator.finalize
        )
        statistics: Dict[str, float] = {}
        if collect:
            if clean_source is not None:
                statistics.update(
                    tensor_state_statistics(clean_source, "clean_erp_source")
                )
            statistics.update(clean_view_stats.values("clean_view"))
            statistics.update(noise_stats.values("fixed_noise"))
            statistics.update(noisy_stats.values("noisy_native_state"))
            statistics.update(model_stats.values("model_prediction"))
            statistics.update(
                clean_native_stats.values("predicted_clean_native")
            )
            statistics.update(clean_rgb_stats.values("predicted_clean_rgb"))
            statistics.update(
                tensor_state_statistics(fused.erp_rgb, "fused_clean_erp")
            )
        record = StepDiagnostics(
            step_index=step_index,
            scheduler_timestep=float(torch.as_tensor(timestep).detach().cpu()),
            num_cameras=len(cameras),
            coverage_percent=float(
                fused.coverage_mask.float().mean().mul(100).cpu()
            ),
            multi_contributor_percent=float(
                (fused.contributor_count > 1).float().mean().mul(100).cpu()
            ),
            weight_min=float(fused.accumulated_weight.min().cpu()),
            weight_max=float(fused.accumulated_weight.max().cpu()),
            weight_mean=float(fused.accumulated_weight.mean().cpu()),
            timings_seconds=timings,
            state_statistics=statistics,
        )
        if self.diagnostics_writer is not None:
            method = getattr(
                self.diagnostics_writer, "on_clean_consensus_step", None
            )
            if method is not None:
                method(step_index, clean_source, fused, record)
        return fused.erp_rgb, record

    @torch.no_grad()
    def run(
        self,
        prepared_conditioning: Any,
        *,
        batch_size: int,
        erp_height: int,
        erp_width: int,
    ) -> ERPX0GenerationResult:
        """Bootstrap once from native noise, then repeatedly re-noise clean views."""

        timesteps = self.backend.timesteps
        if len(timesteps) < 1:
            raise ValueError("Clean consensus requires at least one timestep")
        cameras0 = self.camera_sampler.sample(0, len(timesteps))
        noise_bank = FixedPatchNoiseBank(
            self.consensus_config,
            backend=self.backend,
            num_cameras=len(cameras0),
            batch_size=batch_size,
            height=cameras0[0].height,
            width=cameras0[0].width,
            seed=self.seed,
        )
        records: List[StepDiagnostics] = []
        clean_erp, record = self._step(
            step_index=0,
            timestep=timesteps[0],
            cameras=cameras0,
            noise_bank=noise_bank,
            prepared_conditioning=prepared_conditioning,
            batch_size=batch_size,
            erp_size=(erp_height, erp_width),
            clean_source=None,
        )
        records.append(record)
        for step_index, timestep in enumerate(timesteps[1:], start=1):
            clean_source = clean_erp
            cameras = self.camera_sampler.sample(step_index, len(timesteps))
            if len(cameras) != noise_bank.num_cameras:
                raise ValueError(
                    "camera_index noise binding requires a stable camera-slot count"
                )
            clean_erp, record = self._step(
                step_index=step_index,
                timestep=timestep,
                cameras=cameras,
                noise_bank=noise_bank,
                prepared_conditioning=prepared_conditioning,
                batch_size=batch_size,
                erp_size=(erp_height, erp_width),
                clean_source=clean_source,
            )
            records.append(record)
        return ERPX0GenerationResult(
            erp_rgb=clean_erp,
            steps=records,
            fixed_noise_identities=dict(noise_bank.identities),
        )


def generate_erp_x0_consensus(
    config: ExperimentConfig,
    backend: CleanPredictionBackend,
    *,
    diagnostics_writer: Optional[Any] = None,
) -> ERPX0GenerationResult:
    """Prepare one backend instance and run the predicted-clean ERP pipeline."""

    config.validate()
    sampler = build_camera_sampler(
        config.sampling, config.view, config.experiment.seed
    )
    warp = build_warp_operator(
        config.warp,
        config.fusion,
        ProjectionCache(
            max_entries=config.performance.projection_cache_max_entries,
            cpu_fallback=config.performance.projection_cache_cpu_fallback,
        ),
    )
    backend.prepare(
        num_steps=config.generation.num_inference_steps,
        view_height=config.view.height,
        view_width=config.view.width,
    )
    prompts = load_directional_prompts(config.prompt.path)
    negative = ""
    if config.prompt.negative_path:
        with open(config.prompt.negative_path, "r", encoding="utf-8") as handle:
            negative = handle.read().strip()
    prepared = backend.prepare_prompt_conditioning(prompts, negative)
    pipeline = ERPX0ConsensusPipeline(
        camera_sampler=sampler,
        warp_operator=warp,
        fusion_config=config.fusion,
        consensus_config=config.global_pipeline.clean_consensus,
        backend=backend,
        view_batch_size=config.performance.view_batch_size,
        diagnostics_writer=diagnostics_writer,
        measure_performance=config.debug.measure_performance,
        seed=config.experiment.seed,
    )
    return pipeline.run(
        prepared,
        batch_size=config.generation.batch_size,
        erp_height=config.erp.height,
        erp_width=config.erp.width,
    )
