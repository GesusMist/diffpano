"""Readable orchestration of the reusable DiffPano denoising/fusion stages."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from diffpano.config import FUSION_DTYPE, PixelFusionConfig
from diffpano.diagnostics import (
    _compute_projection_diagnostics,
    _timed,
    write_pixel_fusion_diagnostics,
)
from diffpano.fusion import _fuse_views_to_erp_standard
from diffpano.lpw import forward_lpw_to_views, inverse_lpw_to_erp
from diffpano.projection import extract_views_from_erp_standard
from diffpano.reinjection import _scheduler_prediction_type, reinject_fused_latents
from diffpano.vae import TensorAdapter, _adapt_latents, build_identity_preserving_vae_target, decode_view_latents


@dataclass
class PixelFusionResult:
    fused_prev_latents: torch.Tensor
    fused_clean_latents: torch.Tensor
    fused_views_rgb: torch.Tensor
    fused_erp: torch.Tensor
    valid_mask: torch.Tensor
    contributor_count: torch.Tensor
    accumulated_weight: torch.Tensor
    timings: Dict[str, float]
    diagnostics: Dict[str, torch.Tensor] = field(default_factory=dict)

def should_apply_pixel_fusion(step_index: int, num_steps: int, config: PixelFusionConfig) -> bool:
    if not config.pixel_fusion_enabled:
        return False
    if num_steps <= 0:
        return False
    ratio = step_index / max(num_steps - 1, 1)
    if ratio < config.pixel_fusion_start_ratio or ratio > config.pixel_fusion_end_ratio:
        return False
    return step_index % config.pixel_fusion_every_n_steps == 0


def should_apply_time_travel(step_index: int, num_steps: int, config: PixelFusionConfig) -> bool:
    if not config.time_travel_enabled:
        return False
    if config.time_travel_every_n_steps < 1:
        raise ValueError("time_travel_every_n_steps must be >= 1")
    return step_index > 0 and step_index < num_steps - 1 and step_index % config.time_travel_every_n_steps == 0


def run_time_travel(*args, **kwargs):
    raise NotImplementedError(
        "Pixel-fusion time travel is configured but not wired into this minimally invasive pipeline hook. "
        "Disable time_travel_enabled or add a pipeline-specific step runner."
    )

def apply_pixel_space_fusion(
    *,
    vae: Any,
    scheduler: Any,
    timestep: torch.Tensor,
    clean_latents: torch.Tensor,
    current_latents: torch.Tensor,
    model_output: torch.Tensor,
    prev_latents: torch.Tensor,
    view_dirs: torch.Tensor,
    fovs: Union[Tuple[float, float], Sequence[Tuple[float, float]]],
    erp_height: int,
    erp_width: int,
    config: PixelFusionConfig,
    latent_to_vae_latents: Optional[TensorAdapter] = None,
    vae_latents_to_latent: Optional[TensorAdapter] = None,
    generator: Optional[torch.Generator] = None,
    diagnostic_step_index: Optional[int] = None,
    diagnostic_pipeline_name: str = "pixel_fusion",
) -> PixelFusionResult:
    """Decode predicted-clean view latents, fuse in temporary ERP RGB, encode, and reinject.

    clean/current/model/prev latents use the scheduler representation expected by the original write-back path.
    The optional adapters convert that representation to/from VAE image latents before decoding and after encoding.
    """

    config.validate()
    timings: Dict[str, float] = {}
    if config.reinjection_mode == "noise_consistent":
        prediction_type = _scheduler_prediction_type(scheduler)
        if prediction_type != "flow_prediction":
            raise ValueError(
                "noise_consistent reinjection currently requires flow_prediction so the scheduler state can be "
                f"preserved exactly; got prediction_type={prediction_type!r}"
            )

    vae_clean_latents = _adapt_latents(clean_latents, latent_to_vae_latents)
    view_images = _timed(
        timings,
        "vae_decode",
        lambda: decode_view_latents(vae, vae_clean_latents, config).to(dtype=FUSION_DTYPE),
        synchronize=config.measure_performance,
    )

    if config.warp_mode == "standard":
        aggregate = _fuse_views_to_erp_standard(
            view_images,
            view_dirs,
            fovs,
            erp_height,
            erp_width,
            config,
            timings=timings,
        )
        fused_views, view_valid_mask = _timed(
            timings,
            "erp_to_view_or_forward_lpw",
            lambda: extract_views_from_erp_standard(aggregate.fused_values, aggregate.valid_output_mask, view_images, view_dirs, fovs, config),
            synchronize=config.measure_performance,
        )
    elif config.warp_mode == "lpw":
        aggregate = inverse_lpw_to_erp(
            view_images,
            view_dirs,
            fovs,
            erp_height,
            erp_width,
            config,
            timings=timings,
        )
        fused_views, view_valid_mask = _timed(
            timings,
            "erp_to_view_or_forward_lpw",
            lambda: forward_lpw_to_views(aggregate.fused_values, aggregate.valid_output_mask, view_images, view_dirs, fovs, config),
            synchronize=config.measure_performance,
        )
    else:
        raise ValueError(f"Unsupported warp_mode={config.warp_mode!r}")

    vae_bridge = _timed(
        timings,
        "vae_encode",
        lambda: build_identity_preserving_vae_target(
            vae,
            clean_latents,
            view_images,
            fused_views,
            config,
            latent_to_vae_latents=latent_to_vae_latents,
            vae_latents_to_latent=vae_latents_to_latent,
        ),
        synchronize=config.measure_performance,
    )
    fused_clean_latents_fp32 = vae_bridge.target_clean_latents
    latent_valid_mask = F.interpolate(
        view_valid_mask,
        size=vae_bridge.fused_roundtrip_vae_latents.shape[-2:],
        mode="area",
    ).clamp(0, 1)
    if vae_latents_to_latent is not None:
        latent_valid_mask = vae_latents_to_latent(
            latent_valid_mask.expand_as(vae_bridge.fused_roundtrip_vae_latents)
        )
    fused_prev_latents = _timed(
        timings,
        "reinjection",
        lambda: reinject_fused_latents(
            clean_latents,
            fused_clean_latents_fp32,
            prev_latents,
            model_output,
            torch.zeros((), device=current_latents.device, dtype=FUSION_DTYPE),
            config,
            valid_mask=latent_valid_mask,
            scheduler=scheduler,
            timestep=timestep,
            current_latents=current_latents,
        ),
        synchronize=config.measure_performance,
    )
    for key in ("projection_or_inverse_lpw", "overlap_fusion", "erp_reconstruction", "time_travel"):
        timings.setdefault(key, 0.0)

    fused_clean_latents = fused_clean_latents_fp32

    diagnostics: Dict[str, torch.Tensor] = {}
    if config.save_diagnostics or config.save_masks or config.save_intermediates:
        diagnostics = {
            "fused_erp": aggregate.fused_values.detach(),
            "valid_mask": aggregate.valid_output_mask.detach(),
            "contributor_count": aggregate.contributor_count.detach(),
            "accumulated_weight": aggregate.accumulated_weight.detach(),
            "overlap_mask": (aggregate.contributor_count > 1).to(aggregate.fused_values.dtype).detach(),
            "sampled_camera_directions": view_dirs.detach(),
            "latent_delta_norm": (fused_clean_latents_fp32 - clean_latents.float()).norm().detach()[None],
            "vae_roundtrip_error_norm": vae_bridge.vae_roundtrip_error_norm.detach().reshape(1),
            "fusion_delta_norm": vae_bridge.fusion_delta_norm.detach().reshape(1),
            "base_scheduler_update_norm": (
                prev_latents.float() - current_latents.float()
            ).norm().detach().reshape(1),
            "actual_reinjection_norm": (
                fused_prev_latents.float() - prev_latents.float()
            ).norm().detach().reshape(1),
        }
        diagnostics["reinjection_to_scheduler_update_ratio"] = (
            diagnostics["actual_reinjection_norm"]
            / diagnostics["base_scheduler_update_norm"].clamp_min(config.dpa_eps)
        )
        diagnostics.update(
            {f"timing_{key}_seconds": aggregate.fused_values.new_tensor([value]) for key, value in timings.items()}
        )
        if config.save_diagnostics:
            diagnostics.update(
                _compute_projection_diagnostics(
                    view_images,
                    view_dirs,
                    fovs,
                    erp_height,
                    erp_width,
                    config,
                )
            )
        write_pixel_fusion_diagnostics(
            diagnostics,
            config,
            step_index=diagnostic_step_index,
            pipeline_name=diagnostic_pipeline_name,
        )

    return PixelFusionResult(
        fused_prev_latents=fused_prev_latents,
        fused_clean_latents=fused_clean_latents,
        fused_views_rgb=fused_views,
        fused_erp=aggregate.fused_values,
        valid_mask=aggregate.valid_output_mask,
        contributor_count=aggregate.contributor_count,
        accumulated_weight=aggregate.accumulated_weight,
        timings=timings,
        diagnostics=diagnostics,
    )
