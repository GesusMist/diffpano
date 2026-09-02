"""Model-independent overlap aggregation and RGB fusion."""

import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import torch

from experiments.legacy_spherical.diffpano_legacy.config import FUSION_DTYPE, PixelFusionConfig
from experiments.legacy_spherical.diffpano_legacy.diagnostics import _timed
from experiments.legacy_spherical.diffpano_legacy.projection import (
    _get_perspective_to_erp_grid,
    _normalize_fovs,
    _sample_perspective_image,
)


@dataclass
class OverlapAggregationResult:
    fused_values: torch.Tensor
    accumulated_weight: torch.Tensor
    contributor_count: torch.Tensor
    valid_output_mask: torch.Tensor

def create_patch_weight_map(
    height: int,
    width: int,
    mode: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 1e-6,
) -> torch.Tensor:
    dtype = FUSION_DTYPE
    y = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    if mode == "uniform":
        weight = torch.ones_like(xx)
    elif mode == "cosine":
        weight = torch.cos(xx * math.pi / 2).clamp_min(0) * torch.cos(yy * math.pi / 2).clamp_min(0)
    elif mode == "gaussian":
        sigma = 0.5
        weight = torch.exp(-0.5 * (xx.square() + yy.square()) / (sigma * sigma))
    elif mode == "distance_to_boundary":
        dist_x = 1 - xx.abs()
        dist_y = 1 - yy.abs()
        weight = torch.minimum(dist_x, dist_y).clamp_min(0)
        max_value = weight.max().clamp_min(eps)
        weight = weight / max_value
    else:
        raise ValueError(f"Unsupported weight_mode={mode!r}")
    return weight[None, None]


def _get_patch_weight_map(
    height: int,
    width: int,
    config: PixelFusionConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = ("patch_weight", config.weight_mode, height, width, str(device), str(dtype))
    if key not in config.projection_cache.weights:
        config.projection_cache.weights[key] = create_patch_weight_map(height, width, config.weight_mode, device=device, dtype=dtype)
    return config.projection_cache.weights[key]


def detail_preserving_average(
    values: torch.Tensor,
    masks: torch.Tensor,
    weights: torch.Tensor,
    *,
    alpha: float,
    power: float,
    eps: float,
    dim: int = 0,
) -> torch.Tensor:
    values = values.to(dtype=FUSION_DTYPE)
    masks = masks.to(device=values.device, dtype=FUSION_DTYPE)
    weights = weights.to(device=values.device, dtype=FUSION_DTYPE)
    effective_weight = masks * weights
    ordinary_den = effective_weight.sum(dim=dim).clamp_min(eps)
    ordinary = (values * effective_weight).sum(dim=dim) / ordinary_den

    detail_weight = effective_weight * (values.abs() + eps).pow(power)
    detail_den = detail_weight.sum(dim=dim).clamp_min(eps)
    detail = (values * detail_weight).sum(dim=dim) / detail_den
    return ordinary + alpha * (detail - ordinary)


def aggregate_overlap_contributions(
    values: torch.Tensor,
    masks: torch.Tensor,
    weights: Optional[torch.Tensor],
    mode: str,
    *,
    dpa_alpha: float = 1.0,
    dpa_power: float = 1.0,
    dpa_eps: float = 1e-6,
) -> OverlapAggregationResult:
    values = values.to(dtype=FUSION_DTYPE)
    masks = masks.to(device=values.device, dtype=FUSION_DTYPE)
    if weights is not None:
        weights = weights.to(device=values.device, dtype=FUSION_DTYPE)
    if weights is None or mode == "average":
        weights = torch.ones_like(masks)
    effective_weight = masks * weights
    accumulated_weight = effective_weight.sum(dim=0)
    contributor_count = (masks > 0).sum(dim=0).to(values.dtype)
    valid_output_mask = accumulated_weight > dpa_eps

    if mode == "detail_preserving_average":
        fused = detail_preserving_average(
            values,
            masks,
            weights,
            alpha=dpa_alpha,
            power=dpa_power,
            eps=dpa_eps,
            dim=0,
        )
    elif mode in {"average", "weighted_average"}:
        fused = (values * effective_weight).sum(dim=0) / accumulated_weight.clamp_min(dpa_eps)
    else:
        raise ValueError(f"Unsupported aggregation_mode={mode!r}")

    fused = torch.where(valid_output_mask, fused, torch.zeros_like(fused))
    return OverlapAggregationResult(
        fused_values=fused,
        accumulated_weight=accumulated_weight,
        contributor_count=contributor_count,
        valid_output_mask=valid_output_mask.to(values.dtype),
    )


def _empty_accumulator(
    channels: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    mode: str,
) -> Dict[str, torch.Tensor]:
    accumulator = {
        "ordinary_num": torch.zeros(channels, height, width, device=device, dtype=dtype),
        "ordinary_den": torch.zeros(1, height, width, device=device, dtype=dtype),
        "count": torch.zeros(1, height, width, device=device, dtype=dtype),
    }
    if mode == "detail_preserving_average":
        accumulator["detail_num"] = torch.zeros(channels, height, width, device=device, dtype=dtype)
        accumulator["detail_den"] = torch.zeros(channels, height, width, device=device, dtype=dtype)
    return accumulator


def _accumulate_projected(
    accumulator: Dict[str, torch.Tensor],
    values: torch.Tensor,
    masks: torch.Tensor,
    weights: torch.Tensor,
    config: PixelFusionConfig,
) -> None:
    if config.aggregation_mode == "average":
        weights = torch.ones_like(weights)
    effective_weight = masks * weights
    accumulator["ordinary_num"] += (values * effective_weight).sum(dim=0)
    accumulator["ordinary_den"] += effective_weight.sum(dim=0)
    accumulator["count"] += (masks > 0).sum(dim=0).to(values.dtype)
    if config.aggregation_mode == "detail_preserving_average":
        detail_weight = effective_weight * (values.abs() + config.dpa_eps).pow(config.dpa_power)
        accumulator["detail_num"] += (values * detail_weight).sum(dim=0)
        accumulator["detail_den"] += detail_weight.sum(dim=0)


def _finalize_accumulator(accumulator: Dict[str, torch.Tensor], config: PixelFusionConfig) -> OverlapAggregationResult:
    ordinary_den = accumulator["ordinary_den"].clamp_min(config.dpa_eps)
    ordinary = accumulator["ordinary_num"] / ordinary_den
    if config.aggregation_mode == "detail_preserving_average":
        detail = accumulator["detail_num"] / accumulator["detail_den"].clamp_min(config.dpa_eps)
        fused = ordinary + config.dpa_alpha * (detail - ordinary)
    else:
        fused = ordinary
    valid = accumulator["ordinary_den"] > config.dpa_eps
    fused = torch.where(valid, fused, torch.zeros_like(fused))
    return OverlapAggregationResult(
        fused_values=fused,
        accumulated_weight=accumulator["ordinary_den"],
        contributor_count=accumulator["count"],
        valid_output_mask=valid.to(fused.dtype),
    )


def project_views_to_erp_standard(
    view_images: torch.Tensor,
    view_dirs: torch.Tensor,
    fovs: Union[Tuple[float, float], Sequence[Tuple[float, float]]],
    erp_height: int,
    erp_width: int,
    config: PixelFusionConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse-warp RGB perspective patches [views, 3, H, W] to ERP [views, 3, H_erp, W_erp]."""

    view_images = view_images.to(dtype=FUSION_DTYPE)
    view_dirs = view_dirs.to(device=view_images.device, dtype=FUSION_DTYPE)
    num_views, _, patch_height, patch_width = view_images.shape
    fovs_list = _normalize_fovs(fovs, num_views)
    grid, valid_mask = _get_perspective_to_erp_grid(
        view_dirs,
        fovs_list,
        (patch_height, patch_width),
        (erp_height, erp_width),
        config,
        dtype=view_images.dtype,
        device=view_images.device,
    )
    projected_rgb = _sample_perspective_image(view_images, grid, padding_mode="zeros")
    weight_map = _get_patch_weight_map(patch_height, patch_width, config, device=view_images.device, dtype=view_images.dtype)
    projected_weight = _sample_perspective_image(
        weight_map.expand(num_views, -1, -1, -1),
        grid,
        padding_mode="zeros",
    )
    projected_mask = valid_mask
    projected_weight = projected_weight * projected_mask
    return projected_rgb, projected_mask, projected_weight


def _fuse_views_to_erp_standard(
    view_images: torch.Tensor,
    view_dirs: torch.Tensor,
    fovs: Union[Tuple[float, float], Sequence[Tuple[float, float]]],
    erp_height: int,
    erp_width: int,
    config: PixelFusionConfig,
    projected_confidence: Optional[torch.Tensor] = None,
    timings: Optional[Dict[str, float]] = None,
) -> OverlapAggregationResult:
    view_images = view_images.to(dtype=FUSION_DTYPE)
    view_dirs = view_dirs.to(device=view_images.device, dtype=FUSION_DTYPE)
    if projected_confidence is not None:
        projected_confidence = projected_confidence.to(device=view_images.device, dtype=FUSION_DTYPE)
    channels = view_images.shape[1]
    accumulator = _empty_accumulator(channels, erp_height, erp_width, device=view_images.device, dtype=view_images.dtype, mode=config.aggregation_mode)
    fovs_list = _normalize_fovs(fovs, view_images.shape[0])
    chunk_size = max(1, config.projection_chunk_size)
    for start in range(0, view_images.shape[0], chunk_size):
        end = min(start + chunk_size, view_images.shape[0])
        project = lambda: project_views_to_erp_standard(
            view_images[start:end], view_dirs[start:end], fovs_list[start:end], erp_height, erp_width, config
        )
        if timings is None:
            projected_rgb, projected_mask, projected_weight = project()
        else:
            projected_rgb, projected_mask, projected_weight = _timed(
                timings, "projection_or_inverse_lpw", project, synchronize=config.measure_performance
            )
        if projected_confidence is not None:
            projected_weight = projected_weight * projected_confidence[start:end]
        accumulate = lambda: _accumulate_projected(accumulator, projected_rgb, projected_mask, projected_weight, config)
        if timings is None:
            accumulate()
        else:
            _timed(timings, "overlap_fusion", accumulate, synchronize=config.measure_performance)
    if timings is None:
        return _finalize_accumulator(accumulator, config)
    return _timed(
        timings,
        "overlap_fusion",
        lambda: _finalize_accumulator(accumulator, config),
        synchronize=config.measure_performance,
    )


def render_views_to_erp_standard_weighted(
    decoded_views: Iterable[Tuple[torch.Tensor, torch.Tensor, Tuple[float, float]]],
    erp_height: int,
    erp_width: int,
    config: PixelFusionConfig,
) -> Optional[OverlapAggregationResult]:
    """Render possibly different-sized decoded views with the standard weighted ERP projector."""

    baseline_config = replace(config, warp_mode="standard", aggregation_mode="weighted_average")
    accumulator = None
    for view_image, view_dir, fov in decoded_views:
        view_image = view_image.to(dtype=FUSION_DTYPE)
        view_dir = view_dir.to(device=view_image.device, dtype=FUSION_DTYPE)
        if accumulator is None:
            accumulator = _empty_accumulator(
                view_image.shape[1],
                erp_height,
                erp_width,
                device=view_image.device,
                dtype=FUSION_DTYPE,
                mode="weighted_average",
            )
        projected_rgb, projected_mask, projected_weight = project_views_to_erp_standard(
            view_image,
            view_dir,
            fov,
            erp_height,
            erp_width,
            baseline_config,
        )
        _accumulate_projected(
            accumulator,
            projected_rgb,
            projected_mask,
            projected_weight,
            baseline_config,
        )

    if accumulator is None:
        return None
    return _finalize_accumulator(accumulator, baseline_config)
