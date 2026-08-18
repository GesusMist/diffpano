"""Laplacian Pyramid Warping (LPW) without changes to its equations."""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from diffpano.config import FUSION_DTYPE, PixelFusionConfig
from diffpano.diagnostics import _timed
from diffpano.fusion import OverlapAggregationResult, _fuse_views_to_erp_standard
from diffpano.projection import (
    _get_projection_lod_map,
    _level_size,
    _normalize_fovs,
    extract_views_from_erp_standard,
    spherical_pad_erp,
)


def _gaussian_kernel(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    kernel_1d = torch.tensor([1, 4, 6, 4, 1], device=device, dtype=dtype)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def circular_pad_horizontal(tensor: torch.Tensor, pad: int, vertical_padding_mode: str = "reflect") -> torch.Tensor:
    if vertical_padding_mode not in {"reflect", "replicate"}:
        raise ValueError(f"Unsupported erp_vertical_padding_mode={vertical_padding_mode!r}")
    return spherical_pad_erp(tensor, pad_y=pad, pad_x=pad)


def _pyramid_blur(tensor: torch.Tensor, vertical_padding_mode: str, circular_horizontal: bool) -> torch.Tensor:
    channels = tensor.shape[1]
    kernel = _gaussian_kernel(tensor.dtype, tensor.device).expand(channels, 1, 5, 5)
    if circular_horizontal:
        padded = spherical_pad_erp(tensor, pad_y=2, pad_x=2)
    else:
        padding_mode = vertical_padding_mode if min(tensor.shape[-2:]) > 2 else "replicate"
        padded = F.pad(tensor, (2, 2, 2, 2), mode=padding_mode)
    return F.conv2d(padded, kernel, groups=channels)


def build_gaussian_pyramid(
    tensor: torch.Tensor,
    num_levels: int,
    vertical_padding_mode: str = "reflect",
    *,
    circular_horizontal: bool = False,
) -> List[torch.Tensor]:
    pyramid = [tensor]
    current = tensor
    for _ in range(1, num_levels):
        blurred = _pyramid_blur(current, vertical_padding_mode, circular_horizontal)
        if blurred.shape[-2] < 2 or blurred.shape[-1] < 2:
            break
        current = F.interpolate(
            blurred,
            scale_factor=0.5,
            mode="bilinear",
            align_corners=not circular_horizontal,
            recompute_scale_factor=False,
        )
        pyramid.append(current)
    return pyramid


def build_laplacian_pyramid(
    tensor: torch.Tensor,
    num_levels: int,
    vertical_padding_mode: str = "reflect",
    *,
    circular_horizontal: bool = False,
) -> List[torch.Tensor]:
    gaussian = build_gaussian_pyramid(
        tensor,
        num_levels,
        vertical_padding_mode,
        circular_horizontal=circular_horizontal,
    )
    laplacian = []
    for idx in range(len(gaussian) - 1):
        upsampled = F.interpolate(
            gaussian[idx + 1],
            size=gaussian[idx].shape[-2:],
            mode="bilinear",
            align_corners=not circular_horizontal,
        )
        laplacian.append(gaussian[idx] - upsampled)
    laplacian.append(gaussian[-1])
    return laplacian


def reconstruct_laplacian_pyramid(pyramid: Sequence[torch.Tensor]) -> torch.Tensor:
    current = pyramid[-1]
    for level in reversed(pyramid[:-1]):
        current = F.interpolate(current, size=level.shape[-2:], mode="bilinear", align_corners=True) + level
    return current


def _reconstruct_masked_laplacian_pyramid(
    pyramid: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    current = pyramid[-1]
    current_mask = masks[-1].to(current.dtype)
    for level, level_mask in zip(reversed(pyramid[:-1]), reversed(masks[:-1])):
        upsampled_mask = F.interpolate(current_mask, size=level.shape[-2:], mode="bilinear", align_corners=False)
        upsampled = F.interpolate(current * current_mask, size=level.shape[-2:], mode="bilinear", align_corners=False)
        upsampled = upsampled / upsampled_mask.clamp_min(eps)
        current = level + upsampled
        current_mask = torch.maximum(level_mask.to(current.dtype), upsampled_mask)
    return current, current_mask


def _build_masked_laplacian_pyramid(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    num_levels: int,
    vertical_padding_mode: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Build an ERP pyramid without allowing invalid pixels to darken valid coefficients."""

    gaussian = [tensor]
    masks = [mask.to(tensor.dtype)]
    current, current_mask = tensor, masks[0]
    for _ in range(1, num_levels):
        blurred_mask = _pyramid_blur(current_mask, vertical_padding_mode, circular_horizontal=True)
        blurred_values = _pyramid_blur(current * current_mask, vertical_padding_mode, circular_horizontal=True)
        normalized = blurred_values / blurred_mask.clamp_min(torch.finfo(tensor.dtype).eps)
        if normalized.shape[-2] < 2 or normalized.shape[-1] < 2:
            break
        current = F.interpolate(
            normalized,
            scale_factor=0.5,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        current_mask = F.interpolate(
            blurred_mask,
            size=current.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)
        gaussian.append(current)
        masks.append(current_mask)

    laplacian = []
    for idx in range(len(gaussian) - 1):
        upsampled = F.interpolate(
            gaussian[idx + 1], size=gaussian[idx].shape[-2:], mode="bilinear", align_corners=False
        )
        laplacian.append(gaussian[idx] - upsampled)
    laplacian.append(gaussian[-1])
    return laplacian, masks

def _lod_level_confidence(lod: torch.Tensor, level: int, num_levels: int, interpolation: str) -> torch.Tensor:
    if level == num_levels - 1:
        # The coarsest Gaussian residual carries the pyramid's base signal and must always contribute.
        return torch.ones_like(lod)
    capped = lod.clamp(0, max(num_levels - 1, 0))
    if interpolation == "nearest":
        return (capped.round() == level).to(lod.dtype)
    return (1 - (capped - level).abs()).clamp(0, 1)


def inverse_lpw_to_erp(
    view_images: torch.Tensor,
    view_dirs: torch.Tensor,
    fovs: Union[Tuple[float, float], Sequence[Tuple[float, float]]],
    erp_height: int,
    erp_width: int,
    config: PixelFusionConfig,
    timings: Optional[Dict[str, float]] = None,
) -> OverlapAggregationResult:
    """Project each patch Laplacian level to a matching ERP pyramid level and fuse coefficients jointly."""

    view_images = view_images.to(dtype=FUSION_DTYPE)
    view_dirs = view_dirs.to(device=view_images.device, dtype=FUSION_DTYPE)
    patch_pyramid = build_laplacian_pyramid(view_images, config.lpw_num_levels, config.erp_vertical_padding_mode)
    fovs_list = _normalize_fovs(fovs, view_images.shape[0])
    lod_map = _get_projection_lod_map(
        view_dirs,
        fovs_list,
        view_images.shape[-2:],
        (erp_height, erp_width),
        config,
        dtype=view_images.dtype,
        device=view_images.device,
    )
    erp_levels: List[OverlapAggregationResult] = []
    for level, coeffs in enumerate(patch_pyramid):
        level_erp_height = _level_size(erp_height, level)
        level_erp_width = _level_size(erp_width, level)
        level_lod = F.interpolate(
            lod_map,
            size=(level_erp_height, level_erp_width),
            mode="bilinear",
            align_corners=False,
        )
        level_confidence = _lod_level_confidence(
            level_lod,
            level,
            len(patch_pyramid),
            config.lpw_lod_interpolation,
        )
        result = _fuse_views_to_erp_standard(
            coeffs,
            view_dirs,
            fovs_list,
            level_erp_height,
            level_erp_width,
            config,
            projected_confidence=level_confidence,
            timings=timings,
        )
        erp_levels.append(result)
    reconstruct = lambda: _reconstruct_masked_laplacian_pyramid(
        [level.fused_values.unsqueeze(0) for level in erp_levels],
        [level.valid_output_mask.unsqueeze(0) for level in erp_levels],
        config.dpa_eps,
    )
    fused_erp_batch, valid_mask_batch = reconstruct() if timings is None else _timed(
        timings,
        "erp_reconstruction",
        reconstruct,
        synchronize=config.measure_performance,
    )
    fused_erp = fused_erp_batch[0]
    valid_mask = valid_mask_batch[0]
    return OverlapAggregationResult(
        fused_values=fused_erp,
        accumulated_weight=erp_levels[0].accumulated_weight,
        contributor_count=erp_levels[0].contributor_count,
        valid_output_mask=valid_mask,
    )


def forward_lpw_to_views(
    erp_image: torch.Tensor,
    erp_valid_mask: torch.Tensor,
    original_view_images: torch.Tensor,
    view_dirs: torch.Tensor,
    fovs: Union[Tuple[float, float], Sequence[Tuple[float, float]]],
    config: PixelFusionConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample an ERP Laplacian pyramid back to perspective-patch pyramids, then reconstruct each view."""

    erp_image = erp_image.to(dtype=FUSION_DTYPE)
    erp_valid_mask = erp_valid_mask.to(device=erp_image.device, dtype=FUSION_DTYPE)
    original_view_images = original_view_images.to(device=erp_image.device, dtype=FUSION_DTYPE)
    view_dirs = view_dirs.to(device=erp_image.device, dtype=FUSION_DTYPE)
    erp_pyramid, erp_mask_pyramid = _build_masked_laplacian_pyramid(
        erp_image.unsqueeze(0),
        erp_valid_mask.unsqueeze(0),
        config.lpw_num_levels,
        config.erp_vertical_padding_mode,
    )
    original_pyramid = build_laplacian_pyramid(
        original_view_images,
        len(erp_pyramid),
        config.erp_vertical_padding_mode,
    )
    view_levels = []
    valid_levels = []
    for erp_level, erp_level_mask, original_coefficients in zip(erp_pyramid, erp_mask_pyramid, original_pyramid):
        views, valid = extract_views_from_erp_standard(
            erp_level[0],
            erp_level_mask[0],
            original_coefficients,
            view_dirs,
            fovs,
            config,
        )
        view_levels.append(views)
        valid_levels.append(valid)
    reconstructed = reconstruct_laplacian_pyramid(view_levels)
    full_valid = F.interpolate(valid_levels[0], size=original_view_images.shape[-2:], mode="nearest")
    return reconstructed, full_valid
