"""Laplacian-pyramid utilities with ERP-aware filtering."""

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

from diffpano.projection import spherical_pad_erp


def _gaussian_kernel(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    one_d = torch.tensor([1, 4, 6, 4, 1], dtype=dtype, device=device)
    kernel = one_d[:, None] * one_d[None, :]
    return kernel / kernel.sum()


def pyramid_blur(
    tensor: torch.Tensor,
    *,
    vertical_padding_mode: str,
    spherical_erp: bool,
) -> torch.Tensor:
    channels = tensor.shape[1]
    kernel = _gaussian_kernel(tensor.dtype, tensor.device).expand(channels, 1, 5, 5)
    if spherical_erp:
        padded = spherical_pad_erp(tensor, 2, 2, vertical_padding_mode)
    else:
        mode = vertical_padding_mode if min(tensor.shape[-2:]) > 2 else "replicate"
        padded = F.pad(tensor, (2, 2, 2, 2), mode=mode)
    return F.conv2d(padded, kernel, groups=channels)


def build_gaussian_pyramid(
    tensor: torch.Tensor,
    levels: int,
    *,
    vertical_padding_mode: str = "reflect",
    spherical_erp: bool = False,
) -> List[torch.Tensor]:
    pyramid = [tensor.to(dtype=torch.float32)]
    for _ in range(1, levels):
        blurred = pyramid_blur(
            pyramid[-1], vertical_padding_mode=vertical_padding_mode, spherical_erp=spherical_erp
        )
        if min(blurred.shape[-2:]) < 2:
            break
        pyramid.append(F.interpolate(blurred, scale_factor=0.5, mode="bilinear", align_corners=False))
    return pyramid


def pyramid_upsample(tensor, size, *, spherical_erp=False):
    """Bilinear expansion with optional ERP wrap/pole boundaries (O only)."""
    if not spherical_erp:
        return F.interpolate(tensor,size=size,mode="bilinear",align_corners=False)
    h,w = tensor.shape[-2:];th,tw = size
    padded = spherical_pad_erp(tensor,1,1,"reflect")
    y = 2*((torch.arange(th,device=tensor.device,dtype=tensor.dtype)+.5)*h/th+1)/(h+2)-1
    x = 2*((torch.arange(tw,device=tensor.device,dtype=tensor.dtype)+.5)*w/tw+1)/(w+2)-1
    yy,xx = torch.meshgrid(y,x,indexing="ij")
    grid = torch.stack((xx,yy),-1)[None].expand(tensor.shape[0],-1,-1,-1)
    return F.grid_sample(padded,grid,mode="bilinear",padding_mode="border",align_corners=False)


def build_laplacian_pyramid(
    tensor: torch.Tensor,
    levels: int,
    *,
    vertical_padding_mode: str = "reflect",
    spherical_erp: bool = False,
    periodic_upsampling: bool = False,
) -> List[torch.Tensor]:
    gaussian = build_gaussian_pyramid(
        tensor,
        levels,
        vertical_padding_mode=vertical_padding_mode,
        spherical_erp=spherical_erp,
    )
    result = []
    for fine, coarse in zip(gaussian[:-1], gaussian[1:]):
        result.append(fine - pyramid_upsample(coarse,fine.shape[-2:],spherical_erp=periodic_upsampling))
    result.append(gaussian[-1])
    return result


def reconstruct_laplacian_pyramid(pyramid: Sequence[torch.Tensor], *, spherical_erp=False) -> torch.Tensor:
    if not pyramid:
        raise ValueError("pyramid must contain at least one level")
    current = pyramid[-1]
    for level in reversed(pyramid[:-1]):
        current = pyramid_upsample(current,level.shape[-2:],spherical_erp=spherical_erp) + level
    return current


def reconstruct_masked_laplacian_pyramid(
    pyramid: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    epsilon: float,
    *, spherical_erp=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct coefficients without blending invalid zeros across boundaries.

    Each level carries its own validity mask.  Coarser values are upsampled as
    premultiplied values and normalized by the interpolated mask before the
    finer coefficients are added.  The returned mask is the union propagated
    through the pyramid and may be fractional at footprint boundaries.
    """

    if not pyramid:
        raise ValueError("pyramid must contain at least one level")
    if len(pyramid) != len(masks):
        raise ValueError("pyramid and masks must contain the same number of levels")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    current = pyramid[-1]
    current_mask = masks[-1].to(device=current.device, dtype=current.dtype)
    for level, level_mask in zip(reversed(pyramid[:-1]), reversed(masks[:-1])):
        level_mask = level_mask.to(device=level.device, dtype=level.dtype)
        upsampled_mask = pyramid_upsample(current_mask,level.shape[-2:],spherical_erp=spherical_erp)
        upsampled_values = pyramid_upsample(current * current_mask,level.shape[-2:],spherical_erp=spherical_erp)
        normalized = upsampled_values / upsampled_mask.clamp_min(epsilon)
        current = level + normalized
        current_mask = torch.maximum(level_mask, upsampled_mask)
    return current, current_mask


def lod_level_confidence(
    lod: torch.Tensor, level: int, num_levels: int, interpolation: str
) -> torch.Tensor:
    """Map a fractional Jacobian LOD to one pyramid level."""

    if interpolation not in {"nearest", "linear"}:
        raise ValueError("LOD interpolation must be nearest or linear")
    if level == num_levels - 1:
        return torch.ones_like(lod)
    capped = lod.clamp(0, max(num_levels - 1, 0))
    if interpolation == "nearest":
        return (capped.round() == level).to(lod.dtype)
    return (1 - (capped - level).abs()).clamp(0, 1)
