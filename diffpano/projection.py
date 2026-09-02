"""Exact FP32 inverse resampling between ERP and perspective RGB images."""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from diffpano.camera import PerspectiveCamera
from diffpano.geometry import erp_world_directions, world_to_longitude_latitude


@dataclass
class ERPContribution:
    rgb: torch.Tensor
    valid_mask: torch.Tensor
    weight: torch.Tensor
    lod_map: Optional[torch.Tensor] = None


@dataclass
class ProjectionCache:
    """Device-aware geometry cache; fixed covers reuse all entries across steps."""

    erp_to_view: Dict[Tuple[object, ...], torch.Tensor] = field(default_factory=dict)
    view_to_erp: Dict[Tuple[object, ...], Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    erp_rays: Dict[Tuple[object, ...], torch.Tensor] = field(default_factory=dict)
    lod_maps: Dict[Tuple[object, ...], torch.Tensor] = field(default_factory=dict)


def spherical_pad_erp(
    erp: torch.Tensor,
    pad_y: int = 1,
    pad_x: int = 1,
    vertical_mode: str = "reflect",
) -> torch.Tensor:
    """Pad periodic longitude and handle pole crossings without an ERP seam."""

    if erp.ndim != 4:
        raise ValueError(f"Expected ERP [B,C,H,W], got {tuple(erp.shape)}")
    if pad_x < 0 or pad_y < 0:
        raise ValueError("padding must be nonnegative")
    height, width = erp.shape[-2:]
    if pad_x > width or pad_y > height:
        raise ValueError("padding exceeds the ERP size")
    if pad_y and width % 2:
        raise ValueError("pole padding requires an even ERP width")
    if vertical_mode not in {"reflect", "replicate"}:
        raise ValueError("vertical_mode must be reflect or replicate")
    result = erp
    if pad_y:
        if vertical_mode == "reflect":
            north = torch.roll(erp[..., :pad_y, :].flip(-2), width // 2, dims=-1)
            south = torch.roll(erp[..., -pad_y:, :].flip(-2), width // 2, dims=-1)
        else:
            north = erp[..., :1, :].expand(*erp.shape[:-2], pad_y, width)
            south = erp[..., -1:, :].expand(*erp.shape[:-2], pad_y, width)
        result = torch.cat([north, erp, south], dim=-2)
    if pad_x:
        result = torch.cat([result[..., -pad_x:], result, result[..., :pad_x]], dim=-1)
    return result


def _erp_grid_to_padded(grid: torch.Tensor, height: int, width: int) -> torch.Tensor:
    u = torch.remainder((grid[..., 0] + 1) * 0.5, 1.0)
    v = ((grid[..., 1] + 1) * 0.5).clamp(0, 1)
    grid_x = 2 * (u * width + 1) / (width + 2) - 1
    grid_y = 2 * (v * height + 1) / (height + 2) - 1
    return torch.stack([grid_x, grid_y], dim=-1)


def perspective_world_rays(camera: PerspectiveCamera, *, device: torch.device) -> torch.Tensor:
    """Construct one world ray per perspective output pixel using pixel centres."""

    x = torch.arange(camera.width, device=device, dtype=torch.float32)
    y = torch.arange(camera.height, device=device, dtype=torch.float32)
    x_cam = (2 * (x + 0.5) / camera.width - 1) * math.tan(math.radians(camera.fov_x) / 2)
    y_cam = (1 - 2 * (y + 0.5) / camera.height) * math.tan(math.radians(camera.fov_y) / 2)
    yy, xx = torch.meshgrid(y_cam, x_cam, indexing="ij")
    rays = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    rays = rays / torch.linalg.vector_norm(rays, dim=-1, keepdim=True).clamp_min(1.0e-12)
    return torch.einsum("ij,hwj->hwi", camera.rotation(device), rays)


def erp_to_perspective_grid(
    camera: PerspectiveCamera,
    erp_height: int,
    erp_width: int,
    *,
    device: torch.device,
    cache: Optional[ProjectionCache] = None,
) -> torch.Tensor:
    key = ("erp_to_view", str(device), erp_height, erp_width, *camera.cache_key())
    if cache is not None and key in cache.erp_to_view:
        return cache.erp_to_view[key]
    world = perspective_world_rays(camera, device=device)
    longitude, latitude = world_to_longitude_latitude(world)
    grid = torch.stack([longitude / math.pi, -2 * latitude / math.pi], dim=-1).unsqueeze(0)
    if cache is not None:
        cache.erp_to_view[key] = grid
    return grid


def erp_to_perspective(
    erp_rgb: torch.Tensor,
    camera: PerspectiveCamera,
    *,
    interpolation: str,
    cache: Optional[ProjectionCache] = None,
    vertical_padding_mode: str = "reflect",
) -> torch.Tensor:
    """Ray-cast a perspective RGB view from a horizontally periodic ERP."""

    if erp_rgb.ndim != 4 or erp_rgb.shape[1] != 3:
        raise ValueError("erp_rgb must have shape [B,3,H,W]")
    if interpolation not in {"nearest", "bilinear"}:
        raise ValueError("interpolation must be nearest or bilinear")
    source = erp_rgb.to(dtype=torch.float32)
    height, width = source.shape[-2:]
    grid = erp_to_perspective_grid(camera, height, width, device=source.device, cache=cache)
    padded = spherical_pad_erp(source, 1, 1, vertical_padding_mode)
    padded_grid = _erp_grid_to_padded(grid, height, width).expand(source.shape[0], -1, -1, -1)
    return F.grid_sample(
        padded, padded_grid, mode=interpolation, padding_mode="border", align_corners=False
    )


def _cached_erp_rays(
    height: int, width: int, *, device: torch.device, cache: Optional[ProjectionCache]
) -> torch.Tensor:
    key = (str(device), height, width)
    if cache is not None and key in cache.erp_rays:
        return cache.erp_rays[key]
    rays = erp_world_directions(height, width, device=device)
    if cache is not None:
        cache.erp_rays[key] = rays
    return rays


def perspective_to_erp_grid(
    camera: PerspectiveCamera,
    erp_height: int,
    erp_width: int,
    *,
    device: torch.device,
    cache: Optional[ProjectionCache] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map every ERP target ray into a perspective source image."""

    key = ("view_to_erp", str(device), erp_height, erp_width, *camera.cache_key())
    if cache is not None and key in cache.view_to_erp:
        return cache.view_to_erp[key]
    world = _cached_erp_rays(erp_height, erp_width, device=device, cache=cache)
    camera_dirs = torch.einsum("ji,hwj->hwi", camera.rotation(device), world)
    x, y, z = camera_dirs.unbind(dim=-1)
    tan_x = math.tan(math.radians(camera.fov_x) / 2)
    tan_y = math.tan(math.radians(camera.fov_y) / 2)
    safe_z = torch.where(z.abs() > 1.0e-12, z, torch.ones_like(z))
    x_norm = (x / safe_z) / tan_x
    y_norm = (y / safe_z) / tan_y
    valid = (z > 0) & (x_norm.abs() <= 1) & (y_norm.abs() <= 1)
    grid = torch.stack([x_norm, -y_norm], dim=-1).unsqueeze(0)
    mask = valid.to(torch.float32).unsqueeze(0).unsqueeze(0)
    if cache is not None:
        cache.view_to_erp[key] = (grid, mask)
    return grid, mask


def perspective_to_erp(
    rgb_view: torch.Tensor,
    camera: PerspectiveCamera,
    erp_height: int,
    erp_width: int,
    *,
    interpolation: str,
    weight_map: torch.Tensor,
    cache: Optional[ProjectionCache] = None,
) -> ERPContribution:
    """Inverse-resample one perspective proposal into its irregular ERP footprint."""

    if rgb_view.ndim != 4 or rgb_view.shape[1] != 3:
        raise ValueError("rgb_view must have shape [B,3,H,W]")
    if tuple(rgb_view.shape[-2:]) != (camera.height, camera.width):
        raise ValueError("rgb_view shape does not match its camera")
    if interpolation not in {"nearest", "bilinear"}:
        raise ValueError("interpolation must be nearest or bilinear")
    view = rgb_view.to(dtype=torch.float32)
    grid, mask = perspective_to_erp_grid(
        camera, erp_height, erp_width, device=view.device, cache=cache
    )
    batch_grid = grid.expand(view.shape[0], -1, -1, -1)
    rgb = F.grid_sample(view, batch_grid, mode=interpolation, padding_mode="border", align_corners=False)
    weights = weight_map.to(device=view.device, dtype=torch.float32)
    if weights.shape[0] == 1 and view.shape[0] != 1:
        weights = weights.expand(view.shape[0], -1, -1, -1)
    weight = F.grid_sample(
        weights, batch_grid, mode=interpolation, padding_mode="border", align_corners=False
    )
    mask = mask.expand(view.shape[0], -1, -1, -1)
    return ERPContribution(rgb=rgb * mask, valid_mask=mask, weight=weight * mask)


def projection_lod_map(
    camera: PerspectiveCamera,
    erp_height: int,
    erp_width: int,
    *,
    device: torch.device,
    cache: Optional[ProjectionCache] = None,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Estimate perspective-source pixels per ERP pixel from the inverse-warp Jacobian."""

    key = ("lod", str(device), erp_height, erp_width, *camera.cache_key())
    if cache is not None and key in cache.lod_maps:
        return cache.lod_maps[key]
    grid, valid = perspective_to_erp_grid(camera, erp_height, erp_width, device=device, cache=cache)
    pixel_scale = grid.new_tensor([camera.width / 2, camera.height / 2])
    dx = ((grid[:, :, 1:] - grid[:, :, :-1]) * pixel_scale).norm(dim=-1)
    dy = ((grid[:, 1:] - grid[:, :-1]) * pixel_scale).norm(dim=-1)
    dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")
    lod = torch.log2(torch.maximum(dx, dy).clamp_min(epsilon)).clamp_min(0).unsqueeze(1)
    lod = lod * valid
    if cache is not None:
        cache.lod_maps[key] = lod
    return lod
