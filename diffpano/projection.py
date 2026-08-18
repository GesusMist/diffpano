"""FP32 perspective/ERP projection primitives and spherical padding."""

from typing import Any, List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from diffpano.config import FUSION_DTYPE, PixelFusionConfig
from diffpano.geometry import SphericalFunctions


def _round_tuple(tensor: torch.Tensor) -> Tuple[float, ...]:
    return tuple(round(float(value), 6) for value in tensor.detach().cpu().flatten())


def _normalize_fovs(fovs: Union[Tuple[float, float], Sequence[Tuple[float, float]]], num_views: int) -> List[Tuple[float, float]]:
    if isinstance(fovs, tuple) and len(fovs) == 2 and not isinstance(fovs[0], (tuple, list)):
        return [tuple(float(item) for item in fovs)] * num_views
    if len(fovs) != num_views:
        raise ValueError(f"Expected {num_views} FOV entries, got {len(fovs)}")
    return [tuple(float(item) for item in fov) for fov in fovs]


def _erp_world_grid(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return FP32 rays for pixel-centered ERP coordinates, north at top and south at bottom."""

    dtype = FUSION_DTYPE
    u_range = torch.linspace(0, 1, width * 2 + 1, device=device, dtype=dtype)[1::2]
    v_range = torch.linspace(0, 1, height * 2 + 1, device=device, dtype=dtype)[1::2]
    u, v = torch.meshgrid(u_range, v_range, indexing="xy")
    dx, dy, dz, _ = SphericalFunctions.latlong2world_ours(u, v)
    return torch.stack([dx, dy, dz], dim=-1).reshape(height * width, 3)


def _world_to_perspective_grid(
    world_xyz: torch.Tensor,
    view_dirs: torch.Tensor,
    fovs: Sequence[Tuple[float, float]],
    erp_height: int,
    erp_width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map ERP world rays to perspective grid coordinates using SphereDiff's camera convention."""

    world_xyz = world_xyz.to(dtype=FUSION_DTYPE)
    view_dirs = view_dirs.to(device=world_xyz.device, dtype=FUSION_DTYPE)
    device, dtype = world_xyz.device, world_xyz.dtype
    num_views = view_dirs.shape[0]
    xyz = world_xyz.t().unsqueeze(0).expand(num_views, -1, -1)

    theta_camera, phi_camera = SphericalFunctions.cartesian_to_spherical(view_dirs)
    theta_camera = torch.where(theta_camera > torch.pi, theta_camera - 2 * torch.pi, theta_camera)
    phi_camera = torch.where(phi_camera > torch.pi / 2, phi_camera - torch.pi, phi_camera)
    rotation_matrix = SphericalFunctions.rotation_matrix(theta_camera, phi_camera)

    fov_tensor = torch.tensor(fovs, device=device, dtype=dtype)
    fov_rad = torch.deg2rad(fov_tensor)
    fx = 0.5 / torch.tan(fov_rad[:, 1] / 2)
    fy = 0.5 / torch.tan(fov_rad[:, 0] / 2)
    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fx)
    k_rows = [
        torch.stack([fx, zeros, zeros], dim=-1),
        torch.stack([zeros, fy, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ]
    intrinsics = torch.stack(k_rows, dim=1)
    # SphereDiff's ray einsum treats rays as row vectors (world = camera @ R),
    # so column-vector world rays map back to camera coordinates with R @ world.
    projection = torch.einsum("bij,bjk->bik", intrinsics, rotation_matrix)

    projected = torch.einsum("bij,bjn->bin", projection, xyz)
    eps = torch.finfo(dtype).eps if dtype.is_floating_point else 1e-6
    perspective_u = projected[:, 0] / (projected[:, 2] + eps)
    perspective_v = projected[:, 1] / (projected[:, 2] + eps)
    grid_x = 2 * perspective_u
    grid_y = 2 * perspective_v
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(num_views, erp_height, erp_width, 2)

    forward_vector = torch.tensor([0, 0, -1], device=device, dtype=dtype).expand(num_views, -1)
    forward_vector = torch.einsum("bij,bj->bi", rotation_matrix.permute(0, 2, 1), forward_vector)
    hemisphere_mask = torch.einsum("bjn,bj->bn", xyz, forward_vector) > 0
    in_bounds = (grid_x >= -1) & (grid_x <= 1) & (grid_y >= -1) & (grid_y <= 1)
    valid = (hemisphere_mask & in_bounds).reshape(num_views, 1, erp_height, erp_width)
    return grid, valid.to(dtype)


def _perspective_to_erp_cache_key(
    view_dirs: torch.Tensor,
    fovs: Sequence[Tuple[float, float]],
    patch_size: Tuple[int, int],
    erp_size: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    prefix: str,
) -> Tuple[Any, ...]:
    return (
        prefix,
        str(device),
        str(dtype),
        patch_size,
        erp_size,
        tuple((round(fov[0], 6), round(fov[1], 6)) for fov in fovs),
        _round_tuple(view_dirs),
    )


def _get_perspective_to_erp_grid(
    view_dirs: torch.Tensor,
    fovs: Sequence[Tuple[float, float]],
    patch_size: Tuple[int, int],
    erp_size: Tuple[int, int],
    config: PixelFusionConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
    prefix: str = "perspective_to_erp",
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = FUSION_DTYPE
    key = _perspective_to_erp_cache_key(view_dirs, fovs, patch_size, erp_size, dtype, device, prefix)
    if key not in config.projection_cache.grids:
        world = _erp_world_grid(erp_size[0], erp_size[1], device=device, dtype=dtype)
        config.projection_cache.grids[key] = _world_to_perspective_grid(world, view_dirs.to(device=device, dtype=dtype), fovs, erp_size[0], erp_size[1])
    return config.projection_cache.grids[key]


def _world_to_erp_grid(world_dirs: torch.Tensor, *, erp_height: int, erp_width: int) -> torch.Tensor:
    """Map world rays to pixel-centered ERP grid coordinates for align_corners=False sampling."""

    x, y, z = world_dirs[..., 0], world_dirs[..., 1], world_dirs[..., 2]
    theta = torch.atan2(x, -z)
    v = torch.acos(torch.clamp(y, -1.0, 1.0)) / torch.pi
    grid_x = theta / torch.pi
    grid_y = 2 * v - 1
    return torch.stack([grid_x, grid_y], dim=-1)


def _perspective_pixel_world_dirs(
    view_dirs: torch.Tensor,
    fovs: Sequence[Tuple[float, float]],
    output_size: Tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    dtype = FUSION_DTYPE
    height, width = output_size
    num_views = view_dirs.shape[0]
    fov_tensor = torch.tensor(fovs, device=device, dtype=dtype)
    per_view_dirs = []
    for idx in range(num_views):
        fov_rad = torch.deg2rad(fov_tensor[idx])
        x_range = torch.linspace(torch.tan(fov_rad[1] / 2), -torch.tan(fov_rad[1] / 2), width, device=device, dtype=dtype)
        y_range = torch.linspace(torch.tan(fov_rad[0] / 2), -torch.tan(fov_rad[0] / 2), height, device=device, dtype=dtype)
        xv, yv = torch.meshgrid(x_range, y_range, indexing="xy")
        zv = torch.ones_like(xv)
        pixel_dirs = torch.stack([xv, yv, -zv], dim=-1)
        pixel_dirs = pixel_dirs / torch.linalg.norm(pixel_dirs, dim=-1, keepdim=True).clamp_min(1e-12)

        theta, phi = SphericalFunctions.cartesian_to_spherical(view_dirs[idx:idx + 1].to(device=device, dtype=dtype))
        rotation_matrix = SphericalFunctions.rotation_matrix(theta, phi)
        per_view_dirs.append(torch.einsum("bij,hwi->bhwj", rotation_matrix, pixel_dirs)[0])
    return torch.stack(per_view_dirs, dim=0)


def _get_erp_to_perspective_grid(
    view_dirs: torch.Tensor,
    fovs: Sequence[Tuple[float, float]],
    view_size: Tuple[int, int],
    erp_size: Tuple[int, int],
    config: PixelFusionConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
    prefix: str = "erp_to_perspective",
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = FUSION_DTYPE
    key = _perspective_to_erp_cache_key(view_dirs, fovs, view_size, erp_size, dtype, device, prefix)
    if key not in config.projection_cache.grids:
        world_dirs = _perspective_pixel_world_dirs(view_dirs, fovs, view_size, device=device, dtype=dtype)
        grid = _world_to_erp_grid(world_dirs, erp_height=erp_size[0], erp_width=erp_size[1])
        valid = torch.ones(view_dirs.shape[0], 1, view_size[0], view_size[1], device=device, dtype=dtype)
        config.projection_cache.grids[key] = (grid, valid)
    return config.projection_cache.grids[key]


def _sample_perspective_image(
    values: torch.Tensor,
    grid: torch.Tensor,
    *,
    padding_mode: str = "zeros",
    mode: str = "bilinear",
) -> torch.Tensor:
    """Sample endpoint-defined perspective pixels."""

    return F.grid_sample(values, grid, mode=mode, padding_mode=padding_mode, align_corners=True)


def _sample_erp_image(
    values: torch.Tensor,
    grid: torch.Tensor,
    *,
    padding_mode: str = "zeros",
    mode: str = "bilinear",
) -> torch.Tensor:
    """Sample pixel-centered ERP coordinates u=(x+0.5)/W, v=(y+0.5)/H."""

    return F.grid_sample(values, grid, mode=mode, padding_mode=padding_mode, align_corners=False)

def spherical_pad_erp(erp: torch.Tensor, pad_y: int, pad_x: int) -> torch.Tensor:
    """Pad an ERP with periodic longitude and pole-reflected, half-turned latitude rows."""

    if erp.ndim != 4:
        raise ValueError(f"Expected ERP [B,C,H,W], got {tuple(erp.shape)}")
    if pad_y < 0 or pad_x < 0:
        raise ValueError("ERP padding must be nonnegative")
    height, width = erp.shape[-2:]
    if pad_y > height or pad_x > width:
        raise ValueError(f"ERP padding {(pad_y, pad_x)} exceeds ERP size {(height, width)}")
    if pad_y and width % 2:
        raise ValueError(f"Exact pole padding requires an even ERP width, got {width}")

    padded = erp
    if pad_y:
        half_turn = width // 2
        north = torch.roll(erp[..., :pad_y, :].flip(-2), shifts=half_turn, dims=-1)
        south = torch.roll(erp[..., -pad_y:, :].flip(-2), shifts=half_turn, dims=-1)
        padded = torch.cat([north, erp, south], dim=-2)
    if pad_x:
        padded = torch.cat([padded[..., -pad_x:], padded, padded[..., :pad_x]], dim=-1)
    return padded


def _pad_erp_for_sampling(erp: torch.Tensor, vertical_padding_mode: str) -> torch.Tensor:
    if vertical_padding_mode not in {"reflect", "replicate"}:
        raise ValueError(f"Unsupported erp_vertical_padding_mode={vertical_padding_mode!r}")
    return spherical_pad_erp(erp, pad_y=1, pad_x=1)


def _erp_grid_to_padded_grid(
    grid: torch.Tensor,
    erp_height: int,
    erp_width: int,
    *,
    pad_y: int = 1,
    pad_x: int = 1,
) -> torch.Tensor:
    # Under align_corners=False, u maps to source coordinate u*W-0.5. Adding pad_x
    # moves that coordinate into the padded tensor, whose normalized center coordinate is
    # 2*(u*W+pad_x)/(W+2*pad_x)-1. The latitude expression is analogous.
    u = torch.remainder((grid[..., 0] + 1) * 0.5, 1.0)
    v = (grid[..., 1] + 1) * 0.5
    grid_x = 2 * (u * erp_width + pad_x) / (erp_width + 2 * pad_x) - 1
    grid_y = 2 * (v * erp_height + pad_y) / (erp_height + 2 * pad_y) - 1
    return torch.stack([grid_x, grid_y], dim=-1)


def extract_views_from_erp_standard(
    erp_image: torch.Tensor,
    erp_valid_mask: torch.Tensor,
    original_view_images: torch.Tensor,
    view_dirs: torch.Tensor,
    fovs: Union[Tuple[float, float], Sequence[Tuple[float, float]]],
    config: PixelFusionConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward-warp by sampling ERP into the original perspective patch layout, with invalid fallback."""

    erp_image = erp_image.to(dtype=FUSION_DTYPE)
    erp_valid_mask = erp_valid_mask.to(device=erp_image.device, dtype=FUSION_DTYPE)
    original_view_images = original_view_images.to(device=erp_image.device, dtype=FUSION_DTYPE)
    view_dirs = view_dirs.to(device=erp_image.device, dtype=FUSION_DTYPE)
    num_views, _, patch_height, patch_width = original_view_images.shape
    fovs_list = _normalize_fovs(fovs, num_views)
    grid, _ = _get_erp_to_perspective_grid(
        view_dirs,
        fovs_list,
        (patch_height, patch_width),
        (erp_image.shape[-2], erp_image.shape[-1]),
        config,
        dtype=original_view_images.dtype,
        device=original_view_images.device,
    )
    padded_grid = _erp_grid_to_padded_grid(grid, erp_image.shape[-2], erp_image.shape[-1])
    padded_erp = _pad_erp_for_sampling(
        (erp_image * erp_valid_mask).unsqueeze(0),
        config.erp_vertical_padding_mode,
    )
    padded_mask = _pad_erp_for_sampling(
        erp_valid_mask.unsqueeze(0),
        config.erp_vertical_padding_mode,
    )
    sampled_chunks = []
    sampled_mask_chunks = []
    chunk_size = max(1, config.projection_chunk_size)
    for start in range(0, num_views, chunk_size):
        end = min(start + chunk_size, num_views)
        chunk_views = end - start
        sampled_chunks.append(
            _sample_erp_image(
                padded_erp.expand(chunk_views, -1, -1, -1),
                padded_grid[start:end],
                padding_mode="border",
                mode=config.erp_to_perspective_interpolation_mode,
            )
        )
        sampled_mask_chunks.append(
            _sample_erp_image(
                padded_mask.expand(chunk_views, -1, -1, -1),
                padded_grid[start:end],
                padding_mode="border",
                mode=config.erp_to_perspective_interpolation_mode,
            )
        )
    sampled = torch.cat(sampled_chunks, dim=0)
    sampled_mask = torch.cat(sampled_mask_chunks, dim=0)
    # Sampling coverage alongside RGB supports normalized bilinear boundaries and exact nearest-mask selection.
    # Dividing by sampled coverage preserves valid RGB values before falling back to x0 RGB.
    sampled = sampled / sampled_mask.clamp_min(config.dpa_eps)
    valid = (sampled_mask > config.dpa_eps).to(original_view_images.dtype)
    fused_views = sampled * valid + original_view_images * (1 - valid)
    return fused_views, valid

def _level_size(size: int, level: int) -> int:
    return max(1, int(round(size / (2 ** level))))


def _projection_lod_map(
    grid: torch.Tensor,
    patch_size: Tuple[int, int],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Estimate source-patch pixel footprint per ERP pixel from projection derivatives."""

    if grid.shape[1] < 2 or grid.shape[2] < 2:
        return torch.zeros(grid.shape[:3], device=grid.device, dtype=grid.dtype)
    pixel_scale = grid.new_tensor([(patch_size[1] - 1) / 2, (patch_size[0] - 1) / 2])
    dx = ((grid[:, :, 1:] - grid[:, :, :-1]) * pixel_scale).norm(dim=-1)
    dy = ((grid[:, 1:] - grid[:, :-1]) * pixel_scale).norm(dim=-1)
    dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")
    footprint = torch.maximum(dx, dy).clamp_min(eps)
    return torch.log2(footprint).clamp_min(0)


def _get_projection_lod_map(
    view_dirs: torch.Tensor,
    fovs: Sequence[Tuple[float, float]],
    patch_size: Tuple[int, int],
    erp_size: Tuple[int, int],
    config: PixelFusionConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    key = _perspective_to_erp_cache_key(view_dirs, fovs, patch_size, erp_size, dtype, device, "lpw_lod")
    if key not in config.projection_cache.lod_maps:
        if config.lpw_lod_mode == "none":
            lod = torch.zeros(view_dirs.shape[0], erp_size[0], erp_size[1], device=device, dtype=dtype)
        else:
            grid, _ = _get_perspective_to_erp_grid(
                view_dirs,
                fovs,
                patch_size,
                erp_size,
                config,
                dtype=dtype,
                device=device,
                prefix="lpw_lod_grid",
            )
            lod = _projection_lod_map(grid, patch_size)
        config.projection_cache.lod_maps[key] = lod.unsqueeze(1)
    return config.projection_cache.lod_maps[key]
