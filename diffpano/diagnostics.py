"""Optional diagnostics, intermediate exports, and timing helpers."""

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch

from diffpano.config import FUSION_DTYPE, PixelFusionConfig
from diffpano.projection import _erp_world_grid, _normalize_fovs, _world_to_erp_grid, extract_views_from_erp_standard


def save_exclusive_owner_diagnostics(
    diagnostics: Dict[str, torch.Tensor],
    config: PixelFusionConfig,
    cache_key: Tuple[Any, ...],
    *,
    pipeline_name: str,
    step_index: int,
) -> None:
    if not config.save_owner_map or cache_key in config.projection_cache.saved_owner_map_keys:
        return
    output_dir = Path(config.diagnostics_dir or "pixel_fusion_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {key: value.detach().cpu() for key, value in diagnostics.items()}
    torch.save(payload, output_dir / f"{pipeline_name}_exclusive_owner_map_step_{step_index:04d}.pt")
    config.projection_cache.saved_owner_map_keys.add(cache_key)

def _timed(timings: Dict[str, float], key: str, fn: Callable[[], Any], *, synchronize: bool = False) -> Any:
    if synchronize and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    if synchronize and torch.cuda.is_available():
        torch.cuda.synchronize()
    timings[key] = timings.get(key, 0.0) + (time.perf_counter() - start)
    return value


def _compute_projection_diagnostics(
    view_images: torch.Tensor,
    view_dirs: torch.Tensor,
    fovs: Union[Tuple[float, float], Sequence[Tuple[float, float]]],
    erp_height: int,
    erp_width: int,
    config: PixelFusionConfig,
) -> Dict[str, torch.Tensor]:
    from diffpano.fusion import aggregate_overlap_contributions, project_views_to_erp_standard

    world = _erp_world_grid(erp_height, erp_width, device=view_images.device, dtype=FUSION_DTYPE)
    erp_grid = _world_to_erp_grid(
        world.reshape(erp_height, erp_width, 3),
        erp_height=erp_height,
        erp_width=erp_width,
    )
    expected_x = 2 * (torch.arange(erp_width, device=view_images.device, dtype=FUSION_DTYPE) + 0.5) / erp_width - 1
    expected_y = 2 * (torch.arange(erp_height, device=view_images.device, dtype=FUSION_DTYPE) + 0.5) / erp_height - 1
    expected_x, expected_y = torch.meshgrid(expected_x, expected_y, indexing="xy")
    longitude_error = torch.remainder(erp_grid[..., 0] - expected_x + 1, 2) - 1
    center_error = torch.maximum(longitude_error.abs(), (erp_grid[..., 1] - expected_y).abs()).max()

    first_fov = _normalize_fovs(fovs, view_images.shape[0])[0]
    projected, mask, weight = project_views_to_erp_standard(
        view_images[:1],
        view_dirs[:1],
        first_fov,
        erp_height,
        erp_width,
        config,
    )
    round_trip_erp = aggregate_overlap_contributions(projected, mask, weight, "weighted_average")
    reconstructed, valid = extract_views_from_erp_standard(
        round_trip_erp.fused_values,
        round_trip_erp.valid_output_mask,
        view_images[:1],
        view_dirs[:1],
        first_fov,
        config,
    )
    error = (reconstructed - view_images[:1]).abs()
    valid_rgb = valid.expand_as(error)
    valid_error = error[valid_rgb > 0]
    if valid_error.numel():
        mean_error = valid_error.mean()
        max_error = valid_error.max()
    else:
        mean_error = error.new_tensor(float("nan"))
        max_error = error.new_tensor(float("nan"))

    return {
        "erp_grid_min": erp_grid.amin(dim=(0, 1)).detach(),
        "erp_grid_max": erp_grid.amax(dim=(0, 1)).detach(),
        "erp_pixel_center_max_error": center_error.detach().reshape(1),
        "perspective_round_trip_mean_error": mean_error.detach().reshape(1),
        "perspective_round_trip_max_error": max_error.detach().reshape(1),
    }

def write_pixel_fusion_diagnostics(
    diagnostics: Dict[str, torch.Tensor],
    config: PixelFusionConfig,
    *,
    step_index: Optional[int] = None,
    pipeline_name: str = "pixel_fusion",
) -> None:
    if not config.diagnostics_dir:
        return
    path = Path(config.diagnostics_dir)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {}
    for key, tensor in diagnostics.items():
        if key == "fused_erp" and not config.save_intermediates:
            continue
        if key in {"valid_mask", "contributor_count", "accumulated_weight", "overlap_mask"} and not (config.save_masks or config.save_diagnostics):
            continue
        payload[key] = tensor.detach().cpu()
    if payload:
        safe_pipeline_name = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in pipeline_name
        )
        step_suffix = f"_step-{step_index:04d}" if step_index is not None else ""
        torch.save(
            payload,
            path / f"pixel_fusion_{safe_pipeline_name}_{timestamp}_{os.getpid()}{step_suffix}.pt",
        )


# TEMPORARY DEBUG EXPORT START
# This intentionally lives in one removable block. It exports predicted-clean RGB ERPs without
# enabling the much larger tensor diagnostics payload.
def _temporary_save_rgb_erp_debug(
    erp: torch.Tensor,
    *,
    step_index: int,
    timestep: torch.Tensor,
    output_dir: Path,
    filename_prefix: str,
    description: str,
) -> str:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "interactive")
    timestep_value = float(timestep.detach().float().flatten()[0].cpu())
    timestep_label = f"{timestep_value:g}".replace(".", "p")
    filename = output_dir / (
        f"{filename_prefix}_job-{job_id}_step-{step_index:03d}_timestep-{timestep_label}.png"
    )

    image = erp.detach().float().cpu().clamp(-1, 1)
    if image.ndim != 3 or image.shape[0] < 3:
        raise ValueError(f"Expected RGB ERP [C,H,W], got {tuple(image.shape)}")
    image = ((image[:3] + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).contiguous()
    Image.fromarray(image.numpy(), mode="RGB").save(filename)
    print(f"Saved temporary {description} to {filename}")
    return str(filename)


def temporary_save_fused_clean_erp_debug(
    fused_erp: torch.Tensor,
    *,
    step_index: int,
    timestep: torch.Tensor,
    config: PixelFusionConfig,
    pipeline_name: str,
) -> Optional[str]:
    if not config.temporary_save_fused_erp_per_step:
        return None

    output_dir = Path(config.temporary_fused_erp_dir or "outputs/debug/fused_clean_erp")
    return _temporary_save_rgb_erp_debug(
        fused_erp,
        step_index=step_index,
        timestep=timestep,
        output_dir=output_dir,
        filename_prefix=pipeline_name,
        description="fused clean ERP",
    )


def temporary_save_original_clean_erp_debug(
    decoded_views: Iterable[Tuple[torch.Tensor, torch.Tensor, Tuple[float, float]]],
    *,
    step_index: int,
    timestep: torch.Tensor,
    erp_height: int,
    erp_width: int,
    weighted_average_temperature: float,
    config: PixelFusionConfig,
    pipeline_name: str,
) -> Optional[str]:
    """Render original predicted-clean views with the same standard ERP projector as fusion."""

    if config.pixel_fusion_enabled or not config.temporary_save_original_clean_erp_per_step:
        return None

    from diffpano.fusion import render_views_to_erp_standard_weighted

    _ = weighted_average_temperature  # Kept in the temporary API for existing pipeline call sites.
    aggregate = render_views_to_erp_standard_weighted(decoded_views, erp_height, erp_width, config)
    if aggregate is None:
        return None
    output_dir = Path(config.temporary_original_clean_erp_dir or "outputs/debug/original_clean_erp")
    return _temporary_save_rgb_erp_debug(
        aggregate.fused_values,
        step_index=step_index,
        timestep=timestep,
        output_dir=output_dir,
        filename_prefix=f"{pipeline_name}_original_clean",
        description="original clean ERP",
    )
# TEMPORARY DEBUG EXPORT END
