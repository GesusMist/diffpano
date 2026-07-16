import hashlib
import math
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn.functional as F

from .spherical_functions import SphericalFunctions


TensorAdapter = Callable[[torch.Tensor], torch.Tensor]
FUSION_DTYPE = torch.float32


@dataclass
class ProjectionCache:
    """Small per-call cache for deterministic projection grids and weight maps."""

    grids: Dict[Tuple[Any, ...], Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    weights: Dict[Tuple[Any, ...], torch.Tensor] = field(default_factory=dict)
    lod_maps: Dict[Tuple[Any, ...], torch.Tensor] = field(default_factory=dict)
    owner_maps: Dict[Tuple[Any, ...], "ExclusiveOwnerMap"] = field(default_factory=dict)
    saved_owner_map_keys: Set[Tuple[Any, ...]] = field(default_factory=set)


@dataclass
class PixelFusionConfig:
    pixel_fusion_enabled: bool = False
    random_seed: Optional[int] = None
    pixel_fusion_every_n_steps: int = 1
    pixel_fusion_start_ratio: float = 0.0
    pixel_fusion_end_ratio: float = 1.0

    warp_mode: str = "standard"
    aggregation_mode: str = "weighted_average"
    weight_mode: str = "distance_to_boundary"

    lpw_num_levels: int = 4
    lpw_lod_mode: str = "jacobian"
    lpw_lod_interpolation: str = "linear"
    erp_vertical_padding_mode: str = "reflect"
    erp_to_perspective_interpolation_mode: str = "bilinear"

    dpa_alpha: float = 1.0
    dpa_power: float = 1.0
    dpa_eps: float = 1e-6

    reinjection_mode: str = "noise_consistent"
    reinjection_strength: float = 1.0
    spherical_writeback_mode: str = "exclusive"
    spherical_owner_mode: str = "max_center_weight"
    exclusive_owner_map_static: bool = True
    exclusive_uncovered_mode: str = "error"
    save_owner_map: bool = False

    time_travel_enabled: bool = False
    time_travel_every_n_steps: int = 1
    time_travel_jump_length: int = 1
    time_travel_num_repeats: int = 1
    time_travel_strength: float = 1.0

    vae_chunk_size: int = 4
    save_intermediates: bool = False
    save_masks: bool = False
    save_diagnostics: bool = False
    measure_performance: bool = False
    diagnostics_dir: Optional[str] = None

    projection_chunk_size: int = 1
    vae_sample_posterior: bool = False

    # TEMPORARY DEBUG EXPORT: remove these fields with the temporary debug helpers below.
    temporary_save_fused_erp_per_step: bool = False
    temporary_fused_erp_dir: Optional[str] = None
    temporary_save_original_clean_erp_per_step: bool = False
    temporary_original_clean_erp_dir: Optional[str] = None
    projection_cache: ProjectionCache = field(default_factory=ProjectionCache)

    def to_dict(self) -> Dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "projection_cache"
        }

    @classmethod
    def from_any(cls, value: Optional[Union["PixelFusionConfig", Dict[str, Any], str]]) -> "PixelFusionConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return load_pixel_fusion_config(value)
        if isinstance(value, dict):
            allowed = {field_name for field_name in cls.__dataclass_fields__ if field_name != "projection_cache"}
            return cls(**{key: _coerce_config_value(key, item) for key, item in value.items() if key in allowed})
        raise TypeError(f"Unsupported pixel fusion config type: {type(value)!r}")

    def validate(self) -> None:
        if self.warp_mode not in {"standard", "lpw"}:
            raise ValueError(f"Unsupported warp_mode={self.warp_mode!r}")
        if self.aggregation_mode not in {"average", "weighted_average", "detail_preserving_average"}:
            raise ValueError(f"Unsupported aggregation_mode={self.aggregation_mode!r}")
        if self.weight_mode not in {"uniform", "cosine", "gaussian", "distance_to_boundary"}:
            raise ValueError(f"Unsupported weight_mode={self.weight_mode!r}")
        if self.reinjection_mode not in {"noise_consistent", "replace", "weighted_replace", "residual"}:
            raise ValueError(f"Unsupported reinjection_mode={self.reinjection_mode!r}")
        if self.spherical_writeback_mode not in {"weighted_average", "exclusive"}:
            raise ValueError(f"Unsupported spherical_writeback_mode={self.spherical_writeback_mode!r}")
        if self.spherical_owner_mode != "max_center_weight":
            raise ValueError(f"Unsupported spherical_owner_mode={self.spherical_owner_mode!r}")
        if self.exclusive_uncovered_mode not in {"error", "weighted_average_fallback"}:
            raise ValueError(f"Unsupported exclusive_uncovered_mode={self.exclusive_uncovered_mode!r}")
        if self.random_seed is not None and (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or not 0 <= self.random_seed <= 2**63 - 1
        ):
            raise ValueError("random_seed must be null or an integer from 0 through 2**63 - 1")
        if self.pixel_fusion_every_n_steps < 1:
            raise ValueError("pixel_fusion_every_n_steps must be >= 1")
        if self.vae_chunk_size < 1:
            raise ValueError("vae_chunk_size must be >= 1")
        if self.projection_chunk_size < 1:
            raise ValueError("projection_chunk_size must be >= 1")
        if self.lpw_num_levels < 1:
            raise ValueError("lpw_num_levels must be >= 1")
        if self.lpw_lod_mode not in {"jacobian", "none"}:
            raise ValueError(f"Unsupported lpw_lod_mode={self.lpw_lod_mode!r}")
        if self.lpw_lod_interpolation not in {"linear", "nearest"}:
            raise ValueError(f"Unsupported lpw_lod_interpolation={self.lpw_lod_interpolation!r}")
        if self.erp_to_perspective_interpolation_mode not in {"bilinear", "nearest"}:
            raise ValueError(
                "Unsupported erp_to_perspective_interpolation_mode="
                f"{self.erp_to_perspective_interpolation_mode!r}"
            )


@dataclass
class OverlapAggregationResult:
    fused_values: torch.Tensor
    accumulated_weight: torch.Tensor
    contributor_count: torch.Tensor
    valid_output_mask: torch.Tensor


@dataclass
class ExclusiveOwnerMap:
    owner_patch_id: torch.Tensor
    owner_score: torch.Tensor
    coverage_count: torch.Tensor
    covered_mask: torch.Tensor


@dataclass
class ExclusiveWriteBackResult:
    latents: torch.Tensor
    exclusive_write_count: torch.Tensor


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


def load_pixel_fusion_config(path: str) -> PixelFusionConfig:
    """Load a YAML config through OmegaConf, matching the repository's existing config style."""

    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ImportError("OmegaConf is required to load pixel fusion YAML configs") from exc

    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Pixel fusion config must contain a mapping: {path}")
    return PixelFusionConfig.from_any(data)


def _coerce_config_value(key: str, value: Any) -> Any:
    bool_fields = {
        "pixel_fusion_enabled",
        "time_travel_enabled",
        "save_intermediates",
        "save_masks",
        "save_diagnostics",
        "measure_performance",
        "vae_sample_posterior",
        "exclusive_owner_map_static",
        "save_owner_map",
        "temporary_save_fused_erp_per_step",
        "temporary_save_original_clean_erp_per_step",
    }
    int_fields = {
        "pixel_fusion_every_n_steps",
        "lpw_num_levels",
        "time_travel_every_n_steps",
        "time_travel_jump_length",
        "time_travel_num_repeats",
        "vae_chunk_size",
        "projection_chunk_size",
        "random_seed",
    }
    float_fields = {
        "pixel_fusion_start_ratio",
        "pixel_fusion_end_ratio",
        "dpa_alpha",
        "dpa_power",
        "dpa_eps",
        "reinjection_strength",
        "time_travel_strength",
    }
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"none", "null"}:
            return None
        if key in bool_fields:
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        if key in int_fields:
            return int(value)
        if key in float_fields:
            return float(value)
    return value


def build_pixel_fusion_config(
    pixel_fusion_config: Optional[Union[PixelFusionConfig, Dict[str, Any], str]] = None,
    pixel_fusion_config_path: Optional[str] = None,
    **overrides,
) -> PixelFusionConfig:
    config = PixelFusionConfig.from_any(pixel_fusion_config_path or pixel_fusion_config)
    valid_keys = {key for key in PixelFusionConfig.__dataclass_fields__ if key != "projection_cache"}
    for key, value in overrides.items():
        if key in valid_keys and value is not None:
            setattr(config, key, _coerce_config_value(key, value))
    config.validate()
    return config


def apply_configured_random_seed(
    generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    config: PixelFusionConfig,
    *,
    device: torch.device,
) -> Optional[Union[torch.Generator, List[torch.Generator]]]:
    """Create the generation generator when the experiment config specifies a seed."""

    if config.random_seed is None:
        return generator
    return torch.Generator(device=device).manual_seed(config.random_seed)


def _tensor_content_hash(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().contiguous().cpu()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _exclusive_owner_cache_key(
    num_spherical_points: int,
    patch_indices: Sequence[torch.Tensor],
    patch_scores: Sequence[torch.Tensor],
    patch_ids: Sequence[int],
    patch_view_dirs: Sequence[torch.Tensor],
    patch_fovs: Sequence[Tuple[float, float]],
    device: torch.device,
) -> Tuple[Any, ...]:
    if not (
        len(patch_indices)
        == len(patch_scores)
        == len(patch_ids)
        == len(patch_view_dirs)
        == len(patch_fovs)
    ):
        raise ValueError("Owner-map cache inputs must contain one entry per patch")

    patch_signatures = []
    for patch_id, indices, scores, view_dir, fov in sorted(
        zip(patch_ids, patch_indices, patch_scores, patch_view_dirs, patch_fovs), key=lambda item: int(item[0])
    ):
        indices_flat = indices.detach().reshape(-1).long()
        scores_flat = scores.detach().reshape(-1).to(dtype=FUSION_DTYPE)
        patch_signatures.append(
            (
                int(patch_id),
                tuple(indices.shape),
                _tensor_content_hash(indices_flat),
                tuple(scores.shape),
                _tensor_content_hash(scores_flat),
                _tensor_content_hash(view_dir.detach().reshape(-1).to(dtype=FUSION_DTYPE)),
                tuple(float(value) for value in fov),
            )
        )
    return (int(num_spherical_points), str(device), tuple(patch_signatures))


def build_exclusive_owner_map(
    num_spherical_points: int,
    patch_indices: Sequence[torch.Tensor],
    patch_scores: Sequence[torch.Tensor],
    patch_ids: Sequence[int],
    *,
    device: torch.device,
) -> ExclusiveOwnerMap:
    """Assign each covered spherical point to its highest-scoring stable patch ID."""

    if num_spherical_points < 1:
        raise ValueError("num_spherical_points must be positive")
    if not (len(patch_indices) == len(patch_scores) == len(patch_ids)):
        raise ValueError("patch_indices, patch_scores, and patch_ids must have equal lengths")
    stable_patch_ids = [int(patch_id) for patch_id in patch_ids]
    if len(set(stable_patch_ids)) != len(stable_patch_ids):
        raise ValueError("Exclusive ownership requires unique stable patch IDs")

    owner_patch_id = torch.full((num_spherical_points,), -1, dtype=torch.long, device=device)
    owner_score = torch.full((num_spherical_points,), -torch.inf, dtype=FUSION_DTYPE, device=device)
    coverage_count = torch.zeros((num_spherical_points,), dtype=torch.long, device=device)

    entries = sorted(zip(stable_patch_ids, patch_indices, patch_scores), key=lambda item: item[0])
    for patch_id, indices, scores in entries:
        indices_flat = indices.detach().reshape(-1).to(device=device, dtype=torch.long)
        scores_flat = scores.detach().reshape(-1).to(device=device, dtype=FUSION_DTYPE)
        if indices_flat.numel() != scores_flat.numel():
            raise ValueError(
                f"Patch {patch_id} has {indices_flat.numel()} indices but {scores_flat.numel()} ownership scores"
            )
        if indices_flat.numel() == 0:
            continue
        if indices_flat.min().item() < 0 or indices_flat.max().item() >= num_spherical_points:
            raise IndexError(f"Patch {patch_id} contains a spherical index outside [0, {num_spherical_points})")
        if torch.unique(indices_flat).numel() != indices_flat.numel():
            raise ValueError(f"Patch {patch_id} contains duplicate spherical indices")
        if not torch.isfinite(scores_flat).all():
            raise ValueError(f"Patch {patch_id} contains a non-finite ownership score")

        coverage_count.index_add_(0, indices_flat, torch.ones_like(indices_flat))
        current_score = owner_score[indices_flat]
        current_owner = owner_patch_id[indices_flat]
        wins = (scores_flat > current_score) | ((scores_flat == current_score) & (patch_id < current_owner))
        winning_indices = indices_flat[wins]
        owner_score[winning_indices] = scores_flat[wins]
        owner_patch_id[winning_indices] = patch_id

    covered_mask = coverage_count > 0
    if not torch.equal(covered_mask, torch.isfinite(owner_score)):
        raise AssertionError("Exclusive owner-map coverage and finite owner scores disagree")
    return ExclusiveOwnerMap(
        owner_patch_id=owner_patch_id,
        owner_score=owner_score,
        coverage_count=coverage_count,
        covered_mask=covered_mask,
    )


def get_or_build_exclusive_owner_map(
    num_spherical_points: int,
    patch_indices: Sequence[torch.Tensor],
    patch_scores: Sequence[torch.Tensor],
    patch_ids: Sequence[int],
    patch_view_dirs: Sequence[torch.Tensor],
    patch_fovs: Sequence[Tuple[float, float]],
    config: PixelFusionConfig,
    *,
    device: torch.device,
) -> Tuple[ExclusiveOwnerMap, Tuple[Any, ...], bool]:
    if config.spherical_owner_mode != "max_center_weight":
        raise ValueError(f"Unsupported spherical_owner_mode={config.spherical_owner_mode!r}")
    cache_key = _exclusive_owner_cache_key(
        num_spherical_points,
        patch_indices,
        patch_scores,
        patch_ids,
        patch_view_dirs,
        patch_fovs,
        device,
    )
    if config.exclusive_owner_map_static and cache_key in config.projection_cache.owner_maps:
        return config.projection_cache.owner_maps[cache_key], cache_key, True

    owner_map = build_exclusive_owner_map(
        num_spherical_points,
        patch_indices,
        patch_scores,
        patch_ids,
        device=device,
    )
    if config.exclusive_owner_map_static:
        config.projection_cache.owner_maps[cache_key] = owner_map
    return owner_map, cache_key, False


def write_back_views_weighted_average(
    spherical_latent_template: torch.Tensor,
    corrected_view_latents: Sequence[torch.Tensor],
    patch_indices: Sequence[torch.Tensor],
    patch_scores: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Preserve SphereDiff's weighted spherical latent accumulation behavior."""

    if not (len(corrected_view_latents) == len(patch_indices) == len(patch_scores)):
        raise ValueError("Weighted write-back inputs must contain one entry per patch")
    if spherical_latent_template.ndim < 2:
        raise ValueError("Spherical latent template must include batch and spherical-point dimensions")

    latents_next = torch.zeros_like(spherical_latent_template)
    latents_next_cnt = torch.zeros_like(spherical_latent_template)
    for corrected_patch, indices, scores in zip(corrected_view_latents, patch_indices, patch_scores):
        indices_flat = indices.reshape(-1).to(device=spherical_latent_template.device, dtype=torch.long)
        corrected_patch = corrected_patch.to(
            device=spherical_latent_template.device, dtype=spherical_latent_template.dtype
        )
        if corrected_patch.shape[:-1] != spherical_latent_template.shape[:-1]:
            raise ValueError(
                f"Corrected patch prefix {corrected_patch.shape[:-1]} does not match spherical latent prefix "
                f"{spherical_latent_template.shape[:-1]}"
            )
        if corrected_patch.shape[-1] != indices_flat.numel():
            raise ValueError("Corrected patch point count does not match its sampled indices")
        weight = scores.reshape(-1).to(device=corrected_patch.device, dtype=corrected_patch.dtype)
        if weight.numel() != indices_flat.numel():
            raise ValueError("Patch score count does not match its sampled indices")
        weight = weight.reshape((1,) * (corrected_patch.ndim - 2) + (-1,))
        for batch_index in range(spherical_latent_template.shape[0]):
            latents_next[batch_index, ..., indices_flat] += corrected_patch[batch_index] * weight
            latents_next_cnt[batch_index, ..., indices_flat] += weight

    latents_next_cnt[latents_next_cnt == 0] = 1
    return latents_next / latents_next_cnt


def write_back_views_exclusive(
    spherical_latent_template: torch.Tensor,
    corrected_view_latents: Sequence[torch.Tensor],
    patch_indices: Sequence[torch.Tensor],
    patch_ids: Sequence[int],
    owner_map: ExclusiveOwnerMap,
    *,
    uncovered_mode: str = "error",
    weighted_average_fallback: Optional[torch.Tensor] = None,
    geometry_summary: str = "unavailable",
) -> ExclusiveWriteBackResult:
    """Write every covered spherical point exactly once from its stable owner patch."""

    if uncovered_mode not in {"error", "weighted_average_fallback"}:
        raise ValueError(f"Unsupported exclusive uncovered mode {uncovered_mode!r}")
    if not (len(corrected_view_latents) == len(patch_indices) == len(patch_ids)):
        raise ValueError("Exclusive write-back inputs must contain one entry per patch")
    if len(set(int(patch_id) for patch_id in patch_ids)) != len(patch_ids):
        raise ValueError("Exclusive write-back requires unique stable patch IDs")

    num_spherical_points = spherical_latent_template.shape[-1]
    if owner_map.owner_patch_id.shape != (num_spherical_points,):
        raise ValueError("Owner map does not match the spherical latent point count")
    uncovered_count = int((~owner_map.covered_mask).sum().item())
    if uncovered_count and uncovered_mode == "error":
        uncovered_percent = 100.0 * uncovered_count / num_spherical_points
        raise RuntimeError(
            "Exclusive spherical write-back found "
            f"{uncovered_count}/{num_spherical_points} uncovered points ({uncovered_percent:.4f}%) across "
            f"{len(patch_ids)} patches; view geometry: {geometry_summary}"
        )
    if uncovered_count and weighted_average_fallback is None:
        raise ValueError("weighted_average_fallback mode requires a complete weighted-average spherical result")

    output = torch.empty_like(spherical_latent_template)
    write_count = torch.zeros((num_spherical_points,), dtype=torch.long, device=spherical_latent_template.device)
    for corrected_patch, indices, patch_id in zip(corrected_view_latents, patch_indices, patch_ids):
        indices_flat = indices.reshape(-1).to(device=spherical_latent_template.device, dtype=torch.long)
        corrected_patch = corrected_patch.to(
            device=spherical_latent_template.device, dtype=spherical_latent_template.dtype
        )
        if corrected_patch.shape[:-1] != spherical_latent_template.shape[:-1]:
            raise ValueError(
                f"Corrected patch prefix {corrected_patch.shape[:-1]} does not match spherical latent prefix "
                f"{spherical_latent_template.shape[:-1]}"
            )
        if corrected_patch.shape[-1] != indices_flat.numel():
            raise ValueError("Corrected patch point count does not match its sampled indices")
        local_owner_mask = owner_map.owner_patch_id[indices_flat] == int(patch_id)
        owned_spherical_indices = indices_flat[local_owner_mask]
        output[..., owned_spherical_indices] = corrected_patch[..., local_owner_mask]
        write_count.index_add_(0, owned_spherical_indices, torch.ones_like(owned_spherical_indices))

    if not write_count[owner_map.covered_mask].eq(1).all():
        raise AssertionError("Every covered spherical point must be written exactly once in exclusive mode")
    if uncovered_count:
        fallback = weighted_average_fallback.to(device=output.device, dtype=output.dtype)
        if fallback.shape != output.shape:
            raise ValueError("Weighted-average fallback shape does not match spherical latent shape")
        output[..., ~owner_map.covered_mask] = fallback[..., ~owner_map.covered_mask]

    return ExclusiveWriteBackResult(latents=output, exclusive_write_count=write_count)


def exclusive_owner_diagnostics(
    owner_map: ExclusiveOwnerMap,
    exclusive_write_count: torch.Tensor,
    patch_ids: Sequence[int],
) -> Dict[str, torch.Tensor]:
    unique_patch_ids = sorted(set(int(patch_id) for patch_id in patch_ids))
    histogram = torch.tensor(
        [
            [patch_id, int((owner_map.owner_patch_id == patch_id).sum().item())]
            for patch_id in unique_patch_ids
        ],
        dtype=torch.long,
        device=owner_map.owner_patch_id.device,
    )
    coverage_float = owner_map.coverage_count.to(dtype=FUSION_DTYPE)
    multiply_covered_count = (owner_map.coverage_count > 1).sum()
    total = max(owner_map.coverage_count.numel(), 1)
    return {
        "owner_patch_id": owner_map.owner_patch_id.detach(),
        "owner_score": owner_map.owner_score.detach(),
        "coverage_count": owner_map.coverage_count.detach(),
        "covered_mask": owner_map.covered_mask.detach(),
        "owner_patch_histogram": histogram.detach(),
        "uncovered_count": (~owner_map.covered_mask).sum().detach().reshape(1),
        "multiply_covered_count": multiply_covered_count.detach().reshape(1),
        "exclusive_write_count": exclusive_write_count.detach(),
        "minimum_coverage_count": owner_map.coverage_count.min().detach().reshape(1),
        "maximum_coverage_count": owner_map.coverage_count.max().detach().reshape(1),
        "mean_coverage_count": coverage_float.mean().detach().reshape(1),
        "multiply_covered_percent": coverage_float.new_tensor(
            [100.0 * multiply_covered_count.item() / total]
        ),
    }


def summarize_patch_geometry(
    patch_ids: Sequence[int],
    patch_view_dirs: Sequence[torch.Tensor],
    patch_fovs: Sequence[Tuple[float, float]],
) -> str:
    entries = []
    for patch_id, view_dir, fov in zip(patch_ids, patch_view_dirs, patch_fovs):
        direction = view_dir.detach().reshape(-1, 3)[0].to(dtype=FUSION_DTYPE).cpu().tolist()
        entries.append(
            f"id={int(patch_id)} dir=({direction[0]:.4f},{direction[1]:.4f},{direction[2]:.4f}) "
            f"fov=({float(fov[0]):.2f},{float(fov[1]):.2f})"
        )
    return "; ".join(entries)


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


def _chunk_tensor(tensor: torch.Tensor, chunk_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, tensor.shape[0], chunk_size):
        yield tensor[start:start + chunk_size]


def _vae_scaling_factor(vae: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    value = getattr(getattr(vae, "config", None), "scaling_factor", 1.0)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _vae_shift_factor(vae: Any, *, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    value = getattr(getattr(vae, "config", None), "shift_factor", None)
    if value is None:
        return None
    return torch.as_tensor(value, device=device, dtype=dtype)


def _decode_latent_input(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    scaling_factor = _vae_scaling_factor(vae, device=latents.device, dtype=latents.dtype)
    shift_factor = _vae_shift_factor(vae, device=latents.device, dtype=latents.dtype)
    latents = latents / scaling_factor
    if shift_factor is not None:
        latents = latents + shift_factor
    return latents


def _encode_latent_output(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    scaling_factor = _vae_scaling_factor(vae, device=latents.device, dtype=latents.dtype)
    shift_factor = _vae_shift_factor(vae, device=latents.device, dtype=latents.dtype)
    if shift_factor is not None:
        latents = latents - shift_factor
    return latents * scaling_factor


def _extract_encoded_latents(encoded: Any, *, generator: Optional[torch.Generator], sample_posterior: bool) -> torch.Tensor:
    if hasattr(encoded, "latent_dist"):
        if sample_posterior:
            return encoded.latent_dist.sample(generator=generator)
        if hasattr(encoded.latent_dist, "mean"):
            return encoded.latent_dist.mean
        if hasattr(encoded.latent_dist, "mode"):
            return encoded.latent_dist.mode()
    if hasattr(encoded, "latents"):
        return encoded.latents
    if hasattr(encoded, "latent"):
        return encoded.latent
    if isinstance(encoded, (tuple, list)):
        return encoded[0]
    return encoded


def decode_view_latents(
    vae: Any,
    view_latents: torch.Tensor,
    config: PixelFusionConfig,
) -> torch.Tensor:
    """Decode VAE latents [views, latent_channels, h, w] into RGB [-1, 1] [views, 3, H, W]."""

    original_dtype = view_latents.dtype
    vae_dtype = getattr(vae, "dtype", original_dtype)
    decoded: List[torch.Tensor] = []
    with torch.inference_mode():
        for chunk in _chunk_tensor(view_latents, config.vae_chunk_size):
            model_chunk = _decode_latent_input(vae, chunk.to(dtype=vae_dtype))
            decoded_chunk = vae.decode(model_chunk, return_dict=False)[0]
            decoded.append(decoded_chunk.to(dtype=original_dtype))
    return torch.cat(decoded, dim=0)


def encode_view_images(
    vae: Any,
    view_images: torch.Tensor,
    config: PixelFusionConfig,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Encode RGB [-1, 1] [views, 3, H, W] into scaled VAE latents [views, latent_channels, h, w]."""

    original_dtype = view_images.dtype
    vae_dtype = getattr(vae, "dtype", original_dtype)
    encoded: List[torch.Tensor] = []
    with torch.inference_mode():
        for chunk in _chunk_tensor(view_images, config.vae_chunk_size):
            encoded_chunk = vae.encode(chunk.to(dtype=vae_dtype))
            latents = _extract_encoded_latents(
                encoded_chunk,
                generator=generator,
                sample_posterior=config.vae_sample_posterior,
            )
            encoded.append(_encode_latent_output(vae, latents).to(dtype=original_dtype))
    return torch.cat(encoded, dim=0)


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


def _scheduler_index_for_timestep(scheduler: Any, timestep: torch.Tensor) -> int:
    schedule_timesteps = scheduler.timesteps.to(device=timestep.device)
    if timestep.ndim == 0:
        t = timestep
    else:
        t = timestep.flatten()[0]
    if hasattr(scheduler, "index_for_timestep"):
        return int(scheduler.index_for_timestep(t, schedule_timesteps))
    matches = (schedule_timesteps == t).nonzero()
    if matches.numel() > 0:
        return int(matches.flatten()[0])
    return int(torch.argmin((schedule_timesteps.to(dtype=torch.float32) - t.to(dtype=torch.float32)).abs()))


def _scheduler_sigma_pair(
    scheduler: Any,
    timestep: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(scheduler, "sigmas"):
        raise ValueError(
            f"Pixel fusion requires a scheduler with an explicit sigma schedule, got {scheduler.__class__.__name__}"
        )
    index = _scheduler_index_for_timestep(scheduler, timestep)
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    if index + 1 >= sigmas.shape[0]:
        raise ValueError("Scheduler sigmas do not contain sigma_next for the current timestep")
    return sigmas[index], sigmas[index + 1]


def _scheduler_prediction_type(scheduler: Any) -> Optional[str]:
    prediction_type = getattr(getattr(scheduler, "config", None), "prediction_type", None)
    if prediction_type is not None:
        return str(prediction_type)
    if "flowmatch" in scheduler.__class__.__name__.lower():
        return "flow_prediction"
    return None


def predict_clean_latents(
    scheduler: Any,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert the configured scheduler prediction into predicted-clean view latents x0."""

    sample = sample.to(dtype=FUSION_DTYPE)
    model_output = model_output.to(device=sample.device, dtype=FUSION_DTYPE)
    sigma, sigma_next = _scheduler_sigma_pair(scheduler, timestep, device=sample.device, dtype=FUSION_DTYPE)
    while sigma.ndim < sample.ndim:
        sigma = sigma.unsqueeze(-1)
        sigma_next = sigma_next.unsqueeze(-1)
    prediction_type = _scheduler_prediction_type(scheduler)
    if prediction_type == "flow_prediction":
        # Both FlowMatch Euler and SANA's flow-sigma DPM-Solver use x_t = x0 + sigma * flow.
        clean = sample - sigma * model_output
    elif prediction_type == "sample":
        clean = model_output
    elif prediction_type in {"epsilon", "v_prediction"} and hasattr(scheduler, "_sigma_to_alpha_sigma_t"):
        alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma)
        if prediction_type == "epsilon":
            clean = (sample - sigma_t * model_output) / alpha_t.clamp_min(torch.finfo(sample.dtype).eps)
        else:
            clean = alpha_t * sample - sigma_t * model_output
    else:
        raise ValueError(
            f"Unsupported scheduler prediction_type={prediction_type!r} for predicted-clean pixel fusion "
            f"with {scheduler.__class__.__name__}"
        )
    return clean, sigma, sigma_next


def reinject_fused_latents(
    original_clean_latents: torch.Tensor,
    fused_clean_latents: torch.Tensor,
    original_prev_latents: torch.Tensor,
    model_output: torch.Tensor,
    sigma_next: torch.Tensor,
    config: PixelFusionConfig,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    next_clean_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    output_dtype = original_prev_latents.dtype
    original_clean_latents = original_clean_latents.to(dtype=FUSION_DTYPE)
    fused_clean_latents = fused_clean_latents.to(device=original_clean_latents.device, dtype=FUSION_DTYPE)
    original_prev_latents = original_prev_latents.to(device=original_clean_latents.device, dtype=FUSION_DTYPE)
    model_output = model_output.to(device=original_clean_latents.device, dtype=FUSION_DTYPE)
    sigma_next = sigma_next.to(device=original_clean_latents.device, dtype=FUSION_DTYPE)
    if next_clean_weight is not None:
        next_clean_weight = next_clean_weight.to(device=original_clean_latents.device, dtype=FUSION_DTYPE)
    strength = float(config.reinjection_strength)
    clean_delta = fused_clean_latents - original_clean_latents
    if config.reinjection_mode == "noise_consistent":
        if next_clean_weight is None:
            next_clean_weight = 1 - sigma_next
        # Preserve the scheduler's actual update and alter only its next-step clean-sample component.
        # For flow matching, x_next = alpha_next * x0 + sigma_next * noise, so injecting the full
        # x0 delta at early noisy steps would over-correct by roughly 1 / alpha_next.
        result = original_prev_latents + strength * next_clean_weight * clean_delta
        return result.to(dtype=output_dtype)
    if config.reinjection_mode == "replace":
        result = original_prev_latents * (1 - strength) + fused_clean_latents * strength
        return result.to(dtype=output_dtype)
    if config.reinjection_mode == "weighted_replace":
        if valid_mask is None:
            valid_mask = torch.ones_like(fused_clean_latents[:, :1])
        else:
            valid_mask = valid_mask.to(device=original_clean_latents.device, dtype=FUSION_DTYPE)
        while valid_mask.ndim < fused_clean_latents.ndim:
            valid_mask = valid_mask.unsqueeze(-1)
        alpha = (valid_mask * strength).to(fused_clean_latents.dtype)
        result = original_prev_latents * (1 - alpha) + fused_clean_latents * alpha
        return result.to(dtype=output_dtype)
    if config.reinjection_mode == "residual":
        result = original_prev_latents + strength * clean_delta
        return result.to(dtype=output_dtype)
    raise ValueError(f"Unsupported reinjection_mode={config.reinjection_mode!r}")


def _adapt_latents(latents: torch.Tensor, adapter: Optional[TensorAdapter]) -> torch.Tensor:
    return adapter(latents) if adapter is not None else latents


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
) -> PixelFusionResult:
    """Decode predicted-clean view latents, fuse in temporary ERP RGB, encode, and reinject.

    clean/current/model/prev latents use the scheduler representation expected by the original write-back path.
    The optional adapters convert that representation to/from VAE image latents before decoding and after encoding.
    """

    config.validate()
    timings: Dict[str, float] = {}
    sigma_next = torch.zeros((), device=current_latents.device, dtype=FUSION_DTYPE)
    next_clean_weight = torch.ones((), device=current_latents.device, dtype=FUSION_DTYPE)
    if config.reinjection_mode == "noise_consistent":
        prediction_type = _scheduler_prediction_type(scheduler)
        if prediction_type != "flow_prediction":
            raise ValueError(
                "noise_consistent reinjection currently requires flow_prediction so the scheduler state can be "
                f"preserved exactly; got prediction_type={prediction_type!r}"
            )
        sigma_next = _scheduler_sigma_pair(scheduler, timestep, device=current_latents.device, dtype=FUSION_DTYPE)[1]
        if hasattr(scheduler, "_sigma_to_alpha_sigma_t"):
            next_clean_weight = scheduler._sigma_to_alpha_sigma_t(sigma_next)[0]
        else:
            next_clean_weight = 1 - sigma_next
        while sigma_next.ndim < current_latents.ndim:
            sigma_next = sigma_next.unsqueeze(-1)
            next_clean_weight = next_clean_weight.unsqueeze(-1)

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

    fused_vae_latents = _timed(
        timings,
        "vae_encode",
        lambda: encode_view_images(vae, fused_views, config, generator=generator),
        synchronize=config.measure_performance,
    )
    fused_clean_latents_fp32 = _adapt_latents(fused_vae_latents, vae_latents_to_latent).to(dtype=FUSION_DTYPE)
    latent_valid_mask = F.interpolate(
        view_valid_mask,
        size=fused_vae_latents.shape[-2:],
        mode="area",
    ).clamp(0, 1)
    if vae_latents_to_latent is not None:
        latent_valid_mask = vae_latents_to_latent(latent_valid_mask.expand_as(fused_vae_latents))
    fused_prev_latents = _timed(
        timings,
        "reinjection",
        lambda: reinject_fused_latents(
            clean_latents,
            fused_clean_latents_fp32,
            prev_latents,
            model_output,
            sigma_next,
            config,
            valid_mask=latent_valid_mask,
            next_clean_weight=next_clean_weight,
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
        }
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
        write_pixel_fusion_diagnostics(diagnostics, config)

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


def write_pixel_fusion_diagnostics(diagnostics: Dict[str, torch.Tensor], config: PixelFusionConfig) -> None:
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
        torch.save(payload, path / f"pixel_fusion_{timestamp}_{os.getpid()}.pt")


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

    output_dir = Path(config.temporary_fused_erp_dir or "/home/shig/diffpano/debug_fused_clean_erp")
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

    _ = weighted_average_temperature  # Kept in the temporary API for existing pipeline call sites.
    aggregate = render_views_to_erp_standard_weighted(decoded_views, erp_height, erp_width, config)
    if aggregate is None:
        return None
    output_dir = Path(config.temporary_original_clean_erp_dir or "/home/shig/diffpano/debug_original_clean_erp")
    return _temporary_save_rgb_erp_debug(
        aggregate.fused_values,
        step_index=step_index,
        timestep=timestep,
        output_dir=output_dir,
        filename_prefix=f"{pipeline_name}_original_clean",
        description="original clean ERP",
    )
# TEMPORARY DEBUG EXPORT END
