"""Stable spherical ownership and latent write-back operations."""

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from experiments.legacy_spherical.diffpano_legacy.config import FUSION_DTYPE, PixelFusionConfig


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
