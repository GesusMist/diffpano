"""Geometry-free planar layout, exact crops, and streaming RGB fusion."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from diffpano.config import FusionConfig, PlanarConfig
from diffpano.fusion import create_view_weight_map


@dataclass(frozen=True)
class PlanarPatch:
    index: int
    y: int
    x: int
    size: int


@dataclass(frozen=True)
class PlanarPatchLayout:
    canvas_height: int
    canvas_width: int
    patch_size: int
    patches: Tuple[PlanarPatch, ...]

    @property
    def num_patches(self) -> int:
        return len(self.patches)


@dataclass
class PlanarFusionResult:
    canvas_rgb: torch.Tensor
    accumulated_weight: torch.Tensor
    contributor_count: torch.Tensor
    coverage_mask: torch.Tensor


def _patch_starts(length: int, patch_size: int, stride: int) -> Tuple[int, ...]:
    """Return a covering lattice with an explicitly edge-aligned final patch."""

    if patch_size < 1 or stride < 1:
        raise ValueError("patch_size and stride must be positive")
    if stride > patch_size:
        raise ValueError("stride cannot exceed patch_size because that would leave gaps")
    if patch_size > length:
        raise ValueError(f"Patch size {patch_size} exceeds canvas length {length}")
    starts = list(range(0, length - patch_size + 1, stride))
    final = length - patch_size
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def _shifted_patch_starts(
    length: int, patch_size: int, stride: int, offset: int
) -> Tuple[int, ...]:
    """Shift interior starts without wrapping and retain both covering edges."""

    final = length - patch_size
    if final < 0:
        raise ValueError(f"Patch size {patch_size} exceeds canvas length {length}")
    if final == 0:
        return (0,)
    offset %= stride
    if offset == 0:
        return _patch_starts(length, patch_size, stride)
    return tuple(sorted(set((0, *range(offset, final, stride), final))))


def build_planar_patch_layout(
    canvas_height: int,
    canvas_width: int,
    patch_size: int,
    stride: int,
    *,
    offset: int = 0,
) -> PlanarPatchLayout:
    starts = _patch_starts if offset == 0 else _shifted_patch_starts
    if offset == 0:
        ys = starts(canvas_height, patch_size, stride)
        xs = starts(canvas_width, patch_size, stride)
    else:
        ys = starts(canvas_height, patch_size, stride, offset)
        xs = starts(canvas_width, patch_size, stride, offset)
    positions = tuple((y, x) for y in ys for x in xs)
    patches = tuple(
        PlanarPatch(index=index, y=y, x=x, size=patch_size)
        for index, (y, x) in enumerate(positions)
    )
    return PlanarPatchLayout(canvas_height, canvas_width, patch_size, patches)


def build_planar_patch_layout_for_step(
    config: PlanarConfig, step_index: int
) -> PlanarPatchLayout:
    if step_index < 0:
        raise ValueError("step_index must be nonnegative")
    offset = 0
    if config.patch_strategy in {"shifted", "dynamic"}:
        offset = step_index * config.dynamic_step_size
    return build_planar_patch_layout(
        config.height,
        config.width,
        config.patch_size,
        config.stride,
        offset=offset,
    )


def extract_planar_patch(canvas: torch.Tensor, patch: PlanarPatch) -> torch.Tensor:
    """Return the exact tensor slice for one patch; no sampling operation is used."""

    if canvas.ndim != 4:
        raise ValueError("planar canvas must have shape [B,C,H,W]")
    if patch.y < 0 or patch.x < 0:
        raise ValueError("patch origin must be nonnegative")
    result = canvas[
        ...,
        patch.y : patch.y + patch.size,
        patch.x : patch.x + patch.size,
    ]
    if result.shape[-2:] != (patch.size, patch.size):
        raise ValueError("patch lies outside the planar canvas")
    return result


def extract_planar_patches(
    canvas: torch.Tensor, layout: PlanarPatchLayout
) -> torch.Tensor:
    if canvas.shape[-2:] != (layout.canvas_height, layout.canvas_width):
        raise ValueError("canvas dimensions do not match planar layout")
    return torch.cat(
        [extract_planar_patch(canvas, patch) for patch in layout.patches], dim=0
    )


def planar_prompt_indices(
    layout: PlanarPatchLayout,
    patches: Sequence[PlanarPatch],
    assignment: str,
) -> Tuple[int, ...]:
    """Map patch centers to cached prompt slots without any camera geometry.

    The legacy layout uses five north-to-south row bands and four left-to-right
    columns.  It is a conditioning layout only; it has no spherical semantics.
    """

    if assignment == "global":
        return tuple(8 for _ in patches)
    if assignment != "legacy_directional":
        raise ValueError(f"Unsupported planar prompt assignment {assignment!r}")
    indices = []
    for patch in patches:
        center_y = patch.y + patch.size / 2
        center_x = patch.x + patch.size / 2
        row = min(4, int(5 * center_y / layout.canvas_height))
        column = min(3, int(4 * center_x / layout.canvas_width))
        indices.append(row * 4 + column)
    return tuple(indices)


class PlanarFusionAccumulator:
    """Stream exact patch placements directly into their canvas rectangles."""

    def __init__(self, previous_canvas: torch.Tensor, config: FusionConfig):
        if previous_canvas.ndim != 4 or previous_canvas.shape[1] != 3:
            raise ValueError("previous_canvas must have shape [B,3,H,W]")
        self.previous = previous_canvas.to(dtype=torch.float32)
        self.config = config
        batch, _, height, width = self.previous.shape
        self.ordinary_num = torch.zeros_like(self.previous)
        self.ordinary_den = torch.zeros(
            batch, 1, height, width, device=self.previous.device, dtype=torch.float32
        )
        self.contributor_count = torch.zeros_like(self.ordinary_den)
        self.detail_num: Optional[torch.Tensor] = None
        self.detail_den: Optional[torch.Tensor] = None
        if config.mode == "detail_preserving_average":
            self.detail_num = torch.zeros_like(self.previous)
            self.detail_den = torch.zeros_like(self.previous)
        self._weight_cache = {}

    def _weight(self, size: int) -> torch.Tensor:
        mode = "uniform" if self.config.mode == "average" else self.config.weight_mode
        key = (size, mode)
        if key not in self._weight_cache:
            self._weight_cache[key] = create_view_weight_map(
                size,
                size,
                mode,
                temperature=self.config.spherediff_temperature,
                device=self.previous.device,
            )
        return self._weight_cache[key]

    def accumulate(self, patch_rgb: torch.Tensor, patch: PlanarPatch) -> None:
        rgb = patch_rgb.to(device=self.previous.device, dtype=torch.float32)
        expected = (self.previous.shape[0], 3, patch.size, patch.size)
        if rgb.shape != expected:
            raise ValueError(
                f"planar patch proposal has shape {tuple(rgb.shape)}, expected {expected}"
            )
        region = (
            slice(None),
            slice(None),
            slice(patch.y, patch.y + patch.size),
            slice(patch.x, patch.x + patch.size),
        )
        weight = self._weight(patch.size)
        self.ordinary_num[region].add_(rgb * weight)
        denominator_region = (
            slice(None),
            slice(None),
            slice(patch.y, patch.y + patch.size),
            slice(patch.x, patch.x + patch.size),
        )
        self.ordinary_den[denominator_region].add_(weight)
        self.contributor_count[denominator_region].add_(1.0)
        if self.detail_num is not None and self.detail_den is not None:
            detail_weight = weight * (
                rgb.abs() + self.config.epsilon
            ).pow(self.config.power)
            self.detail_num[region].add_(rgb * detail_weight)
            self.detail_den[region].add_(detail_weight)

    def finalize(self) -> PlanarFusionResult:
        covered = self.contributor_count > 0
        floor = torch.finfo(torch.float32).tiny
        ordinary = self.ordinary_num / self.ordinary_den.clamp_min(floor)
        if self.detail_num is not None and self.detail_den is not None:
            detail = self.detail_num / self.detail_den.clamp_min(floor)
            fused = ordinary + self.config.alpha * (detail - ordinary)
        else:
            fused = ordinary
        if self.config.uncovered_mode != "keep_previous":
            raise ValueError(f"Unsupported uncovered mode {self.config.uncovered_mode!r}")
        canvas = torch.where(covered, fused, self.previous)
        return PlanarFusionResult(
            canvas, self.ordinary_den, self.contributor_count, covered
        )


def fuse_planar_patches(
    previous_canvas: torch.Tensor,
    layout: PlanarPatchLayout,
    patches: torch.Tensor,
    config: FusionConfig,
) -> PlanarFusionResult:
    if patches.ndim != 4:
        raise ValueError("patches must have shape [N*B,3,P,P]")
    batch_size = previous_canvas.shape[0]
    if patches.shape[0] != layout.num_patches * batch_size:
        raise ValueError("patch count does not match planar layout and canvas batch")
    accumulator = PlanarFusionAccumulator(previous_canvas, config)
    for patch, value in zip(
        layout.patches, patches.split(batch_size, dim=0)
    ):
        accumulator.accumulate(value, patch)
    return accumulator.finalize()
