"""EXPERIMENTAL PLANAR ABLATION: direct 2D patch extraction, fusion, and writeback.

This module is intentionally isolated from the spherical pipelines so the no-warp
experiment can be removed by deleting this file and its dedicated pipeline/config.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .pixel_fusion import (
    ExclusiveOwnerMap,
    PixelFusionConfig,
    build_exclusive_owner_map,
    create_patch_weight_map,
    write_back_views_exclusive,
    write_back_views_weighted_average,
)
from .spherical_functions import SphericalFunctions


@dataclass(frozen=True)
class PlanarPatchLayout:
    canvas_height: int
    canvas_width: int
    patch_height: int
    patch_width: int
    positions: Tuple[Tuple[int, int], ...]

    @property
    def num_patches(self) -> int:
        return len(self.positions)


@dataclass
class PlanarPatchFusionConfig:
    random_seed: Optional[int] = None
    patch_latent_height: int = 20
    patch_latent_width: int = 20
    patch_stride_height: int = 10
    patch_stride_width: int = 10

    aggregation_mode: str = "detail_preserving_average"
    weight_mode: str = "distance_to_boundary"
    dpa_alpha: float = 1.0
    dpa_power: float = 1.0
    dpa_eps: float = 1e-6

    reinjection_mode: str = "noise_consistent"
    reinjection_strength: float = 1.0
    latent_writeback_mode: str = "exclusive"

    vae_chunk_size: int = 4
    vae_sample_posterior: bool = False

    def validate(self) -> None:
        if self.random_seed is not None and (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or not 0 <= self.random_seed <= 2**63 - 1
        ):
            raise ValueError("random_seed must be null or an integer from 0 through 2**63 - 1")
        for name in (
            "patch_latent_height",
            "patch_latent_width",
            "patch_stride_height",
            "patch_stride_width",
            "vae_chunk_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.patch_stride_height > self.patch_latent_height:
            raise ValueError("patch_stride_height cannot exceed patch_latent_height because that would leave gaps")
        if self.patch_stride_width > self.patch_latent_width:
            raise ValueError("patch_stride_width cannot exceed patch_latent_width because that would leave gaps")
        if self.aggregation_mode not in {"average", "weighted_average", "detail_preserving_average"}:
            raise ValueError(f"Unsupported aggregation_mode={self.aggregation_mode!r}")
        if self.weight_mode not in {"uniform", "cosine", "gaussian", "distance_to_boundary"}:
            raise ValueError(f"Unsupported weight_mode={self.weight_mode!r}")
        if self.reinjection_mode not in {"noise_consistent", "replace", "weighted_replace", "residual"}:
            raise ValueError(f"Unsupported reinjection_mode={self.reinjection_mode!r}")
        if self.latent_writeback_mode not in {"weighted_average", "exclusive"}:
            raise ValueError(f"Unsupported latent_writeback_mode={self.latent_writeback_mode!r}")
        if self.dpa_eps <= 0:
            raise ValueError("dpa_eps must be positive")
        if self.dpa_power < 0:
            raise ValueError("dpa_power must be nonnegative")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_pixel_fusion_config(self) -> PixelFusionConfig:
        config = PixelFusionConfig(
            pixel_fusion_enabled=True,
            random_seed=self.random_seed,
            aggregation_mode=self.aggregation_mode,
            weight_mode=self.weight_mode,
            dpa_alpha=self.dpa_alpha,
            dpa_power=self.dpa_power,
            dpa_eps=self.dpa_eps,
            reinjection_mode=self.reinjection_mode,
            reinjection_strength=self.reinjection_strength,
            spherical_writeback_mode=self.latent_writeback_mode,
            vae_chunk_size=self.vae_chunk_size,
            vae_sample_posterior=self.vae_sample_posterior,
        )
        config.validate()
        return config

    @classmethod
    def from_any(
        cls,
        value: Optional[Union["PlanarPatchFusionConfig", Dict[str, Any], str]],
    ) -> "PlanarPatchFusionConfig":
        if value is None:
            config = cls()
        elif isinstance(value, cls):
            config = value
        elif isinstance(value, str):
            try:
                from omegaconf import OmegaConf
            except ImportError as exc:
                raise ImportError("OmegaConf is required to load planar patch YAML configs") from exc
            data = OmegaConf.to_container(OmegaConf.load(value), resolve=True)
            if not isinstance(data, dict):
                raise ValueError(f"Planar patch config must contain a mapping: {value}")
            config = cls.from_any(data)
        elif isinstance(value, dict):
            allowed = set(cls.__dataclass_fields__)
            unknown = sorted(set(value) - allowed)
            if unknown:
                raise ValueError(f"Unknown planar patch config fields: {unknown}")
            config = cls(**value)
        else:
            raise TypeError(f"Unsupported planar patch config type: {type(value)!r}")
        config.validate()
        return config


@dataclass
class PlanarBlendResult:
    fused_values: torch.Tensor
    accumulated_weight: torch.Tensor
    contributor_count: torch.Tensor
    valid_output_mask: torch.Tensor


def _patch_starts(length: int, patch_size: int, stride: int) -> Tuple[int, ...]:
    if patch_size > length:
        raise ValueError(f"Patch size {patch_size} exceeds canvas length {length}")
    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def build_planar_patch_layout(
    canvas_height: int,
    canvas_width: int,
    patch_height: int,
    patch_width: int,
    stride_height: int,
    stride_width: int,
) -> PlanarPatchLayout:
    y_starts = _patch_starts(canvas_height, patch_height, stride_height)
    x_starts = _patch_starts(canvas_width, patch_width, stride_width)
    return PlanarPatchLayout(
        canvas_height=canvas_height,
        canvas_width=canvas_width,
        patch_height=patch_height,
        patch_width=patch_width,
        positions=tuple((y, x) for y in y_starts for x in x_starts),
    )


def scale_planar_patch_layout(
    layout: PlanarPatchLayout,
    scale_height: int,
    scale_width: int,
) -> PlanarPatchLayout:
    if scale_height < 1 or scale_width < 1:
        raise ValueError("Planar layout scales must be positive integers")
    return PlanarPatchLayout(
        canvas_height=layout.canvas_height * scale_height,
        canvas_width=layout.canvas_width * scale_width,
        patch_height=layout.patch_height * scale_height,
        patch_width=layout.patch_width * scale_width,
        positions=tuple((y * scale_height, x * scale_width) for y, x in layout.positions),
    )


def extract_planar_patches(canvas: torch.Tensor, layout: PlanarPatchLayout) -> torch.Tensor:
    """Extract exact slices without interpolation; batch size one is intentional for this ablation."""

    if canvas.ndim != 4 or canvas.shape[0] != 1:
        raise ValueError(f"Expected canvas [1,C,H,W], got {tuple(canvas.shape)}")
    if canvas.shape[-2:] != (layout.canvas_height, layout.canvas_width):
        raise ValueError("Canvas dimensions do not match the planar patch layout")
    return torch.cat(
        [
            canvas[..., y:y + layout.patch_height, x:x + layout.patch_width]
            for y, x in layout.positions
        ],
        dim=0,
    )


def blend_planar_patches(
    patches: torch.Tensor,
    layout: PlanarPatchLayout,
    config: PixelFusionConfig,
) -> PlanarBlendResult:
    """Blend exact 2D patch placements with the same weighted/DPA formulas as pixel fusion."""

    if patches.ndim != 4 or patches.shape[0] != layout.num_patches:
        raise ValueError(f"Expected {layout.num_patches} patches [P,C,H,W], got {tuple(patches.shape)}")
    if patches.shape[-2:] != (layout.patch_height, layout.patch_width):
        raise ValueError("Patch dimensions do not match the planar patch layout")
    config.validate()
    patches = patches.to(dtype=torch.float32)
    channels = patches.shape[1]
    ordinary_num = patches.new_zeros(channels, layout.canvas_height, layout.canvas_width)
    ordinary_den = patches.new_zeros(1, layout.canvas_height, layout.canvas_width)
    contributor_count = patches.new_zeros(1, layout.canvas_height, layout.canvas_width)
    detail_num = patches.new_zeros(channels, layout.canvas_height, layout.canvas_width)
    detail_den = patches.new_zeros(channels, layout.canvas_height, layout.canvas_width)

    if config.aggregation_mode == "average":
        patch_weight = patches.new_ones(1, layout.patch_height, layout.patch_width)
    else:
        patch_weight = create_patch_weight_map(
            layout.patch_height,
            layout.patch_width,
            config.weight_mode,
            device=patches.device,
            dtype=patches.dtype,
            eps=config.dpa_eps,
        )[0].clamp_min(config.dpa_eps)

    for patch, (y, x) in zip(patches, layout.positions):
        region = (..., slice(y, y + layout.patch_height), slice(x, x + layout.patch_width))
        ordinary_num[region] += patch * patch_weight
        ordinary_den[region] += patch_weight
        contributor_count[region] += 1
        if config.aggregation_mode == "detail_preserving_average":
            detail_weight = patch_weight * (patch.abs() + config.dpa_eps).pow(config.dpa_power)
            detail_num[region] += patch * detail_weight
            detail_den[region] += detail_weight

    # Direct patch layouts guarantee geometric coverage. Keep that separate from
    # tapered windows, whose border values may be exactly zero before the floor.
    denominator_floor = torch.finfo(patches.dtype).tiny
    valid = contributor_count > 0
    ordinary = ordinary_num / ordinary_den.clamp_min(denominator_floor)
    if config.aggregation_mode == "detail_preserving_average":
        detail = detail_num / detail_den.clamp_min(denominator_floor)
        fused = ordinary + config.dpa_alpha * (detail - ordinary)
    else:
        fused = ordinary
    fused = torch.where(valid, fused, torch.zeros_like(fused))
    return PlanarBlendResult(
        fused_values=fused,
        accumulated_weight=ordinary_den,
        contributor_count=contributor_count,
        valid_output_mask=valid.to(dtype=patches.dtype),
    )


def planar_patch_flat_indices(layout: PlanarPatchLayout, *, device: torch.device) -> List[torch.Tensor]:
    indices = []
    for y, x in layout.positions:
        yy, xx = torch.meshgrid(
            torch.arange(y, y + layout.patch_height, device=device),
            torch.arange(x, x + layout.patch_width, device=device),
            indexing="ij",
        )
        indices.append((yy * layout.canvas_width + xx).reshape(-1))
    return indices


def planar_patch_center_scores(
    layout: PlanarPatchLayout,
    config: PixelFusionConfig,
    *,
    device: torch.device,
) -> List[torch.Tensor]:
    score = create_patch_weight_map(
        layout.patch_height,
        layout.patch_width,
        config.weight_mode,
        device=device,
        dtype=torch.float32,
        eps=config.dpa_eps,
    ).reshape(-1).clamp_min(config.dpa_eps)
    return [score] * layout.num_patches


def build_planar_owner_map(
    layout: PlanarPatchLayout,
    config: PixelFusionConfig,
    *,
    device: torch.device,
) -> ExclusiveOwnerMap:
    return build_exclusive_owner_map(
        layout.canvas_height * layout.canvas_width,
        planar_patch_flat_indices(layout, device=device),
        planar_patch_center_scores(layout, config, device=device),
        list(range(layout.num_patches)),
        device=device,
    )


def write_back_planar_latents(
    latent_template: torch.Tensor,
    corrected_patches: torch.Tensor,
    layout: PlanarPatchLayout,
    config: PixelFusionConfig,
    *,
    mode: str,
    owner_map: Optional[ExclusiveOwnerMap] = None,
) -> torch.Tensor:
    if latent_template.shape[0] != 1 or latent_template.shape[-2:] != (
        layout.canvas_height,
        layout.canvas_width,
    ):
        raise ValueError("Latent template does not match planar layout")
    if corrected_patches.shape[0] != layout.num_patches:
        raise ValueError("Corrected patch count does not match planar layout")
    flat_template = latent_template.reshape(1, latent_template.shape[1], 1, -1)
    flat_patches = [patch.unsqueeze(0).reshape(1, patch.shape[0], 1, -1) for patch in corrected_patches]
    indices = planar_patch_flat_indices(layout, device=latent_template.device)
    if mode == "exclusive":
        if owner_map is None:
            owner_map = build_planar_owner_map(layout, config, device=latent_template.device)
        result = write_back_views_exclusive(
            flat_template,
            flat_patches,
            indices,
            list(range(layout.num_patches)),
            owner_map,
            uncovered_mode="error",
            geometry_summary=f"planar layout {layout}",
        ).latents
    elif mode == "weighted_average":
        result = write_back_views_weighted_average(
            flat_template,
            flat_patches,
            indices,
            planar_patch_center_scores(layout, config, device=latent_template.device),
        )
    else:
        raise ValueError(f"Unsupported planar latent writeback mode {mode!r}")
    return result.reshape_as(latent_template)


def planar_patch_prompt_indices(
    layout: PlanarPatchLayout,
    prompt_directions: torch.Tensor,
) -> torch.Tensor:
    """Assign prompts exactly as SphereDiff does, using each planar patch center as an ERP-like direction."""

    centers = []
    for y, x in layout.positions:
        u = (x + layout.patch_width / 2) / layout.canvas_width
        v = (y + layout.patch_height / 2) / layout.canvas_height
        theta = (2 * u - 1) * torch.pi
        phi = (v - 0.5) * torch.pi
        centers.append((theta, phi))
    theta = prompt_directions.new_tensor([item[0] for item in centers])
    phi = prompt_directions.new_tensor([item[1] for item in centers])
    patch_directions = SphericalFunctions.spherical_to_cartesian(theta, phi)
    cosine_similarity = torch.einsum("ni,ki->nk", patch_directions, prompt_directions)
    return cosine_similarity.argmax(dim=-1)
