"""Model-independent standard and Laplacian-pyramid RGB warp operators."""

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from diffpano.camera import PerspectiveCamera
from diffpano.config import FusionConfig, WarpConfig
from diffpano.fusion import create_view_weight_map
from diffpano.lpw import build_laplacian_pyramid, lod_level_confidence, reconstruct_laplacian_pyramid
from diffpano.projection import (
    ERPContribution,
    ProjectionCache,
    erp_to_perspective,
    perspective_to_erp,
    projection_lod_map,
)


class WarpOperator(ABC):
    @abstractmethod
    def erp_to_perspective(self, erp_rgb: torch.Tensor, camera: PerspectiveCamera) -> torch.Tensor:
        ...

    @abstractmethod
    def perspective_to_erp(
        self, rgb_view: torch.Tensor, camera: PerspectiveCamera, erp_size: Tuple[int, int]
    ) -> ERPContribution:
        ...


class StandardWarpOperator(WarpOperator):
    def __init__(
        self,
        warp_config: WarpConfig,
        fusion_config: FusionConfig,
        cache: ProjectionCache = None,
    ):
        self.warp_config = warp_config
        self.fusion_config = fusion_config
        self.cache = cache or ProjectionCache()
        self._weight_cache: Dict[Tuple[object, ...], torch.Tensor] = {}

    def _weight_map(self, camera: PerspectiveCamera, device: torch.device) -> torch.Tensor:
        key = (
            self.fusion_config.weight_mode,
            self.fusion_config.spherediff_temperature,
            camera.height,
            camera.width,
            str(device),
        )
        if key not in self._weight_cache:
            self._weight_cache[key] = create_view_weight_map(
                camera.height,
                camera.width,
                self.fusion_config.weight_mode,
                temperature=self.fusion_config.spherediff_temperature,
                device=device,
            )
        return self._weight_cache[key]

    def erp_to_perspective(self, erp_rgb: torch.Tensor, camera: PerspectiveCamera) -> torch.Tensor:
        return erp_to_perspective(
            erp_rgb,
            camera,
            interpolation=self.warp_config.erp_to_perspective.interpolation,
            cache=self.cache,
            vertical_padding_mode=self.warp_config.lpw.vertical_padding_mode,
        )

    def perspective_to_erp(
        self, rgb_view: torch.Tensor, camera: PerspectiveCamera, erp_size: Tuple[int, int]
    ) -> ERPContribution:
        return perspective_to_erp(
            rgb_view,
            camera,
            erp_size[0],
            erp_size[1],
            interpolation=self.warp_config.perspective_to_erp.interpolation,
            weight_map=self._weight_map(camera, rgb_view.device),
            cache=self.cache,
        )


def _level_size(value: int, level: int) -> int:
    return max(1, int(round(value / (2**level))))


def _level_camera(camera: PerspectiveCamera, level: int) -> PerspectiveCamera:
    return replace(
        camera,
        height=_level_size(camera.height, level),
        width=_level_size(camera.width, level),
    )


class LaplacianPyramidWarpOperator(StandardWarpOperator):
    """RGB LPW with Jacobian LOD independent of scalar overlap weighting."""

    def erp_to_perspective(self, erp_rgb: torch.Tensor, camera: PerspectiveCamera) -> torch.Tensor:
        pyramid = build_laplacian_pyramid(
            erp_rgb.to(dtype=torch.float32),
            self.warp_config.lpw.levels,
            vertical_padding_mode=self.warp_config.lpw.vertical_padding_mode,
            spherical_erp=True,
        )
        view_levels = []
        for level, coefficients in enumerate(pyramid):
            view_levels.append(
                erp_to_perspective(
                    coefficients,
                    _level_camera(camera, level),
                    interpolation=self.warp_config.erp_to_perspective.interpolation,
                    cache=self.cache,
                    vertical_padding_mode=self.warp_config.lpw.vertical_padding_mode,
                )
            )
        return reconstruct_laplacian_pyramid(view_levels)

    def perspective_to_erp(
        self, rgb_view: torch.Tensor, camera: PerspectiveCamera, erp_size: Tuple[int, int]
    ) -> ERPContribution:
        view_pyramid = build_laplacian_pyramid(
            rgb_view.to(dtype=torch.float32),
            self.warp_config.lpw.levels,
            vertical_padding_mode=self.warp_config.lpw.vertical_padding_mode,
            spherical_erp=False,
        )
        if self.warp_config.lpw.lod_mode == "none":
            full_lod = torch.zeros(
                1, 1, erp_size[0], erp_size[1], device=rgb_view.device, dtype=torch.float32
            )
        else:
            full_lod = projection_lod_map(
                camera, erp_size[0], erp_size[1], device=rgb_view.device, cache=self.cache
            )
        erp_levels = []
        base: ERPContribution = None
        for level, coefficients in enumerate(view_pyramid):
            level_camera = _level_camera(camera, level)
            level_erp = (_level_size(erp_size[0], level), _level_size(erp_size[1], level))
            contribution = perspective_to_erp(
                coefficients,
                level_camera,
                level_erp[0],
                level_erp[1],
                interpolation=self.warp_config.perspective_to_erp.interpolation,
                weight_map=self._weight_map(level_camera, rgb_view.device),
                cache=self.cache,
            )
            lod = F.interpolate(full_lod, size=level_erp, mode="bilinear", align_corners=False)
            confidence = lod_level_confidence(
                lod, level, len(view_pyramid), self.warp_config.lpw.lod_interpolation
            )
            erp_levels.append(contribution.rgb * confidence)
            if level == 0:
                base = contribution
        reconstructed = reconstruct_laplacian_pyramid(erp_levels)
        return ERPContribution(
            rgb=reconstructed * base.valid_mask,
            valid_mask=base.valid_mask,
            weight=base.weight,
            lod_map=full_lod,
        )


def build_warp_operator(
    warp_config: WarpConfig,
    fusion_config: FusionConfig,
    cache: ProjectionCache = None,
) -> WarpOperator:
    if warp_config.mode == "standard":
        return StandardWarpOperator(warp_config, fusion_config, cache)
    if warp_config.mode == "lpw":
        return LaplacianPyramidWarpOperator(warp_config, fusion_config, cache)
    raise ValueError(f"Unsupported warp mode {warp_config.mode!r}")
