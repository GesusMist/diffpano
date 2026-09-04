"""Model-independent standard and Laplacian-pyramid RGB warp operators."""

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from diffpano.camera import PerspectiveCamera
from diffpano.config import FusionConfig, WarpConfig
from diffpano.fusion import FusionResult, RGBFusionAccumulator, create_view_weight_map
from diffpano.lpw import (
    build_laplacian_pyramid,
    lod_level_confidence,
    reconstruct_laplacian_pyramid,
    reconstruct_masked_laplacian_pyramid,
)
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

    def create_fusion_accumulator(
        self, previous_erp: torch.Tensor
    ) -> Optional["LaplacianPyramidFusionAccumulator"]:
        """Return a dedicated multi-view accumulator when the warp requires one."""

        return None


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
    """RGB LPW whose inverse direction is fused jointly per pyramid level."""

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
        del rgb_view, camera, erp_size
        raise RuntimeError(
            "LPW perspective-to-ERP requires joint multi-view fusion; "
            "use create_fusion_accumulator()"
        )

    def create_fusion_accumulator(
        self, previous_erp: torch.Tensor
    ) -> "LaplacianPyramidFusionAccumulator":
        return LaplacianPyramidFusionAccumulator(self, previous_erp)


class LaplacianPyramidFusionAccumulator:
    """Stream perspective RGB into jointly fused ERP Laplacian levels.

    Only per-level running numerators/denominators are retained.  Individual
    perspective views and their projected ERP tensors are discarded after
    ``accumulate`` returns.  Level zero supplies the full-resolution weight and
    contributor diagnostics exposed through :class:`FusionResult`.
    """

    def __init__(
        self,
        operator: LaplacianPyramidWarpOperator,
        previous_erp: torch.Tensor,
    ):
        if previous_erp.ndim != 4 or previous_erp.shape[1] != 3:
            raise ValueError("previous_erp must have shape [B,3,H,W]")
        self.operator = operator
        self.previous = previous_erp.to(dtype=torch.float32)
        self.erp_size = tuple(self.previous.shape[-2:])
        self.level_accumulators: List[RGBFusionAccumulator] = []
        self.level_shapes: List[Tuple[int, int]] = []

    def _initialize_levels(self, count: int) -> None:
        self.level_shapes = [
            (_level_size(self.erp_size[0], level), _level_size(self.erp_size[1], level))
            for level in range(count)
        ]
        batch = self.previous.shape[0]
        self.level_accumulators = [
            RGBFusionAccumulator(
                torch.zeros(
                    batch,
                    3,
                    height,
                    width,
                    device=self.previous.device,
                    dtype=torch.float32,
                ),
                self.operator.fusion_config,
            )
            for height, width in self.level_shapes
        ]

    def accumulate(
        self,
        rgb_view: torch.Tensor,
        camera: PerspectiveCamera,
    ) -> ERPContribution:
        """Project one view's coefficients and update every ERP level."""

        if rgb_view.shape[0] != self.previous.shape[0]:
            raise ValueError("LPW view batch size does not match the persistent ERP")
        view_pyramid = build_laplacian_pyramid(
            rgb_view.to(device=self.previous.device, dtype=torch.float32),
            self.operator.warp_config.lpw.levels,
            vertical_padding_mode=self.operator.warp_config.lpw.vertical_padding_mode,
            spherical_erp=False,
        )
        if not self.level_accumulators:
            self._initialize_levels(len(view_pyramid))
        elif len(view_pyramid) != len(self.level_accumulators):
            raise ValueError("all LPW views must produce the same number of pyramid levels")

        use_lod = self.operator.warp_config.lpw.lod_mode == "jacobian"
        if use_lod:
            full_lod = projection_lod_map(
                camera,
                self.erp_size[0],
                self.erp_size[1],
                device=self.previous.device,
                cache=self.operator.cache,
            )
        else:
            full_lod = torch.zeros(
                1,
                1,
                self.erp_size[0],
                self.erp_size[1],
                device=self.previous.device,
                dtype=torch.float32,
            )

        for level, (coefficients, level_erp, accumulator) in enumerate(
            zip(view_pyramid, self.level_shapes, self.level_accumulators)
        ):
            level_camera = replace(
                camera,
                height=coefficients.shape[-2],
                width=coefficients.shape[-1],
            )
            contribution = perspective_to_erp(
                coefficients,
                level_camera,
                level_erp[0],
                level_erp[1],
                interpolation=self.operator.warp_config.perspective_to_erp.interpolation,
                weight_map=self.operator._weight_map(level_camera, self.previous.device),
                cache=self.operator.cache,
            )
            if not use_lod:
                # lod_mode=none retains every band; it does not mean LOD zero,
                # which would incorrectly select only band zero.
                confidence = torch.ones_like(contribution.valid_mask[:1])
            else:
                level_lod = F.interpolate(
                    full_lod,
                    size=level_erp,
                    mode="bilinear",
                    align_corners=False,
                )
                confidence = lod_level_confidence(
                    level_lod,
                    level,
                    len(view_pyramid),
                    self.operator.warp_config.lpw.lod_interpolation,
                )
            accumulator.accumulate(contribution, confidence=confidence)

        # This standard RGB projection is diagnostic-only.  LPW fusion above
        # consumes pyramid coefficients, never this reconstructed/full RGB view.
        diagnostic = perspective_to_erp(
            rgb_view.to(device=self.previous.device, dtype=torch.float32),
            camera,
            self.erp_size[0],
            self.erp_size[1],
            interpolation=self.operator.warp_config.perspective_to_erp.interpolation,
            weight_map=self.operator._weight_map(camera, self.previous.device),
            cache=self.operator.cache,
        )
        diagnostic.lod_map = full_lod
        return diagnostic

    def finalize(self) -> FusionResult:
        if not self.level_accumulators:
            return RGBFusionAccumulator(
                self.previous, self.operator.fusion_config
            ).finalize()
        levels = [accumulator.finalize() for accumulator in self.level_accumulators]
        reconstructed, reconstructed_mask = reconstruct_masked_laplacian_pyramid(
            [level.erp_rgb for level in levels],
            [level.coverage_mask for level in levels],
            self.operator.fusion_config.epsilon,
        )
        covered = reconstructed_mask > self.operator.fusion_config.epsilon
        fused = torch.where(covered, reconstructed, self.previous)
        finest = levels[0]
        return FusionResult(
            erp_rgb=fused,
            accumulated_weight=finest.accumulated_weight,
            contributor_count=finest.contributor_count,
            coverage_mask=covered,
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
