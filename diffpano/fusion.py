"""Streaming RGB-space fusion for synchronous ERP contributions."""

import math
from dataclasses import dataclass
from typing import Optional

import torch

from diffpano.config import FusionConfig
from diffpano.projection import ERPContribution


def create_view_weight_map(
    height: int,
    width: int,
    mode: str,
    *,
    temperature: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Return an FP32 scalar confidence map in perspective coordinates."""

    y = 2 * (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height - 1
    x = 2 * (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    if mode == "uniform":
        weight = torch.ones_like(xx)
    elif mode == "cosine":
        weight = torch.cos(xx * math.pi / 2).clamp_min(0) * torch.cos(yy * math.pi / 2).clamp_min(0)
    elif mode == "gaussian":
        weight = torch.exp(-0.5 * (xx.square() + yy.square()) / 0.5**2)
    elif mode == "distance_to_boundary":
        weight = torch.minimum(1 - xx.abs(), 1 - yy.abs()).clamp_min(0)
        weight = weight / weight.max().clamp_min(1.0e-12)
    elif mode == "spherediff_center":
        weight = torch.exp(-torch.sqrt(xx.square() + yy.square()) / temperature)
    else:
        raise ValueError(f"Unsupported weight mode {mode!r}")
    return weight.unsqueeze(0).unsqueeze(0)


@dataclass
class FusionResult:
    erp_rgb: torch.Tensor
    accumulated_weight: torch.Tensor
    contributor_count: torch.Tensor
    coverage_mask: torch.Tensor


class RGBFusionAccumulator:
    """Accumulate one full-ERP contribution at a time without storing N views."""

    def __init__(self, previous_erp: torch.Tensor, config: FusionConfig):
        if previous_erp.ndim != 4 or previous_erp.shape[1] != 3:
            raise ValueError("previous_erp must have shape [B,3,H,W]")
        self.previous = previous_erp.to(dtype=torch.float32)
        self.config = config
        batch, channels, height, width = self.previous.shape
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

    def accumulate(
        self,
        contribution: ERPContribution,
        confidence: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate one contribution with optional multiplicative trust.

        ``confidence`` is deliberately part of the effective fusion weight,
        not the coefficient value.  Standard warp never supplies it and is
        therefore numerically unchanged; LPW uses it for per-level Jacobian
        LOD confidence.
        """

        rgb = contribution.rgb.to(device=self.previous.device, dtype=torch.float32)
        mask = contribution.valid_mask.to(device=self.previous.device, dtype=torch.float32)
        weight = contribution.weight.to(device=self.previous.device, dtype=torch.float32)
        if rgb.shape != self.previous.shape or mask.shape != self.ordinary_den.shape:
            raise ValueError("ERP contribution shape does not match the persistent canvas")
        if self.config.mode == "average":
            weight = torch.ones_like(mask)
        effective = mask * weight
        if confidence is not None:
            confidence = confidence.to(device=self.previous.device, dtype=torch.float32)
            if confidence.shape[0] == 1 and mask.shape[0] != 1:
                confidence = confidence.expand(mask.shape[0], -1, -1, -1)
            if confidence.shape != mask.shape:
                raise ValueError("fusion confidence shape does not match the contribution mask")
            effective = effective * confidence
        self.ordinary_num.add_(rgb * effective)
        self.ordinary_den.add_(effective)
        self.contributor_count.add_((mask > 0).to(torch.float32))
        if self.detail_num is not None and self.detail_den is not None:
            detail_weight = effective * (rgb.abs() + self.config.epsilon).pow(self.config.power)
            self.detail_num.add_(rgb * detail_weight)
            self.detail_den.add_(detail_weight)

    def finalize(self) -> FusionResult:
        covered = self.ordinary_den > self.config.epsilon
        ordinary = self.ordinary_num / self.ordinary_den.clamp_min(self.config.epsilon)
        if self.detail_num is not None and self.detail_den is not None:
            detail = self.detail_num / self.detail_den.clamp_min(self.config.epsilon)
            fused = ordinary + self.config.alpha * (detail - ordinary)
        else:
            fused = ordinary
        if self.config.uncovered_mode != "keep_previous":
            raise ValueError(f"Unsupported uncovered mode {self.config.uncovered_mode!r}")
        next_erp = torch.where(covered, fused, self.previous)
        return FusionResult(next_erp, self.ordinary_den, self.contributor_count, covered)


def fuse_contributions(
    previous_erp: torch.Tensor,
    contributions,
    config: FusionConfig,
) -> FusionResult:
    accumulator = RGBFusionAccumulator(previous_erp, config)
    for contribution in contributions:
        accumulator.accumulate(contribution)
    return accumulator.finalize()


def detail_preserving_average(
    values: torch.Tensor,
    masks: torch.Tensor,
    weights: torch.Tensor,
    *,
    alpha: float,
    power: float,
    epsilon: float,
    dim: int = 0,
) -> torch.Tensor:
    """Vectorized DPA reference used by unit tests and small diagnostics."""

    values, masks, weights = [tensor.to(dtype=torch.float32) for tensor in (values, masks, weights)]
    effective = masks * weights
    ordinary = (values * effective).sum(dim) / effective.sum(dim).clamp_min(epsilon)
    detail_weight = effective * (values.abs() + epsilon).pow(power)
    detail = (values * detail_weight).sum(dim) / detail_weight.sum(dim).clamp_min(epsilon)
    return ordinary + alpha * (detail - ordinary)
