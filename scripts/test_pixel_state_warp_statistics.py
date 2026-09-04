#!/usr/bin/env python3
"""Measure projection/fusion drift on Gaussian ERP state with an identity local model."""

import argparse
import json
from pathlib import Path

import torch

from diffpano.camera import SphereDiffFixedCameraSampler
from diffpano.config import FusionConfig, SamplingDirectionConfig, ViewConfig, WarpConfig
from diffpano.erp_pipeline import ERPRGBPipeline
from diffpano.pipelines.base import MockViewDenoiser
from diffpano.warp import StandardWarpOperator


class IdentityPixelDenoiser(MockViewDenoiser):
    def __init__(self, num_steps: int):
        super().__init__(num_steps=num_steps)
        self.state_diagnostics_enabled = True
        self.last_model_prediction = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--erp-height", type=int, default=64)
    parser.add_argument("--erp-width", type=int, default=128)
    parser.add_argument("--view-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--erp-to-view", choices=("nearest", "bilinear"), default="nearest")
    parser.add_argument("--view-to-erp", choices=("nearest", "bilinear"), default="bilinear")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    initial = torch.randn(1, 3, args.erp_height, args.erp_width)
    view = ViewConfig(height=args.view_size, width=args.view_size)
    fusion = FusionConfig(mode="average", weight_mode="uniform")
    warp = StandardWarpOperator(
        WarpConfig(
            mode="standard",
            erp_to_perspective=SamplingDirectionConfig(args.erp_to_view),
            perspective_to_erp=SamplingDirectionConfig(args.view_to_erp),
        ),
        fusion,
    )
    denoiser = IdentityPixelDenoiser(args.steps)
    pipeline = ERPRGBPipeline(
        camera_sampler=SphereDiffFixedCameraSampler(view),
        warp_operator=warp,
        fusion_config=fusion,
        view_denoiser=denoiser,
        view_batch_size=1,
    )
    result = pipeline.run(initial, prepared_conditioning=None)
    report = [
        {"step": step.step_index, **step.state_statistics}
        for step in result.steps
    ]
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
