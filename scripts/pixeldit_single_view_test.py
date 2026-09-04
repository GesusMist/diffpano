#!/usr/bin/env python3
"""Validate DiffPano order-one PixelDiT against NVIDIA sampling without ERP warping."""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from diffpano.camera import camera_for_direction
from diffpano.config import load_experiment_config
from diffpano.diagnostics import tensor_to_pil
from diffpano.initialization import load_directional_prompts, set_random_seed
from diffpano.pipelines import build_view_denoiser
from scripts.generate import _configure_denoiser


def _peak_gpu_memory() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    torch.cuda.synchronize()
    scale = 1024.0 ** 3
    return {
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / scale,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / scale,
    }


def run_test(
    config_path: str,
    output_path: str,
    *,
    compare_order_one: bool,
    save_official_order_two: bool = False,
    mode: str = "single_view_no_warp",
) -> dict:
    config = load_experiment_config(config_path)
    if config.model.pipeline != "pixeldit":
        raise ValueError("This test requires model.pipeline=pixeldit")
    set_random_seed(config.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    denoiser = build_view_denoiser(config)
    _configure_denoiser(config, denoiser)
    denoiser.prepare(
        num_steps=config.generation.num_inference_steps,
        view_height=config.view.height,
        view_width=config.view.width,
    )
    prompts = load_directional_prompts(config.prompt.path)
    bank = denoiser.prepare_prompt_conditioning(prompts)
    camera = camera_for_direction(
        0,
        0,
        height=config.view.height,
        width=config.view.width,
        fov_x=config.view.fov_x,
        fov_y=config.view.fov_y,
    )
    conditioning = denoiser.conditioning_for_cameras(bank, [camera], batch_size=1)
    generator = torch.Generator(device=denoiser.device).manual_seed(config.experiment.seed)
    initial = torch.randn(
        1,
        3,
        config.view.height,
        config.view.width,
        device=denoiser.device,
        dtype=torch.float32,
        generator=generator,
    )

    state = initial.clone()
    for timestep in denoiser.timesteps:
        state = denoiser.denoise_step(state, timestep, conditioning)

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(state[0]).save(output / "diffpano_order1.png")
    report = {
        "mode": mode,
        "seed": config.experiment.seed,
        "steps": len(denoiser.timesteps),
        "flow_shift": denoiser.solver.flow_shift,
        "checkpoint": denoiser.checkpoint_path,
        "official_commit": denoiser.official_commit,
    }
    if compare_order_one:
        reference = denoiser.official_order_one_sample(initial.clone(), conditioning)
        tensor_to_pil(reference[0]).save(output / "official_order1.png")
        difference = (state - reference).abs().float()
        report.update(
            maximum_absolute_error=float(difference.max()),
            mean_absolute_error=float(difference.mean()),
            numerically_close=bool(torch.allclose(state, reference, atol=5.0e-3, rtol=5.0e-3)),
        )
        if not report["numerically_close"]:
            raise RuntimeError(
                "DiffPano first-order loop does not match official PixelDiT order-one inference: "
                f"max error {report['maximum_absolute_error']}"
            )
    if save_official_order_two:
        reference_order_two = denoiser.official_sample(initial.clone(), conditioning, order=2)
        tensor_to_pil(reference_order_two[0]).save(output / "official_order2.png")
        report["official_order_two_saved"] = True
    report.update(_peak_gpu_memory())
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pixeldit_standard_average.yaml")
    parser.add_argument("--output", default="outputs/pixeldit-single-view")
    parser.add_argument("--skip-official-comparison", action="store_true")
    parser.add_argument("--save-official-order-two", action="store_true")
    args = parser.parse_args()
    report = run_test(
        args.config,
        args.output,
        compare_order_one=not args.skip_official_comparison,
        save_official_order_two=args.save_official_order_two,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
