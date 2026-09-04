#!/usr/bin/env python3
"""Run PixelDiT through one ERP perspective camera before multi-camera experiments."""

import argparse
import json
from pathlib import Path

import torch

from diffpano.camera import CameraSampler, camera_for_direction
from diffpano.config import load_experiment_config
from diffpano.diagnostics import tensor_to_pil
from diffpano.erp_pipeline import ERPRGBPipeline
from diffpano.fusion import FusionConfig
from diffpano.initialization import initialize_erp_canvas, load_directional_prompts, set_random_seed
from diffpano.pipelines import build_view_denoiser
from diffpano.projection import ProjectionCache
from diffpano.warp import build_warp_operator
from scripts.generate import _configure_denoiser


class OneCameraSampler(CameraSampler):
    def __init__(self, camera):
        self.camera = camera

    def sample(self, step_index, num_steps):
        del step_index, num_steps
        return [self.camera]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pixeldit_standard_average.yaml")
    parser.add_argument("--output", default="outputs/pixeldit-one-view-erp")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
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
    camera = camera_for_direction(
        0,
        0,
        height=config.view.height,
        width=config.view.width,
        fov_x=config.view.fov_x,
        fov_y=config.view.fov_y,
    )
    sampler = OneCameraSampler(camera)
    warp = build_warp_operator(
        config.warp,
        config.fusion,
        ProjectionCache(
            max_entries=config.performance.projection_cache_max_entries,
            cpu_fallback=config.performance.projection_cache_cpu_fallback,
        ),
    )
    prompts = load_directional_prompts(config.prompt.path)
    bank = denoiser.prepare_prompt_conditioning(prompts)
    generator = torch.Generator(device=denoiser.device).manual_seed(config.experiment.seed)
    initial = initialize_erp_canvas(
        config.initialization,
        batch_size=1,
        height=config.erp.height,
        width=config.erp.width,
        device=denoiser.device,
        generator=generator,
        camera_sampler=sampler,
        warp_operator=warp,
        fusion_config=FusionConfig(mode="average", weight_mode="uniform"),
        view_denoiser=denoiser,
    )
    pipeline = ERPRGBPipeline(
        camera_sampler=sampler,
        warp_operator=warp,
        fusion_config=config.fusion,
        view_denoiser=denoiser,
        view_batch_size=1,
    )
    result = pipeline.run(initial, bank)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(result.erp_rgb[0]).save(output / "one_view_erp.png")
    report = {
        "steps": len(result.steps),
        "camera_yaw_degrees": 0,
        "camera_pitch_degrees": 0,
        "final_state_statistics": result.steps[-1].state_statistics,
    }
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        scale = 1024.0 ** 3
        report["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated() / scale
        report["peak_gpu_reserved_gib"] = torch.cuda.max_memory_reserved() / scale
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
