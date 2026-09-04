#!/usr/bin/env python3
"""Validate clean-prediction consensus through one real ERP camera."""

import argparse
import json
from pathlib import Path

import torch

from diffpano.camera import CameraSampler, camera_for_direction
from diffpano.config import load_experiment_config
from diffpano.diagnostics import tensor_to_pil
from diffpano.erp_x0_pipeline import ERPX0ConsensusPipeline
from diffpano.initialization import load_directional_prompts, set_random_seed
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


@torch.no_grad()
def run_test(config_path: str, output_path: str) -> dict:
    config = load_experiment_config(config_path)
    if config.global_pipeline.mode != "erp_x0_consensus":
        raise ValueError("This test requires global_pipeline.mode=erp_x0_consensus")
    set_random_seed(config.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    backend = build_view_denoiser(config)
    _configure_denoiser(config, backend)
    backend.prepare(
        num_steps=config.generation.num_inference_steps,
        view_height=config.view.height,
        view_width=config.view.width,
    )
    prompts = load_directional_prompts(config.prompt.path)
    negative = ""
    if config.prompt.negative_path:
        negative = Path(config.prompt.negative_path).read_text(encoding="utf-8").strip()
    prepared = backend.prepare_prompt_conditioning(prompts, negative)
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
    result = ERPX0ConsensusPipeline(
        camera_sampler=sampler,
        warp_operator=warp,
        fusion_config=config.fusion,
        consensus_config=config.global_pipeline.clean_consensus,
        backend=backend,
        view_batch_size=1,
        measure_performance=True,
        seed=config.experiment.seed,
    ).run(
        prepared,
        batch_size=config.generation.batch_size,
        erp_height=config.erp.height,
        erp_width=config.erp.width,
    )
    if not torch.isfinite(result.erp_rgb).all():
        raise RuntimeError("One-view ERP output contains non-finite values")
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(result.erp_rgb[0]).save(output / "one_view_erp.png")
    report = {
        "mode": "one_view_erp",
        "backend": config.model.pipeline,
        "steps": len(result.steps),
        "model_evaluations": len(result.steps),
        "fixed_noise_identities": result.fixed_noise_identities,
        "final_step": {
            "coverage_percent": result.steps[-1].coverage_percent,
            "state_statistics": result.steps[-1].state_statistics,
            "timings_seconds": result.steps[-1].timings_seconds,
        },
    }
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        scale = 1024.0**3
        report["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated() / scale
        report["peak_gpu_reserved_gib"] = torch.cuda.max_memory_reserved() / scale
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run_test(args.config, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
