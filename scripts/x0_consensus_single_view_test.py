#!/usr/bin/env python3
"""Validate clean-prediction recurrence for one real view without ERP warping."""

import argparse
import json
from pathlib import Path

import torch

from diffpano.camera import camera_for_direction
from diffpano.config import load_experiment_config
from diffpano.diagnostics import tensor_state_statistics, tensor_to_pil
from diffpano.initialization import load_directional_prompts, set_random_seed
from diffpano.noise import FixedPatchNoiseBank
from diffpano.pipelines import build_view_denoiser
from scripts.generate import _configure_denoiser


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
    conditioning = backend.conditioning_for_cameras(
        prepared, [camera], batch_size=config.generation.batch_size
    )
    noise_bank = FixedPatchNoiseBank(
        config.global_pipeline.clean_consensus,
        backend=backend,
        num_cameras=1,
        batch_size=config.generation.batch_size,
        height=config.view.height,
        width=config.view.width,
        seed=config.experiment.seed,
    )
    first_noise = noise_bank.get([0], device=backend.device)
    clean_rgb = None
    step_statistics = []
    for step_index, timestep in enumerate(backend.timesteps):
        fixed_noise = noise_bank.get([0], device=backend.device)
        if not torch.equal(first_noise, fixed_noise):
            raise RuntimeError("Fixed noise changed during the single-view recurrence")
        if clean_rgb is None:
            noisy = backend.make_initial_noisy_state(fixed_noise, timestep)
        else:
            clean_native = backend.encode_clean(clean_rgb)
            noisy = backend.add_fixed_noise(clean_native, fixed_noise, timestep)
        predicted_clean = backend.predict_clean_native(noisy, timestep, conditioning)
        clean_rgb = backend.decode_clean(predicted_clean).float()
        if not torch.isfinite(clean_rgb).all():
            raise RuntimeError(f"Non-finite clean prediction at step {step_index}")
        step_statistics.append(
            {
                "step_index": step_index,
                "scheduler_timestep": float(torch.as_tensor(timestep).cpu()),
                **tensor_state_statistics(clean_rgb, "predicted_clean_rgb"),
            }
        )

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(clean_rgb[0]).save(output / "single_view_predicted_clean.png")
    report = {
        "mode": "single_view_no_panorama",
        "backend": config.model.pipeline,
        "seed": config.experiment.seed,
        "steps": len(backend.timesteps),
        "model_evaluations": len(backend.timesteps),
        "fixed_noise_identity": noise_bank.identity(0),
        "step_statistics": step_statistics,
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
