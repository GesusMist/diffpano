#!/usr/bin/env python3
"""Compare normal first-order sampling and x0-renoising on exactly one patch."""

import argparse
import hashlib
import json
import os
import platform
import time

import torch
from omegaconf import OmegaConf

from diffpano.config import load_experiment_config
from diffpano.diagnostics import tensor_to_pil
from diffpano.initialization import set_random_seed, load_directional_prompts
from diffpano.planar_pipeline import _negative_prompt
from diffpano.native_multidiffusion import prepare_native_backend
from diffpano.pipelines import build_view_denoiser
from diffpano.trajectory import compare_single_patch
from scripts.generate import _configure_denoiser, _run_directory


def run(config):
    config.validate()
    set_random_seed(config.experiment.seed)
    destination = _run_directory(config)
    destination.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(config.to_dict()), destination / "config.yaml")
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    backend = build_view_denoiser(config)
    _configure_denoiser(config, backend)
    prepared = prepare_native_backend(config, backend)
    conditioning = backend.conditioning_for_prompt_indices(prepared, [8], batch_size=config.generation.batch_size)
    p = config.native_multidiffusion.patch_size
    generator = torch.Generator(device="cpu").manual_seed(config.experiment.seed)
    epsilon = torch.randn(config.generation.batch_size, backend.native_channels, p, p, generator=generator).to(backend.device)
    result = compare_single_patch(backend, conditioning, epsilon)
    tensor_to_pil(result.native_final[0]).save(destination / "native_final.png")
    tensor_to_pil(result.x0_renoise_final[0]).save(destination / "x0_renoise_final.png")
    epsilon_cpu = result.initial_epsilon.cpu().contiguous()
    torch.save(epsilon_cpu, destination / "initial_epsilon.pt")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    scheduler = getattr(getattr(backend, "pipeline", None), "scheduler", None)
    metadata = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "fairness": {
            "same_backend_instance": True, "same_conditioning_object": True,
            "same_initialized_state_bit_exact": True, "schedule_unchanged": True,
            "epsilon_unchanged": True, "same_guidance": True,
            "same_local_resolution": True,
            "equal_guided_prediction_counts": result.model_evaluations["native"] == result.model_evaluations["x0_renoise"] == len(backend.timesteps),
        },
        "prompt": load_directional_prompts(config.prompt.path)[2],
        "negative_prompt": _negative_prompt(config) or getattr(backend, "negative_prompt", ""),
        "scheduler_class": type(scheduler).__name__ if scheduler is not None else "PixelDiT official first-order DPM",
        "scheduler_timesteps": backend.timesteps.detach().cpu().tolist(),
        "scheduler_sigmas": scheduler.sigmas.detach().cpu().tolist() if scheduler is not None and hasattr(scheduler, "sigmas") else None,
        "flow_shift": getattr(getattr(backend, "solver", None), "flow_shift", None),
        "pixeldit_flow_schedule": backend.solver.schedule.detach().cpu().tolist() if hasattr(backend, "solver") else None,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None,
        "steps": result.steps, "model_evaluations": result.model_evaluations,
        "evaluation_unit": "guided prediction (CFG forwards identical for both trajectories)",
        "same_initial_epsilon": True,
        "epsilon_sha256": hashlib.sha256(epsilon_cpu.numpy().tobytes()).hexdigest(),
        "native_shape": list(epsilon.shape), "spatial_factor": backend.native_spatial_factor,
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None,
        "scheduler_shift_mu": getattr(backend, "scheduler_shift_mu", None),
        "config": config.to_dict(),
    }
    (destination / "trajectory.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Trajectory saved to {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_experiment_config(args.config))


if __name__ == "__main__":
    main()
