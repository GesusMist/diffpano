#!/usr/bin/env python3
"""Generate a panorama with a persistent ERP RGB diffusion state."""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf

from diffpano.config import ExperimentConfig, load_experiment_config
from diffpano.diagnostics import DiagnosticsWriter, tensor_to_pil
from diffpano.erp_pipeline import generate_erp_rgb
from diffpano.erp_x0_pipeline import generate_erp_x0_consensus
from diffpano.initialization import set_random_seed
from diffpano.metadata import save_run_metadata
from diffpano.pipelines import build_view_denoiser, precision_dtype


def _safe_component(value: str) -> str:
    value = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in value)
    return value.strip("-") or "run"


def _run_directory(config: ExperimentConfig) -> Path:
    now = datetime.now()
    group = config.output.group or now.strftime("%Y-%m-%d")
    run_id = config.output.run_id
    if not run_id:
        run_id = f"{now.strftime('%Y%m%d-%H%M%S')}-{os.environ.get('SLURM_JOB_ID', 'local')}"
    return (
        Path(config.output.directory)
        / _safe_component(group)
        / _safe_component(config.experiment.name)
        / _safe_component(run_id)
    )


def _configure_denoiser(config: ExperimentConfig, denoiser) -> None:
    if config.model.pipeline == "pixeldit":
        if config.model.vae_slicing or config.model.vae_tiling:
            raise ValueError("PixelDiT does not support autoencoder slicing or tiling")
        if config.model.cpu_offload:
            denoiser.enable_model_cpu_offload()
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required when model.cpu_offload=false")
            denoiser.to(
                torch.device("cuda"), dtype=precision_dtype(config.model.precision)
            )
        return
    pipeline = denoiser.pipeline
    if config.model.vae_slicing:
        pipeline.enable_vae_slicing() if hasattr(pipeline, "enable_vae_slicing") else pipeline.vae.enable_slicing()
    if config.model.vae_tiling:
        pipeline.enable_vae_tiling() if hasattr(pipeline, "enable_vae_tiling") else pipeline.vae.enable_tiling()
    if config.model.cpu_offload:
        denoiser.enable_model_cpu_offload()
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required when model.cpu_offload=false")
        denoiser.to(torch.device("cuda"), dtype=precision_dtype(config.model.precision))


def _generate_with_selected_global_pipeline(config, denoiser, diagnostics):
    if config.global_pipeline.mode == "erp_rgb_state":
        return generate_erp_rgb(
            config, denoiser, diagnostics_writer=diagnostics
        )
    if config.global_pipeline.mode == "erp_x0_consensus":
        return generate_erp_x0_consensus(
            config, denoiser, diagnostics_writer=diagnostics
        )
    raise ValueError(
        f"Unsupported global pipeline {config.global_pipeline.mode!r}"
    )


def run(config: ExperimentConfig) -> Path:
    config.validate()
    set_random_seed(config.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    run_dir = _run_directory(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(config.to_dict()), run_dir / "config.yaml")
    log_path = run_dir / "run.log"
    log_path.write_text(f"started_at={datetime.now().isoformat()}\n", encoding="utf-8")

    denoiser = build_view_denoiser(config)
    _configure_denoiser(config, denoiser)
    diagnostics = DiagnosticsWriter(run_dir / "intermediates", config.debug)
    result = _generate_with_selected_global_pipeline(
        config, denoiser, diagnostics
    )
    peak_gpu_memory = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        scale = 1024.0 ** 3
        peak_gpu_memory = {
            "allocated_gib": torch.cuda.max_memory_allocated() / scale,
            "reserved_gib": torch.cuda.max_memory_reserved() / scale,
        }
    result.peak_gpu_memory_gib = peak_gpu_memory
    result_path = run_dir / "result.png"
    if config.output.save_final:
        tensor_to_pil(result.erp_rgb[0]).save(result_path)
    if config.output.save_metadata:
        save_run_metadata(str(run_dir / "metadata.json"), config, denoiser, result, str(result_path))
    with log_path.open("a", encoding="utf-8") as handle:
        for step in result.steps:
            handle.write(
                f"step={step.step_index} timestep={step.scheduler_timestep} cameras={step.num_cameras} "
                f"coverage={step.coverage_percent:.4f} overlap={step.multi_contributor_percent:.4f} "
                f"weights=({step.weight_min:.6g},{step.weight_max:.6g},{step.weight_mean:.6g}) "
                f"timings={step.timings_seconds} "
                f"state={step.state_statistics}\n"
            )
        handle.write(f"peak_gpu_memory_gib={peak_gpu_memory}\n")
        handle.write(f"completed_at={datetime.now().isoformat()}\nresult={result_path}\n")
    print(f"Run saved to {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DiffPano with a persistent ERP RGB canvas.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(load_experiment_config(args.config))


if __name__ == "__main__":
    main()
