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


def run(config: ExperimentConfig) -> Path:
    config.validate()
    set_random_seed(config.experiment.seed)
    run_dir = _run_directory(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(config.to_dict()), run_dir / "config.yaml")
    log_path = run_dir / "run.log"
    log_path.write_text(f"started_at={datetime.now().isoformat()}\n", encoding="utf-8")

    denoiser = build_view_denoiser(config)
    _configure_denoiser(config, denoiser)
    diagnostics = DiagnosticsWriter(run_dir / "intermediates", config.debug)
    result = generate_erp_rgb(config, denoiser, diagnostics_writer=diagnostics)
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
                f"timings={step.timings_seconds}\n"
            )
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
