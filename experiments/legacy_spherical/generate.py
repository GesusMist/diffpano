#!/usr/bin/env python3
"""Canonical DiffPano generation entrypoint."""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from experiments.legacy_spherical.diffpano_legacy.config import ExperimentConfig, load_experiment_config
from experiments.legacy_spherical.diffpano_legacy.initialization import load_directional_prompts, set_random_seed
from experiments.legacy_spherical.diffpano_legacy.metadata import save_run_metadata
from experiments.legacy_spherical.diffpano_legacy.pipelines import build_pipeline, precision_dtype


def _safe_component(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in value)
    return cleaned.strip("-") or "run"


def _run_directory(config: ExperimentConfig) -> Path:
    run_id = config.output.run_id
    if not run_id:
        suffix = os.environ.get("SLURM_JOB_ID", "local")
        run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{suffix}"
    return Path(config.output.directory) / _safe_component(config.experiment.name) / _safe_component(run_id)


def _diagnostics_enabled(config: ExperimentConfig) -> bool:
    debug = config.debug
    return config.output.save_intermediate or any(
        (
            debug.enabled,
            debug.save_predicted_x0,
            debug.save_original_clean_erp,
            debug.save_fused_erp,
            debug.save_owner_map,
            debug.save_projection_diagnostics,
            debug.save_masks,
            debug.measure_performance,
        )
    )


def build_call_kwargs(config: ExperimentConfig, run_dir: Path) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "prompt_txt_path": config.prompt.path,
        "negative_prompt_txt_path": config.prompt.negative_path or "",
        "num_inference_steps": config.generation.num_inference_steps,
        "guidance_scale": config.generation.guidance_scale,
        "height": config.generation.height,
        "width": config.generation.width,
        "n_spherical_points": config.sphere.num_points,
        "weighted_average_temperature": config.sphere.weighted_average_temperature,
        "erp_height": config.sphere.erp_height,
        "erp_width": config.sphere.erp_width,
    }
    if config.model.pipeline == "sana":
        kwargs["use_resolution_binning"] = config.generation.use_resolution_binning
    if config.generation.num_frames is not None and config.is_video:
        kwargs["num_frames"] = config.generation.num_frames
    if config.model.pipeline in {"sana", "flux"}:
        diagnostics_dir = str(run_dir / "intermediates") if _diagnostics_enabled(config) else None
        kwargs["pixel_fusion_config"] = config.to_pixel_fusion_config(diagnostics_dir).to_dict()
    kwargs.update(config.generation.additional_call_kwargs)
    return kwargs


def _configure_pipeline(config: ExperimentConfig, pipe: Any) -> None:
    if config.model.vae_slicing:
        pipe.enable_vae_slicing() if hasattr(pipe, "enable_vae_slicing") else pipe.vae.enable_slicing()
    if config.model.vae_tiling:
        pipe.enable_vae_tiling() if hasattr(pipe, "enable_vae_tiling") else pipe.vae.enable_tiling()
    if config.model.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required when model.cpu_offload=false")
        dtype = precision_dtype(config.model.precision)
        pipe.to(torch.device("cuda"), dtype=dtype)
    if getattr(getattr(pipe, "scheduler", None), "config", {}).get("solver_order", 1) > 1:
        print("Warning: solver_order > 1 is unsupported; preserving legacy behavior by setting it to 1.")
        pipe.scheduler.config.solver_order = 1


def run(config: ExperimentConfig) -> Path:
    config.validate()
    if config.model.pipeline == "planar_sana":
        raise ValueError("planar_sana is an ablation; use python scripts/planar_test.py --config <config>")
    prompts = load_directional_prompts(config.prompt.path)
    config.resolved_prompts = prompts
    set_random_seed(config.experiment.seed)

    run_dir = _run_directory(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"
    resolved_path = run_dir / "config.yaml"
    OmegaConf.save(OmegaConf.create(config.to_dict()), resolved_path)
    log_path.write_text(f"started_at={datetime.now().isoformat()}\n", encoding="utf-8")

    print(OmegaConf.to_yaml(OmegaConf.create(config.to_dict())))
    pipe = build_pipeline(config)
    _configure_pipeline(config, pipe)
    call_kwargs = build_call_kwargs(config, run_dir)
    output = pipe(**call_kwargs)

    result_path = run_dir / ("result.mp4" if config.is_video else "result.png")
    if config.output.save_final:
        if config.is_video:
            export_to_video(output.frames[0], str(result_path), fps=config.output.fps)
        else:
            output.images[0].save(result_path)
    if config.output.save_metadata:
        save_run_metadata(str(run_dir / "metadata.json"), config, pipe, call_kwargs, str(result_path))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"completed_at={datetime.now().isoformat()}\n")
        handle.write(f"result={result_path}\n")
    print(f"Run saved to {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a DiffPano panorama from one experiment config.")
    parser.add_argument("--config", default="config.yaml", help="Path to the complete experiment YAML.")
    args = parser.parse_args()
    run(load_experiment_config(args.config))


if __name__ == "__main__":
    main()
