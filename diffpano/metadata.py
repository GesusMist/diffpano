"""Compact reproducibility metadata for ERP-RGB runs."""

import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def save_run_metadata(path: str, config: Any, denoiser: Any, result: Any, output_file: str) -> str:
    if config.model.pipeline == "pixeldit":
        model_source = denoiser.checkpoint_path or config.pixeldit.model_path or config.pixeldit.checkpoint_name
    else:
        model_source = config.model.path or config.model.id

    scheduler = getattr(getattr(denoiser, "pipeline", None), "scheduler", None)
    scheduler_config = getattr(scheduler, "config", None)
    metadata = {
        "schema_version": 9,
        "node": platform.node(),
        "scheduler_timesteps": denoiser.timesteps.detach().cpu().tolist(),
        "pixeldit_flow_schedule": denoiser.solver.schedule.detach().cpu().tolist() if hasattr(denoiser, "solver") else None,
        "scheduler_sigmas": scheduler.sigmas.detach().cpu().tolist() if scheduler is not None and hasattr(scheduler, "sigmas") else None,
        "architecture": (
            "local_native_states_global_clean_rgb"
            if config.global_pipeline.mode == "implied_endpoint_consensus"
            else "persistent_planar_native"
            if config.global_pipeline.mode == "native_multidiffusion"
            else f"persistent_{config.canvas.mode}_rgb"
            if config.global_pipeline.mode == "erp_rgb_state"
            else f"persistent_predicted_clean_{config.canvas.mode}_rgb"
        ),
        "native_geometry": ({
            "channels": denoiser.native_channels,
            "spatial_factor": denoiser.native_spatial_factor,
            "local_rgb_shape": denoiser.rgb_spatial_shape_for_native(
                config.native_multidiffusion.patch_size, config.native_multidiffusion.patch_size),
            "scheduler_shift_mu": getattr(denoiser, "scheduler_shift_mu", None),
            "scheduler_image_seq_len": getattr(denoiser, "scheduler_image_seq_len", None),
        } if config.global_pipeline.mode in {"native_multidiffusion", "implied_endpoint_consensus"} else None),
        "canvas_mode": config.canvas.mode,
        "global_pipeline_mode": config.global_pipeline.mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_file": str(Path(output_file).resolve()),
        "experiment": asdict(config.experiment),
        "model": {
            "backend": config.model.pipeline,
            "source": model_source,
            "official_commit": getattr(denoiser, "official_commit", None),
            "flow_shift": getattr(getattr(denoiser, "solver", None), "flow_shift", None),
            "adapter": denoiser.__class__.__name__,
        },
        "scheduler": ({
            "class": type(scheduler).__name__,
            "order": getattr(scheduler, "order", None),
            "solver_order": getattr(scheduler_config, "solver_order", None),
            "prediction_type": getattr(scheduler_config, "prediction_type", None),
            "use_flow_sigmas": getattr(scheduler_config, "use_flow_sigmas", None),
            "use_dynamic_shifting": getattr(scheduler_config, "use_dynamic_shifting", None),
            "init_noise_sigma": float(getattr(scheduler, "init_noise_sigma", 1.0)),
        } if scheduler is not None else {"class": "PixelDiT official first-order DPM"}),
        "backend_details": denoiser.backend_metadata() if callable(getattr(denoiser, "backend_metadata", None)) else None,
        "config": config.to_dict(),
        "runtime_seconds": getattr(result, "runtime_seconds", None),
        "peak_gpu_memory_gib": getattr(result, "peak_gpu_memory_gib", {}),
        "consensus_audit": getattr(result, "audit", None),
        "fixed_noise_identities": getattr(
            result, "fixed_noise_identities", {}
        ),
        "steps": [asdict(step) for step in result.steps],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    if os.environ.get("SLURM_JOB_ID"):
        metadata["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(destination)
