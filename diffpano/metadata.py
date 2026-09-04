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

    metadata = {
        "schema_version": 6,
        "architecture": (
            "persistent_erp_rgb"
            if config.global_pipeline.mode == "erp_rgb_state"
            else "persistent_predicted_clean_erp_rgb"
        ),
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
        "config": config.to_dict(),
        "peak_gpu_memory_gib": getattr(result, "peak_gpu_memory_gib", {}),
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
