import inspect
import json
import math
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import torch


RECORDED_CALL_PARAMETERS = (
    "prompt_txt_path",
    "negative_prompt_txt_path",
    "num_inference_steps",
    "timesteps",
    "sigmas",
    "guidance_scale",
    "true_cfg_scale",
    "num_images_per_prompt",
    "height",
    "width",
    "num_frames",
    "eta",
    "n_spherical_points",
    "weighted_average_temperature",
    "erp_height",
    "erp_width",
    "pixel_fusion_config_path",
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() <= 4096:
            return tensor.tolist()
        return {
            "type": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    if isinstance(value, torch.Generator):
        return {
            "device": str(value.device),
            "initial_seed": value.initial_seed(),
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    return repr(value)


def effective_call_parameters(pipe: Any, supplied: Mapping[str, Any]) -> Dict[str, Any]:
    """Record supplied arguments plus the small set of defaults relevant to generation."""

    signature = inspect.signature(pipe.__call__)
    effective: Dict[str, Any] = {}
    for name in RECORDED_CALL_PARAMETERS:
        if name in supplied:
            effective[name] = supplied[name]
            continue
        parameter = signature.parameters.get(name)
        if parameter is not None and parameter.default is not inspect.Parameter.empty:
            effective[name] = parameter.default

    # Keep uncommon explicit overrides without dumping every pipeline default.
    for name, value in supplied.items():
        if name != "pixel_fusion_config":
            effective.setdefault(name, value)
    return _json_safe(effective)


def _count_histogram(values: Any) -> Dict[str, int]:
    histogram: Dict[str, int] = {}
    for value in values or []:
        key = str(int(value))
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def _compact_patch_counts(counts_by_step: Any) -> Dict[str, Any]:
    histograms = [_count_histogram(values) for values in counts_by_step or []]
    if not histograms:
        return {}
    if all(histogram == histograms[0] for histogram in histograms[1:]):
        return {
            "same_for_all_steps": True,
            "histogram": histograms[0],
        }
    return {
        "same_for_all_steps": False,
        "histograms_by_step": histograms,
    }


def _compact_runtime(runtime: Mapping[str, Any]) -> Dict[str, Any]:
    compact = {
        "random_seeds": runtime.get("generator_initial_seeds"),
        "n_spherical_points": runtime.get("n_spherical_points"),
        "denoise_steps": runtime.get("num_denoising_steps"),
        "denoise_timesteps": runtime.get("denoise_timesteps"),
        "patches_per_denoising_step": runtime.get("num_dynamic_view_patches_per_step"),
        "denoise_patch_points": _compact_patch_counts(runtime.get("denoise_patch_point_counts_by_step")),
        "pixel_fusion_applied_steps": [
            index for index, applied in enumerate(runtime.get("pixel_fusion_applied_by_step", [])) if applied
        ],
    }
    return _json_safe({key: value for key, value in compact.items() if value not in (None, {})})


def _pipeline_parameters(args: Any, pipe: Any) -> Dict[str, Any]:
    parameters = {"class": pipe.__class__.__name__}
    for name in ("pretrained_model_name_or_path", "revision", "variant", "mixed_precision"):
        value = getattr(args, name, None)
        if value is not None:
            parameters[name] = value
    return _json_safe(parameters)


def build_output_metadata(args: Any, pipe: Any, call_kwargs: Mapping[str, Any], output_file: str) -> Dict[str, Any]:
    runtime = getattr(pipe, "sphere_diff_run_metadata", {}) or {}
    metadata = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_file": str(Path(output_file).resolve()),
        "pipeline": _pipeline_parameters(args, pipe),
        "prompts": _json_safe(getattr(args, "prompt_to_log", None)),
        "generation_parameters": effective_call_parameters(pipe, call_kwargs),
        "runtime": _compact_runtime(runtime),
    }

    pixel_fusion_config = runtime.get("pixel_fusion_config", call_kwargs.get("pixel_fusion_config"))
    if pixel_fusion_config is not None:
        metadata["pixel_fusion_config"] = _json_safe(pixel_fusion_config)
    if os.environ.get("SLURM_JOB_ID"):
        metadata["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    return _json_safe(metadata)


def save_output_metadata(args: Any, pipe: Any, call_kwargs: Mapping[str, Any], output_file: str) -> str:
    metadata_file = str(Path(output_file).with_suffix(".json"))
    metadata = build_output_metadata(args, pipe, call_kwargs, output_file)
    with open(metadata_file, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return metadata_file
