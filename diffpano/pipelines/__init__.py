"""Local view-denoiser registry. Imports are lazy to keep CPU tests lightweight."""

from typing import Any

import torch

from diffpano.pipelines.base import MockViewDenoiser, ViewDenoiser, resolve_model_source


def precision_dtype(precision: str) -> torch.dtype:
    try:
        return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    except KeyError as exc:
        raise ValueError(f"Unsupported precision {precision!r}") from exc


def build_view_denoiser(config: Any) -> ViewDenoiser:
    source = resolve_model_source(config.model.path, config.model.id)
    kwargs = dict(config.model.additional_pipeline_kwargs)
    kwargs.update(
        revision=config.model.revision,
        variant=config.model.variant,
        torch_dtype=precision_dtype(config.model.precision),
    )
    # Diffusers does not accept explicit None values uniformly across all loaders.
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    common = {
        "guidance_scale": config.generation.guidance_scale,
        "vae_chunk_size": config.performance.vae_chunk_size,
        "measure_performance": config.debug.measure_performance,
    }
    if config.model.pipeline == "sana":
        from diffpano.pipelines.sana import SanaViewDenoiser

        return SanaViewDenoiser.from_pretrained(source, **common, **kwargs)
    if config.model.pipeline == "flux":
        from diffpano.pipelines.flux import FluxViewDenoiser

        return FluxViewDenoiser.from_pretrained(
            source, true_cfg_scale=config.generation.true_cfg_scale, **common, **kwargs
        )
    raise ValueError(f"Unsupported local denoiser {config.model.pipeline!r}")


__all__ = [
    "MockViewDenoiser",
    "ViewDenoiser",
    "build_view_denoiser",
    "precision_dtype",
]
