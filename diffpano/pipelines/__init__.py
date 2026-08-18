"""Model adapter registry for spherical DiffPano generation."""

from typing import Any, Dict, Type

import torch

from diffpano.pipelines.base import DiffPanoPipelineAdapter, resolve_model_source
from diffpano.pipelines.flux import SphericalFluxPipeline
from diffpano.pipelines.hunyuan_video import SphericalHunyuanVideoPipeline
from diffpano.pipelines.ltx_video import SphericalLTXPipeline
from diffpano.pipelines.sana import SphericalSanaPipeline

DiffPanoSanaPipeline = SphericalSanaPipeline
DiffPanoFluxPipeline = SphericalFluxPipeline
DiffPanoHunyuanVideoPipeline = SphericalHunyuanVideoPipeline
DiffPanoLTXVideoPipeline = SphericalLTXPipeline

PIPELINE_REGISTRY: Dict[str, Type[DiffPanoPipelineAdapter]] = {
    "sana": SphericalSanaPipeline,
    "flux": SphericalFluxPipeline,
    "hunyuan_video": SphericalHunyuanVideoPipeline,
    "ltx_video": SphericalLTXPipeline,
}


def precision_dtype(precision: str) -> torch.dtype:
    try:
        return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    except KeyError as exc:
        raise ValueError(f"Unsupported model.precision={precision!r}") from exc


def build_pipeline(config: Any) -> DiffPanoPipelineAdapter:
    """Construct the configured backend without scattered pipeline conditionals."""

    try:
        pipeline_cls = PIPELINE_REGISTRY[config.model.pipeline]
    except KeyError as exc:
        names = ", ".join(sorted(PIPELINE_REGISTRY))
        raise ValueError(f"Unknown pipeline {config.model.pipeline!r}; choose one of: {names}") from exc
    source = resolve_model_source(config.model.path, config.model.id)
    kwargs = dict(config.model.additional_pipeline_kwargs)
    kwargs.update(revision=config.model.revision, variant=config.model.variant, torch_dtype=precision_dtype(config.model.precision))
    return pipeline_cls.from_pretrained(source, **kwargs)


__all__ = [
    "PIPELINE_REGISTRY",
    "build_pipeline",
    "DiffPanoSanaPipeline",
    "DiffPanoFluxPipeline",
    "DiffPanoHunyuanVideoPipeline",
    "DiffPanoLTXVideoPipeline",
]
