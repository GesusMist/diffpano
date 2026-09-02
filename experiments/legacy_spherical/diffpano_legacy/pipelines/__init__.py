"""Registry for the archived spherical-latent SANA and FLUX image adapters."""

from typing import Any, Dict, Type

import torch

from experiments.legacy_spherical.diffpano_legacy.pipelines.base import (
    DiffPanoPipelineAdapter,
    resolve_model_source,
)
from experiments.legacy_spherical.diffpano_legacy.pipelines.flux import SphericalFluxPipeline
from experiments.legacy_spherical.diffpano_legacy.pipelines.sana import SphericalSanaPipeline

PIPELINE_REGISTRY: Dict[str, Type[DiffPanoPipelineAdapter]] = {
    "sana": SphericalSanaPipeline,
    "flux": SphericalFluxPipeline,
}


def precision_dtype(precision: str) -> torch.dtype:
    try:
        return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    except KeyError as exc:
        raise ValueError(f"Unsupported model.precision={precision!r}") from exc


def build_pipeline(config: Any) -> DiffPanoPipelineAdapter:
    try:
        pipeline_cls = PIPELINE_REGISTRY[config.model.pipeline]
    except KeyError as exc:
        raise ValueError("The archived baseline supports only sana and flux") from exc
    source = resolve_model_source(config.model.path, config.model.id)
    kwargs = dict(config.model.additional_pipeline_kwargs)
    kwargs.update(
        revision=config.model.revision,
        variant=config.model.variant,
        torch_dtype=precision_dtype(config.model.precision),
    )
    return pipeline_cls.from_pretrained(source, **kwargs)


__all__ = ["PIPELINE_REGISTRY", "SphericalFluxPipeline", "SphericalSanaPipeline", "build_pipeline"]
