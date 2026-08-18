"""Minimal interface documenting model-specific DiffPano adapter boundaries."""

from pathlib import Path
from typing import Any, Protocol


class DiffPanoPipelineAdapter(Protocol):
    """Structural interface used by the common generation entrypoint."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> "DiffPanoPipelineAdapter":
        ...

    def __call__(self, **kwargs: Any) -> Any:
        ...

    def to(self, *args: Any, **kwargs: Any) -> "DiffPanoPipelineAdapter":
        ...

    def enable_model_cpu_offload(self) -> None:
        ...


def resolve_model_source(path: str, model_id: str) -> str:
    """Prefer an explicit local path, otherwise use the configured model ID."""

    if path:
        return str(Path(path).expanduser())
    if model_id:
        return model_id
    raise ValueError("model.path or model.id must be configured")
