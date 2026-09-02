"""Local RGB diffusion interface used by the global ERP orchestrator."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import torch

from diffpano.camera import PerspectiveCamera


class ViewDenoiser(ABC):
    """Advance perspective RGB by exactly one backend-native diffusion step."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...

    @property
    @abstractmethod
    def timesteps(self) -> torch.Tensor:
        ...

    @abstractmethod
    def prepare(self, *, num_steps: int, view_height: int, view_width: int) -> None:
        ...

    @abstractmethod
    def prepare_prompt_conditioning(self, prompts: Sequence[str], negative_prompt: str = "") -> Any:
        ...

    @abstractmethod
    def conditioning_for_cameras(
        self,
        prepared_conditioning: Any,
        cameras: Sequence[PerspectiveCamera],
        *,
        batch_size: int,
    ) -> Any:
        ...

    @abstractmethod
    def denoise_step(self, rgb_view: torch.Tensor, timestep: Any, conditioning: Any) -> torch.Tensor:
        """Return RGB at the next local scheduler state."""

    @abstractmethod
    def sample_native_rgb(
        self, *, batch_size: int, height: int, width: int, generator: torch.Generator
    ) -> torch.Tensor:
        ...


class MockViewDenoiser(ViewDenoiser):
    """Small deterministic denoiser for geometry and synchronization tests."""

    def __init__(self, num_steps: int = 1, delta: float = 0.0, device: torch.device = torch.device("cpu")):
        self._device = device
        self._timesteps = torch.arange(num_steps - 1, -1, -1, device=device)
        self.delta = delta

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def timesteps(self) -> torch.Tensor:
        return self._timesteps

    def prepare(self, *, num_steps: int, view_height: int, view_width: int) -> None:
        del view_height, view_width
        self._timesteps = torch.arange(num_steps - 1, -1, -1, device=self.device)

    def prepare_prompt_conditioning(self, prompts: Sequence[str], negative_prompt: str = "") -> Any:
        del negative_prompt
        return list(prompts)

    def conditioning_for_cameras(self, prepared_conditioning, cameras, *, batch_size: int):
        del prepared_conditioning
        return torch.tensor(
            [camera.yaw for camera in cameras for _ in range(batch_size)],
            device=self.device,
            dtype=torch.float32,
        )

    def denoise_step(self, rgb_view: torch.Tensor, timestep: Any, conditioning: Any) -> torch.Tensor:
        del timestep, conditioning
        return rgb_view.to(dtype=torch.float32) + self.delta

    def sample_native_rgb(self, *, batch_size: int, height: int, width: int, generator):
        return torch.randn(batch_size, 3, height, width, device=self.device, generator=generator)


def resolve_model_source(path: str, model_id: str) -> str:
    if path:
        return str(Path(path).expanduser())
    if model_id:
        return model_id
    raise ValueError("model.path or model.id must be configured")


def ensure_first_order_scheduler(scheduler: Any) -> None:
    """Reject history-dependent solvers for independently reconstructed view states."""

    order = int(getattr(scheduler, "order", 1))
    solver_order = int(getattr(getattr(scheduler, "config", None), "solver_order", 1))
    if order > 1 or solver_order > 1:
        raise ValueError(
            "ERP-RGB local steps require a first-order scheduler because each camera is independently VAE-encoded"
        )


def reset_scheduler_step_state(scheduler: Any) -> None:
    """Give every same-timestep camera an independent scheduler call state."""

    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    if hasattr(scheduler, "model_outputs"):
        scheduler.model_outputs = [None] * len(scheduler.model_outputs)
