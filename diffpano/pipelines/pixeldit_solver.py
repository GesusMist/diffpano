"""Official-schedule, first-order flow stepping for synchronized PixelDiT states."""

from dataclasses import dataclass
from typing import Any, Tuple

import torch


def pixeldit_time_schedule(
    num_steps: int,
    flow_shift: float,
    *,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Reproduce PixelDiT ``time_uniform_flow`` for ``t_T=1`` and ``t_0=1e-3``."""

    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if flow_shift <= 0:
        raise ValueError("flow_shift must be positive")
    betas = torch.linspace(1.0, 0.001, num_steps + 1, device=device, dtype=torch.float32)
    sigmas = 1.0 - betas
    shifted = flow_shift * sigmas / (1.0 + (flow_shift - 1.0) * sigmas)
    return shifted.flip(dims=[0])


@dataclass
class PixelDiTFirstOrderSolver:
    """One DPM-Solver++ order-one update on the official shifted flow schedule."""

    flow_shift: float = 4.0
    schedule: torch.Tensor = None

    def prepare(self, num_steps: int, *, device: torch.device) -> torch.Tensor:
        self.schedule = pixeldit_time_schedule(num_steps, self.flow_shift, device=device)
        return self.schedule[:-1]

    @property
    def timesteps(self) -> torch.Tensor:
        if self.schedule is None:
            return torch.empty(0)
        return self.schedule[:-1]

    def bounds_for(self, timestep: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.schedule is None:
            raise RuntimeError("prepare() must be called before step()")
        current = torch.as_tensor(timestep, device=self.schedule.device, dtype=self.schedule.dtype)
        distances = (self.schedule[:-1] - current).abs()
        index = int(distances.argmin().item())
        if float(distances[index]) > 1.0e-6:
            raise ValueError(f"Unknown PixelDiT timestep {float(current)}")
        return self.schedule[index], self.schedule[index + 1]

    def step(self, x_t: torch.Tensor, flow_prediction: torch.Tensor, timestep: Any) -> torch.Tensor:
        """Advance from a raw flow prediction with the official order-one equation."""

        if x_t.shape != flow_prediction.shape:
            raise ValueError("PixelDiT flow prediction must match the RGB pixel state shape")
        current, _ = self.bounds_for(timestep)
        current = current.to(device=x_t.device, dtype=x_t.dtype)
        flow_prediction = flow_prediction.to(device=x_t.device, dtype=x_t.dtype)
        noise_prediction = (1.0 - current) * flow_prediction + x_t
        return self.step_from_noise(x_t, noise_prediction, timestep)

    def step_from_noise(
        self, x_t: torch.Tensor, noise_prediction: torch.Tensor, timestep: Any
    ) -> torch.Tensor:
        """Apply the exact upstream DPM-Solver++ first update to a wrapped prediction."""

        if x_t.shape != noise_prediction.shape:
            raise ValueError("PixelDiT wrapped prediction must match the RGB pixel state shape")
        current, following = self.bounds_for(timestep)
        current = current.to(device=x_t.device, dtype=x_t.dtype)
        following = following.to(device=x_t.device, dtype=x_t.dtype)
        noise_prediction = noise_prediction.to(device=x_t.device, dtype=x_t.dtype)

        alpha_current = 1.0 - current
        sigma_current = current
        alpha_following = 1.0 - following
        sigma_following = following
        clean_prediction = (x_t - sigma_current * noise_prediction) / alpha_current
        lambda_current = torch.log(alpha_current) - torch.log(sigma_current)
        lambda_following = torch.log(alpha_following) - torch.log(sigma_following)
        phi_one = torch.expm1(-(lambda_following - lambda_current))
        return (
            sigma_following / sigma_current * x_t
            - alpha_following * phi_one * clean_prediction
        )
