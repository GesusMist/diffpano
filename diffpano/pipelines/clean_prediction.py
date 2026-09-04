"""Backend-native clean-prediction contract and exact scheduler math helpers."""

from typing import Any, Protocol, Sequence, runtime_checkable

import torch

from diffpano.camera import PerspectiveCamera


@runtime_checkable
class CleanPredictionBackend(Protocol):
    """Additional operations used by the clean-ERP consensus global pipeline."""

    @property
    def device(self) -> torch.device:
        ...

    @property
    def timesteps(self) -> torch.Tensor:
        ...

    def sample_fixed_noise(
        self,
        *,
        batch_size: int,
        height: int,
        width: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        ...

    def make_initial_noisy_state(
        self, fixed_noise: torch.Tensor, timestep: Any
    ) -> torch.Tensor:
        ...

    def encode_clean(self, rgb_clean: torch.Tensor) -> torch.Tensor:
        ...

    def add_fixed_noise(
        self,
        clean_state: torch.Tensor,
        fixed_noise: torch.Tensor,
        timestep: Any,
    ) -> torch.Tensor:
        ...

    def predict_clean_native(
        self, noisy_state: torch.Tensor, timestep: Any, conditioning: Any
    ) -> torch.Tensor:
        ...

    def decode_clean(self, clean_state: torch.Tensor) -> torch.Tensor:
        ...

    def conditioning_for_cameras(
        self,
        prepared_conditioning: Any,
        cameras: Sequence[PerspectiveCamera],
        *,
        batch_size: int,
    ) -> Any:
        ...


def _timestep_batch(timestep: Any, batch: int, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(timestep, device=device)
    if value.ndim == 0:
        value = value.unsqueeze(0)
    if value.numel() == 1:
        value = value.expand(batch)
    if value.numel() != batch:
        raise ValueError("Timestep batch does not match native-state batch")
    return value


def flow_sigma(
    scheduler: Any,
    timestep: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the exact inference sigma paired with a FlowMatch timestep."""

    schedule = scheduler.timesteps.to(device=device)
    current = torch.as_tensor(timestep, device=device, dtype=schedule.dtype)
    distances = (schedule - current).abs()
    index = int(distances.argmin().item())
    tolerance = max(1.0e-6, float(current.abs()) * 1.0e-6)
    if float(distances[index]) > tolerance:
        raise ValueError(f"Unknown flow scheduler timestep {float(current)}")
    return scheduler.sigmas[index].to(device=device, dtype=dtype)


def flow_add_noise(
    scheduler: Any,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: Any,
) -> torch.Tensor:
    """Use the scheduler's official flow forward-process helper."""

    if clean.shape != noise.shape:
        raise ValueError("Clean flow state and fixed noise must have identical shapes")
    timesteps = _timestep_batch(timestep, clean.shape[0], clean.device)
    native_noise = noise.to(clean)
    if callable(getattr(scheduler, "scale_noise", None)):
        # FlowMatchEulerDiscreteScheduler (Flux).
        return scheduler.scale_noise(clean, timesteps, native_noise)
    if callable(getattr(scheduler, "add_noise", None)):
        # DPMSolverMultistepScheduler with use_flow_sigmas=True (SANA).
        return scheduler.add_noise(clean, native_noise, timesteps)
    raise TypeError(
        f"{type(scheduler).__name__} exposes neither scale_noise nor add_noise"
    )


def flow_predicted_clean(
    scheduler: Any,
    noisy: torch.Tensor,
    flow_prediction: torch.Tensor,
    timestep: Any,
) -> torch.Tensor:
    """Recover x0 from x_sigma=(1-sigma)x0+sigma*epsilon and v=epsilon-x0."""

    if noisy.shape != flow_prediction.shape:
        raise ValueError("Noisy flow state and prediction must have identical shapes")
    sigma = flow_sigma(
        scheduler, timestep, device=noisy.device, dtype=noisy.dtype
    )
    return noisy - sigma * flow_prediction.to(noisy)


def ddim_predicted_clean(
    scheduler: Any,
    noisy: torch.Tensor,
    model_prediction: torch.Tensor,
    timestep: Any,
) -> torch.Tensor:
    """Mirror DDIMScheduler's official pred_original_sample computation only."""

    if noisy.shape != model_prediction.shape:
        raise ValueError("Noisy DDIM state and prediction must have identical shapes")
    timestep_tensor = torch.as_tensor(timestep, device=noisy.device, dtype=torch.long)
    if timestep_tensor.numel() != 1:
        raise ValueError("DDIM clean conversion expects one scheduler timestep")
    alpha = scheduler.alphas_cumprod.to(
        device=noisy.device, dtype=noisy.dtype
    )[timestep_tensor]
    beta = 1.0 - alpha
    prediction_type = scheduler.config.prediction_type
    prediction = model_prediction.to(noisy)
    if prediction_type == "epsilon":
        clean = (noisy - beta.sqrt() * prediction) / alpha.sqrt()
    elif prediction_type == "sample":
        clean = prediction
    elif prediction_type == "v_prediction":
        clean = alpha.sqrt() * noisy - beta.sqrt() * prediction
    else:
        raise ValueError(f"Unsupported DDIM prediction_type={prediction_type!r}")
    if scheduler.config.thresholding:
        clean = scheduler._threshold_sample(clean)
    elif scheduler.config.clip_sample:
        clean = clean.clamp(
            -scheduler.config.clip_sample_range,
            scheduler.config.clip_sample_range,
        )
    return clean
