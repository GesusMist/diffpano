"""Scheduler-aware predicted-clean correction and reinjection."""

from typing import Any, Optional, Tuple

import torch

from experiments.legacy_spherical.diffpano_legacy.config import FUSION_DTYPE, PixelFusionConfig


def _scheduler_index_for_timestep(scheduler: Any, timestep: torch.Tensor) -> int:
    schedule_timesteps = scheduler.timesteps.to(device=timestep.device)
    if timestep.ndim == 0:
        t = timestep
    else:
        t = timestep.flatten()[0]
    if hasattr(scheduler, "index_for_timestep"):
        return int(scheduler.index_for_timestep(t, schedule_timesteps))
    matches = (schedule_timesteps == t).nonzero()
    if matches.numel() > 0:
        return int(matches.flatten()[0])
    return int(torch.argmin((schedule_timesteps.to(dtype=torch.float32) - t.to(dtype=torch.float32)).abs()))


def _scheduler_sigma_pair(
    scheduler: Any,
    timestep: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(scheduler, "sigmas"):
        raise ValueError(
            f"Pixel fusion requires a scheduler with an explicit sigma schedule, got {scheduler.__class__.__name__}"
        )
    index = _scheduler_index_for_timestep(scheduler, timestep)
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    if index + 1 >= sigmas.shape[0]:
        raise ValueError("Scheduler sigmas do not contain sigma_next for the current timestep")
    return sigmas[index], sigmas[index + 1]


def _scheduler_prediction_type(scheduler: Any) -> Optional[str]:
    prediction_type = getattr(getattr(scheduler, "config", None), "prediction_type", None)
    if prediction_type is not None:
        return str(prediction_type)
    if "flowmatch" in scheduler.__class__.__name__.lower():
        return "flow_prediction"
    return None


def predict_clean_latents(
    scheduler: Any,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert the configured scheduler prediction into predicted-clean view latents x0."""

    sample = sample.to(dtype=FUSION_DTYPE)
    model_output = model_output.to(device=sample.device, dtype=FUSION_DTYPE)
    sigma, sigma_next = _scheduler_sigma_pair(scheduler, timestep, device=sample.device, dtype=FUSION_DTYPE)
    while sigma.ndim < sample.ndim:
        sigma = sigma.unsqueeze(-1)
        sigma_next = sigma_next.unsqueeze(-1)
    prediction_type = _scheduler_prediction_type(scheduler)
    if prediction_type == "flow_prediction":
        # Both FlowMatch Euler and SANA's flow-sigma DPM-Solver use x_t = x0 + sigma * flow.
        clean = sample - sigma * model_output
    elif prediction_type == "sample":
        clean = model_output
    elif prediction_type in {"epsilon", "v_prediction"} and hasattr(scheduler, "_sigma_to_alpha_sigma_t"):
        alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma)
        if prediction_type == "epsilon":
            clean = (sample - sigma_t * model_output) / alpha_t.clamp_min(torch.finfo(sample.dtype).eps)
        else:
            clean = alpha_t * sample - sigma_t * model_output
    else:
        raise ValueError(
            f"Unsupported scheduler prediction_type={prediction_type!r} for predicted-clean pixel fusion "
            f"with {scheduler.__class__.__name__}"
        )
    return clean, sigma, sigma_next


def _reset_scheduler_for_independent_first_order_step(scheduler: Any) -> None:
    """Reset state mutated by an earlier independent patch step at the same timestep."""

    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    if hasattr(scheduler, "model_outputs"):
        for index in range(len(scheduler.model_outputs)):
            scheduler.model_outputs[index] = None
    if hasattr(scheduler, "lower_order_nums"):
        scheduler.lower_order_nums = 0
    if hasattr(scheduler, "last_sample"):
        scheduler.last_sample = None


def step_with_fused_clean_prediction(
    scheduler: Any,
    timestep: torch.Tensor,
    current_latents: torch.Tensor,
    original_model_output: torch.Tensor,
    original_clean_latents: torch.Tensor,
    target_clean_latents: torch.Tensor,
    reinjection_strength: float,
    *,
    original_prev_latents: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Rerun a flow scheduler using an identity-preserving fused clean prediction."""

    strength = float(reinjection_strength)
    if strength == 0 and original_prev_latents is not None:
        return original_prev_latents
    prediction_type = _scheduler_prediction_type(scheduler)
    if prediction_type != "flow_prediction":
        raise ValueError(
            "noise_consistent reinjection requires flow_prediction so the corrected clean prediction can be "
            f"converted back to scheduler flow; got prediction_type={prediction_type!r}"
        )
    if not (
        current_latents.shape
        == original_model_output.shape
        == original_clean_latents.shape
        == target_clean_latents.shape
    ):
        raise ValueError("Flow reinjection tensors must all have the same shape")

    output_dtype = original_prev_latents.dtype if original_prev_latents is not None else current_latents.dtype
    original_clean = original_clean_latents.to(dtype=FUSION_DTYPE)
    target_clean = target_clean_latents.to(device=original_clean.device, dtype=FUSION_DTYPE)
    original_flow = original_model_output.to(device=original_clean.device, dtype=FUSION_DTYPE)
    sigma_current = _scheduler_sigma_pair(
        scheduler,
        timestep,
        device=original_clean.device,
        dtype=FUSION_DTYPE,
    )[0]
    while sigma_current.ndim < original_clean.ndim:
        sigma_current = sigma_current.unsqueeze(-1)
    if torch.any(sigma_current.abs() <= torch.finfo(FUSION_DTYPE).eps):
        raise ValueError("Cannot reconstruct corrected flow when sigma_current is zero")

    clean_correction = strength * (target_clean - original_clean)
    if valid_mask is not None:
        mask = valid_mask.to(device=original_clean.device, dtype=FUSION_DTYPE)
        while mask.ndim < clean_correction.ndim:
            mask = mask.unsqueeze(-1)
        clean_correction = clean_correction * mask

    # Since original_clean = current - sigma_current * original_flow, this is algebraically
    # identical to (current - corrected_clean) / sigma_current while preserving the original
    # model flow exactly when the clean correction is zero.
    corrected_flow = original_flow - clean_correction / sigma_current
    _reset_scheduler_for_independent_first_order_step(scheduler)
    corrected_prev = scheduler.step(
        corrected_flow,
        timestep,
        current_latents,
        return_dict=False,
    )[0]
    return corrected_prev.to(dtype=output_dtype)


def reinject_fused_latents(
    original_clean_latents: torch.Tensor,
    fused_clean_latents: torch.Tensor,
    original_prev_latents: torch.Tensor,
    model_output: torch.Tensor,
    sigma_next: torch.Tensor,
    config: PixelFusionConfig,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    next_clean_weight: Optional[torch.Tensor] = None,
    scheduler: Optional[Any] = None,
    timestep: Optional[torch.Tensor] = None,
    current_latents: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if config.reinjection_mode == "noise_consistent":
        if scheduler is None or timestep is None or current_latents is None:
            raise ValueError(
                "noise_consistent reinjection requires scheduler, timestep, and current_latents"
            )
        return step_with_fused_clean_prediction(
            scheduler,
            timestep,
            current_latents,
            model_output,
            original_clean_latents,
            fused_clean_latents,
            config.reinjection_strength,
            original_prev_latents=original_prev_latents,
            valid_mask=valid_mask,
        )

    del sigma_next, next_clean_weight
    output_dtype = original_prev_latents.dtype
    original_clean_latents = original_clean_latents.to(dtype=FUSION_DTYPE)
    fused_clean_latents = fused_clean_latents.to(device=original_clean_latents.device, dtype=FUSION_DTYPE)
    original_prev_latents = original_prev_latents.to(device=original_clean_latents.device, dtype=FUSION_DTYPE)
    del model_output
    strength = float(config.reinjection_strength)
    clean_delta = fused_clean_latents - original_clean_latents
    if config.reinjection_mode == "replace":
        result = original_prev_latents * (1 - strength) + fused_clean_latents * strength
        return result.to(dtype=output_dtype)
    if config.reinjection_mode == "weighted_replace":
        if valid_mask is None:
            valid_mask = torch.ones_like(fused_clean_latents[:, :1])
        else:
            valid_mask = valid_mask.to(device=original_clean_latents.device, dtype=FUSION_DTYPE)
        while valid_mask.ndim < fused_clean_latents.ndim:
            valid_mask = valid_mask.unsqueeze(-1)
        alpha = (valid_mask * strength).to(fused_clean_latents.dtype)
        result = original_prev_latents * (1 - alpha) + fused_clean_latents * alpha
        return result.to(dtype=output_dtype)
    if config.reinjection_mode == "residual":
        result = original_prev_latents + strength * clean_delta
        return result.to(dtype=output_dtype)
    raise ValueError(f"Unsupported reinjection_mode={config.reinjection_mode!r}")
