"""Controlled single-patch native sampling versus fixed-epsilon x0 renoising."""

from dataclasses import dataclass

import torch

from diffpano.pipelines.base import reset_scheduler_step_state
from diffpano.pipelines.clean_prediction import flow_sigma


def state_stats(value, prefix):
    value = value.detach().float()
    return {f"{prefix}_mean": float(value.mean()), f"{prefix}_std": float(value.std(unbiased=False))}


def schedule_stats(backend, timestep):
    scheduler = getattr(getattr(backend, "pipeline", None), "scheduler", None)
    if scheduler is not None and hasattr(scheduler, "sigmas"):
        return {"scheduler_sigma": float(flow_sigma(scheduler, timestep, device=backend.device, dtype=torch.float32))}
    if scheduler is not None and hasattr(scheduler, "alphas_cumprod"):
        return {"alpha_cumprod": float(scheduler.alphas_cumprod[int(timestep)])}
    if hasattr(backend, "solver"):
        current, _ = backend.solver.bounds_for(timestep)
        return {"flow_time": float(current)}
    return {}


@dataclass
class TrajectoryResult:
    native_final: torch.Tensor
    x0_renoise_final: torch.Tensor
    initial_epsilon: torch.Tensor
    steps: list
    model_evaluations: dict


@torch.no_grad()
def compare_single_patch(backend, conditioning, epsilon):
    """One backend, one conditioning object, one frozen schedule and Gaussian.

    Model evaluations count guided predictions. FLUX true CFG executes two
    transformer forwards per prediction in BOTH trajectories.
    """
    timesteps = backend.timesteps.clone()
    if len(timesteps) < 1:
        raise ValueError("Trajectory comparison requires at least one timestep")
    original = epsilon.detach().clone()
    native = backend.initialize_native_state(original.clone())
    renoise_initial = backend.make_initial_noisy_state(original.clone(), timesteps[0])
    if not torch.equal(native, renoise_initial):
        raise ValueError("Native and x0 trajectories must begin with exactly the same initialized Gaussian state")
    clean = None
    records = []
    counts = {"native": 0, "x0_renoise": 0}
    # Interleaving keeps memory bounded. Explicit scheduler resets prevent A's
    # integration state from leaking into B's forward noising operation.
    for i, timestep in enumerate(timesteps):
        native = backend.denoise_native_step(native, timestep, conditioning)
        counts["native"] += 1
        scheduler = getattr(getattr(backend, "pipeline", None), "scheduler", None)
        if scheduler is not None:
            reset_scheduler_step_state(scheduler)
        noisy = (renoise_initial.clone() if i == 0
                 else backend.add_fixed_noise(clean, original.clone(), timestep))
        record = {"step_index": i, "scheduler_timestep": float(timestep),
                  **schedule_stats(backend, timestep), **state_stats(native, "native_state"),
                  **state_stats(noisy, "x0_renoise_noisy_state")}
        clean = backend.predict_clean_native(noisy, timestep, conditioning)
        counts["x0_renoise"] += 1
        record.update(state_stats(clean, "predicted_clean"))
        if not bool(torch.isfinite(native).all() and torch.isfinite(clean).all()):
            raise ValueError(f"Non-finite single-patch trajectory at step {i}")
        records.append(record)
    if not torch.equal(epsilon, original) or not torch.equal(backend.timesteps, timesteps):
        raise AssertionError("Trajectory comparison mutated its Gaussian or timestep schedule")
    # Decode endpoints only. Intermediate latent decoding is deliberately absent.
    a = backend.decode_native_canvas(native)
    b = backend.decode_clean(clean)
    records[-1].update(final_rgb_L1=float((a-b).abs().mean()),
                       final_rgb_RMSE=float((a-b).square().mean().sqrt()))
    return TrajectoryResult(a, b, original, records, counts)
