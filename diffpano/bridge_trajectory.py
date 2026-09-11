"""Single-patch C/D controls. No spatial fusion code or patch accumulator."""
from dataclasses import dataclass

import torch

from diffpano.pipelines.base import reset_scheduler_step_state
from diffpano.trajectory import conditioning_digest
from diffpano.vae_bridge import projection_metrics


@dataclass
class BridgeTrajectoryResult:
    final_rgb: torch.Tensor
    steps: list
    model_evaluations: int


@torch.no_grad()
def run_roundtrip_trajectory(backend, initial_state, conditioning, bridge=None):
    state = initial_state.clone().float()
    timesteps = backend.timesteps.clone()
    digest = conditioning_digest(conditioning)
    scheduler = getattr(getattr(backend,'pipeline',None),'scheduler',None)
    steps=[]; evaluations=0
    for index,timestep in enumerate(timesteps):
        if scheduler is not None: reset_scheduler_step_state(scheduler)
        before = getattr(backend,'guided_prediction_count',0)
        endpoints = backend.predict_clean_and_endpoint(state,timestep,conditioning)
        delta = getattr(backend,'guided_prediction_count',0)-before
        if delta!=1: raise AssertionError('C/D requires one guided prediction per step')
        evaluations += delta
        # Exactly one decode and one encode of this step's clean prediction.
        roundtrip = backend.encode_clean(backend.decode_clean(endpoints.clean))
        corrected = roundtrip if bridge is None else bridge(roundtrip.clone(),timestep)
        metrics = projection_metrics(endpoints.clean,roundtrip,corrected if bridge is not None else None)
        state = endpoints.reconstruct_next(clean=corrected)
        if not bool(torch.isfinite(state).all()): raise ValueError('Non-finite C/D state at step {}'.format(index))
        steps.append(dict(step=index,timestep=float(timestep),**endpoints.schedule_stats(),**metrics))
    if not torch.equal(timesteps,backend.timesteps) or digest!=conditioning_digest(conditioning):
        raise AssertionError('C/D schedule or conditioning changed')
    return BridgeTrajectoryResult(backend.decode_native_canvas(state),steps,evaluations)
