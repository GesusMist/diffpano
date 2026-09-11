"""Endpoint reconstruction in native coordinates, without model evaluations."""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from diffpano.pipelines.clean_prediction import ddim_predicted_clean, flow_sigma


@dataclass
class EndpointPrediction:
    clean: torch.Tensor
    endpoint: torch.Tensor
    alpha: torch.Tensor
    sigma: torch.Tensor
    next_alpha: torch.Tensor
    next_sigma: torch.Tensor

    def reconstruct_next(self, endpoint=None, *, clean=None):
        clean = self.clean if clean is None else clean.to(self.clean)
        if clean.shape != self.clean.shape:
            raise ValueError("Replacement clean and original clean shapes differ")
        endpoint = self.endpoint if endpoint is None else endpoint.to(self.clean)
        if endpoint.shape != self.clean.shape:
            raise ValueError('Endpoint and clean shapes differ')
        return self.next_alpha * clean + self.next_sigma * endpoint

    def schedule_stats(self):
        return {name: float(getattr(self, name)) for name in
                ('alpha', 'sigma', 'next_alpha', 'next_sigma')}


@runtime_checkable
class EndpointBackend(Protocol):
    def native_step_with_endpoints(self, state, timestep, conditioning): ...
    def predict_clean_and_endpoint(self, state, timestep, conditioning): ...


def ddim_endpoints(scheduler, state, prediction, timestep, clean=None):
    if scheduler.config.prediction_type != 'epsilon':
        raise ValueError('Endpoint DDIM control requires epsilon prediction')
    if type(scheduler).__name__ != 'DDIMScheduler':
        raise ValueError('Endpoint DDIM control requires DDIMScheduler')
    t = int(timestep)
    if not bool((scheduler.timesteps == t).any()):
        raise ValueError('Unknown DDIM timestep')
    # Match DDIM.step, including its final_alpha_cumprod convention. Eta=0
    # and use_clipped_model_output=False are the existing adapter defaults.
    previous = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    current = scheduler.alphas_cumprod[t].to(state)
    following = (scheduler.alphas_cumprod[previous] if previous >= 0
                 else scheduler.final_alpha_cumprod).to(state)
    clean = ddim_predicted_clean(scheduler, state, prediction, timestep) if clean is None else clean
    return EndpointPrediction(clean, prediction.to(state), current.sqrt(), (1-current).sqrt(),
                              following.sqrt(), (1-following).sqrt())


def flow_bounds(scheduler, timestep, state):
    sigma = flow_sigma(scheduler, timestep, device=state.device, dtype=state.dtype)
    times = scheduler.timesteps.to(state.device)
    t = torch.as_tensor(timestep, device=state.device, dtype=times.dtype)
    if callable(getattr(scheduler, 'index_for_timestep', None)):
        index = int(scheduler.index_for_timestep(t, times))
    else:
        index = int((times-t).abs().argmin())
    # Look up current time first; the following sigma includes the scheduler's
    # actual terminal value. Never derive sigma from a loop index or re-shift it.
    return sigma, scheduler.sigmas[index+1].to(state)


def flow_endpoints(state, velocity, sigma, next_sigma, clean=None):
    sigma, next_sigma = torch.as_tensor(sigma).to(state), torch.as_tensor(next_sigma).to(state)
    velocity = velocity.to(state)
    if state.shape != velocity.shape:
        raise ValueError('Velocity and native state shapes differ')
    clean = state - sigma * velocity if clean is None else clean
    endpoint = state + (1-sigma) * velocity
    return EndpointPrediction(clean, endpoint, 1-sigma, sigma, 1-next_sigma, next_sigma)


def pixel_endpoints(state, clean, sigma, next_sigma):
    sigma, next_sigma = torch.as_tensor(sigma).to(state), torch.as_tensor(next_sigma).to(state)
    if float(sigma) < 0 or float(next_sigma) < 0:
        raise ValueError('Negative flow time')
    if float(sigma) == 0:
        if float(next_sigma) != 0:
            raise ValueError('Cannot infer a noise endpoint from a clean state at sigma=0')
        # Endpoint is unidentifiable but contributes exactly zero at termination.
        endpoint = torch.zeros_like(clean)
    else:
        # Float64 intermediate avoids avoidable cancellation at near-zero times;
        # no clamping of the actual scheduler sigma.
        endpoint = ((state.double() - (1-sigma.double()) * clean.double()) / sigma.double()).to(state)
    return EndpointPrediction(clean, endpoint, 1-sigma, sigma, 1-next_sigma, next_sigma)
