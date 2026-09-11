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


@dataclass
class ThreeWayTrajectoryResult:
    finals: dict
    initial_epsilon: torch.Tensor
    steps: list
    model_evaluations: dict
    fairness: dict


def difference_stats(a, b, prefix):
    difference = (a-b).float()
    return {prefix+'_mae': float(difference.abs().mean()),
            prefix+'_rmse': float(difference.square().mean().sqrt()),
            prefix+'_max_abs': float(difference.abs().max())}


def full_state_stats(value, prefix):
    return {**state_stats(value, prefix), prefix+'_min': float(value.min()),
            prefix+'_max': float(value.max())}


def conditioning_digest(value):
    """Hash actual conditioning tensors, including masks and negative embeddings."""
    import hashlib
    from dataclasses import fields, is_dataclass
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((tuple(item.shape), item.dtype)).encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            for part in item: visit(part)
        elif is_dataclass(item):
            for field in fields(item): visit(getattr(item, field.name))
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


@torch.no_grad()
def compare_three_way(backend, conditioning, epsilon):
    """Three independent native-space states; native one-step audit adds no forward.

    A and B use their existing transition implementations unchanged. C extracts
    both endpoints from one prediction, then reconstructs the scheduled next state.
    Guided counters are incremented in the actual backend prediction entrypoints.
    """
    timesteps = backend.timesteps.clone()
    if len(timesteps) == 0:
        raise ValueError('Empty trajectory schedule')
    original = epsilon.detach().clone()
    native = backend.initialize_native_state(original.clone())
    fixed_initial = backend.make_initial_noisy_state(original.clone(), timesteps[0])
    implied = backend.initialize_native_state(original.clone())
    if not (torch.equal(native, fixed_initial) and torch.equal(native, implied)):
        raise ValueError('Three trajectories must start from bit-identical states')
    condition_hash = conditioning_digest(conditioning)
    scheduler = getattr(getattr(backend, 'pipeline', None), 'scheduler', None)
    watched = [(backend, 'timesteps', timesteps)]
    for obj, names in ((scheduler, ('sigmas', 'alphas_cumprod')),
                       (getattr(backend, 'solver', None), ('schedule',))):
        for name in names:
            value = getattr(obj, name, None)
            if isinstance(value, torch.Tensor): watched.append((obj, name, value.clone()))
    attributes = ('guidance_scale', 'true_cfg_scale', 'cfg_scale', 'interval_guidance',
                  'dtype', '_view_size', 'checkpoint_path', 'official_commit')
    settings = {name: getattr(backend, name, None) for name in attributes}
    counts = {'native': 0, 'fixed_initial_noise': 0, 'model_implied_endpoint': 0}
    def evaluated(name, operation):
        before = getattr(backend, 'guided_prediction_count', 0)
        result = operation()
        delta = getattr(backend, 'guided_prediction_count', 0) - before
        if delta != 1:
            raise AssertionError(f'{name} used {delta} guided predictions; expected exactly one')
        counts[name] += delta
        return result
    records, clean = [], None
    for index, timestep in enumerate(timesteps):
        previous_native = native.clone()
        native, endpoints = evaluated('native', lambda: backend.native_step_with_endpoints(
            previous_native, timestep, conditioning))
        # The same prediction used by A supplies all three one-step proposals.
        native_implied = endpoints.reconstruct_next()
        native_fixed = endpoints.reconstruct_next(original)
        record = {'step': index, 'timestep': float(timestep), **endpoints.schedule_stats(),
                  **difference_stats(native, native_implied, 'native_vs_implied'),
                  **difference_stats(native, native_fixed, 'native_vs_fixed')}
        if scheduler is not None: reset_scheduler_step_state(scheduler)
        fixed_noisy = (fixed_initial.clone() if index == 0
                       else backend.add_fixed_noise(clean, original.clone(), timestep))
        clean = evaluated('fixed_initial_noise', lambda: backend.predict_clean_native(
            fixed_noisy, timestep, conditioning))
        if scheduler is not None: reset_scheduler_step_state(scheduler)
        implied_endpoints = evaluated('model_implied_endpoint', lambda: backend.predict_clean_and_endpoint(
            implied, timestep, conditioning))
        implied = implied_endpoints.reconstruct_next()
        for prefix, state in (('native_state', native), ('fixed_noisy_state', fixed_noisy),
                              ('fixed_clean_state', clean), ('implied_state', implied)):
            if not bool(torch.isfinite(state).all()):
                raise ValueError(f'Non-finite {prefix} at step {index}')
            record.update(full_state_stats(state, prefix))
        record.update(difference_stats(native, implied, 'trajectory_native_vs_implied'))
        records.append(record)
        if any(not torch.equal(getattr(obj, name).to(device=snapshot.device), snapshot) for obj, name, snapshot in watched):
            raise AssertionError('Shared schedule mutated during comparison')
        if any(getattr(backend, name, None) != value for name, value in settings.items()):
            raise AssertionError('Backend guidance, precision, checkpoint or resolution changed')
    if not torch.equal(epsilon, original) or condition_hash != conditioning_digest(conditioning):
        raise AssertionError('Initial Gaussian or shared conditioning was mutated')
    if any(value != len(timesteps) for value in counts.values()):
        raise AssertionError('Unequal model evaluation counts')
    # Exactly one endpoint decode per trajectory; B retains its existing final x0.
    finals = {'native': backend.decode_native_canvas(native),
              'fixed_initial_noise': backend.decode_clean(clean),
              'model_implied_endpoint': backend.decode_native_canvas(implied)}
    for name, value in finals.items():
        if value.shape != finals['native'].shape or not bool(torch.isfinite(value).all()):
            raise AssertionError(f'Invalid decoded endpoint for {name}')
    records[-1].update(difference_stats(finals['native'], finals['model_implied_endpoint'], 'final_rgb_native_vs_implied'),
                       **difference_stats(finals['native'], finals['fixed_initial_noise'], 'final_rgb_native_vs_fixed'))
    fairness = dict(same_initialized_state_bit_exact=True, epsilon_unchanged=True,
                    schedule_unchanged=True, conditioning_unchanged=True,
                    same_backend_checkpoint_precision_guidance_resolution=True,
                    equal_guided_prediction_counts=True, conditioning_sha256=condition_hash)
    return ThreeWayTrajectoryResult(finals, original, records, counts, fairness)
