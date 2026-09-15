"""Experiment I: preserve the frozen current state after clean consensus."""
import torch


def interpolate_from_current_state(current_state, replacement_clean, current_alpha,
                                   current_sigma, next_alpha, next_sigma, *, flow=False):
    """Use prepared scheduler coefficients, without reading an old endpoint.

    sigma=0 cannot identify a residual. Only an already-consistent clean
    terminal state with next_sigma=0 is accepted; otherwise fail explicitly.
    No coefficient is clamped, shifted, or inferred from a loop index.
    """
    x, clean = current_state, replacement_clean
    if x.shape != clean.shape or x.dtype != clean.dtype or x.device != clean.device:
        raise ValueError('Current state and clean must have identical shape/dtype/device')
    a, s, an, sn = [torch.as_tensor(v, device=x.device, dtype=x.dtype) for v in
                    (current_alpha, current_sigma, next_alpha, next_sigma)]
    if any(v.numel() != 1 or not bool(torch.isfinite(v)) for v in (a,s,an,sn)):
        raise ValueError('Schedule coefficients must be finite scalars')
    if float(s) < 0 or float(sn) < 0:
        raise ValueError('Negative scheduler sigma')
    if flow and (not torch.allclose(a,1-s) or not torch.allclose(an,1-sn)):
        raise ValueError('Flow interpolation requires actual straight-flow coefficients')
    if float(s) == 0:
        if float(sn) != 0 or not torch.allclose(x,a*clean,atol=2e-6,rtol=2e-6):
            raise ValueError('sigma=0 requires a consistent already-clean terminal state')
        return an*clean
    if flow:
        return clean + (sn/s)*(x-clean)
    return an*clean + (sn/s)*(x-a*clean)


def transition_diagnostics(current_state, corrected_clean, pair, next_i):
    """Tensor measurements of both transitions on the SAME I source state.

    The counterfactual G tensors never enter the I trajectory. For both DDIM
    and flow, I-G = -(next_sigma/sigma)*alpha*(corrected-original), up to the
    finite-precision error of the original endpoint decomposition.
    """
    x, c = current_state, corrected_clean
    a,s,an,sn = pair.alpha,pair.sigma,pair.next_alpha,pair.next_sigma
    delta = c-pair.clean
    old_current = a*c+s*pair.endpoint
    next_g = pair.reconstruct_next(clean=c)
    new_endpoint = (x-a*c)/s if float(s) != 0 else torch.zeros_like(c)
    new_current = a*c+s*new_endpoint
    theoretical = -(sn/s)*a*delta if float(s) != 0 else torch.zeros_like(c)
    measured = next_i-next_g
    identity_error = measured-theoretical
    torch.testing.assert_close(new_current,x,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(measured,theoretical,atol=5e-6,rtol=5e-6)
    mae = lambda t: float(t.abs().mean())
    values = dict(clean_consensus_delta_mae=mae(delta),
        g_current_state_mismatch=mae(old_current-x),
        i_current_state_mismatch=mae(new_current-x),
        i_current_state_error_max_abs=float((new_current-x).abs().max()),
        next_state_G_vs_I_mae=mae(measured),
        analytical_next_state_delta_mae=mae(theoretical),
        next_state_identity_error_mae=mae(identity_error),
        next_state_identity_error_max_abs=float(identity_error.abs().max()),
        analytical_g_current_mismatch=mae(a*delta),
        g_current_identity_error_mae=mae(old_current-x-a*delta),
        current_state_std=float(x.std(unbiased=False)),
        original_clean_std=float(pair.clean.std(unbiased=False)))
    for key in ('clean_consensus_delta_mae','g_current_state_mismatch',
                'i_current_state_mismatch','next_state_G_vs_I_mae'):
        for norm in ('current_state_std','original_clean_std'):
            values[key+'_over_'+norm] = values[key]/max(values[norm],1e-12)
    return values
