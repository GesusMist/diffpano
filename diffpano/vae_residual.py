"""Training-free LookingGlass Sec. 4.1 residual correction, in raw native units."""
import torch

from diffpano.trajectory import difference_stats


def vae_residual(clean, roundtrip):
    if clean.shape != roundtrip.shape or clean.device != roundtrip.device or clean.dtype != roundtrip.dtype:
        raise ValueError('VAE residual requires identical shape, device and native dtype')
    return clean-roundtrip


def recover_identity(clean, roundtrip):
    residual=vae_residual(clean,roundtrip)
    corrected=roundtrip+residual
    # Floating-point cancellation bound, rather than an assumption of exact zero.
    bound=8*torch.finfo(clean.dtype).eps*(clean.abs()+roundtrip.abs()+1)
    if not bool(torch.isfinite(corrected).all()) or not bool(((corrected-clean).abs()<=bound).all()):
        raise AssertionError('Single-patch VAE residual identity failed')
    return corrected


def residual_metrics(clean, roundtrip, corrected=None):
    residual=vae_residual(clean,roundtrip)
    stats=difference_stats(clean,roundtrip,'roundtrip_residual')
    std=float(clean.std(unbiased=False))
    stats.update(residual_std=float(residual.std(unbiased=False)),
        relative_residual_mae=stats['roundtrip_residual_mae']/max(std,1e-8),normalization_epsilon=1e-8,
        latent_before_mean=float(clean.mean()),latent_before_std=std,
        latent_after_mean=float(roundtrip.mean()),latent_after_std=float(roundtrip.std(unbiased=False)))
    if corrected is not None:stats.update(difference_stats(corrected,clean,'correction_recovery'))
    return stats


@torch.no_grad()
def identity_vae_preflight(backend, native_shape):
    """Real VAE algebra oracle, before any model trajectory/image experiment."""
    rows=[]
    for amplitude in (1.,3.):
        z=torch.randn(native_shape,generator=torch.Generator().manual_seed(19)).to(backend.device)*amplitude
        rt=backend.encode_clean(backend.decode_clean(z))
        corrected=recover_identity(z,rt)
        torch.testing.assert_close(corrected,z,atol=2e-6,rtol=2e-6)
        rows.append(dict(amplitude=amplitude,**residual_metrics(z,rt,corrected)))
    return rows


@torch.no_grad()
def single_patch_residual_trajectory(backend, initial, conditioning, *, correction=False):
    from diffpano.pipelines.base import reset_scheduler_step_state
    from diffpano.trajectory import conditioning_digest
    state=initial.clone().float();times=backend.timesteps.clone();records=[];count=0
    condition_hash=conditioning_digest(conditioning)
    scheduler=getattr(getattr(backend,'pipeline',None),'scheduler',None)
    for index,timestep in enumerate(times):
        if scheduler is not None:reset_scheduler_step_state(scheduler)
        before=getattr(backend,'guided_prediction_count',0)
        pair=backend.predict_clean_and_endpoint(state,timestep,conditioning)
        delta=getattr(backend,'guided_prediction_count',0)-before
        if delta!=1:raise AssertionError('Expected one guided prediction for C/D')
        count+=delta
        # C and D each do exactly one clean decode and one encode per timestep.
        rt=backend.encode_clean(backend.decode_clean(pair.clean))
        clean=recover_identity(pair.clean,rt) if correction else rt
        stats=residual_metrics(pair.clean,rt,clean if correction else None)
        # No geometry, fusion, residual endpoint modification or fixed noise.
        state=pair.reconstruct_next(clean=clean)
        if not bool(torch.isfinite(state).all()):raise ValueError('Non-finite C/D state')
        records.append(dict(step=index,timestep=float(timestep),**pair.schedule_stats(),**stats))
    if not torch.equal(times,backend.timesteps) or condition_hash!=conditioning_digest(conditioning):
        raise AssertionError('C/D schedule or conditioning changed')
    return backend.decode_native_canvas(state),records,count


def fuse_native_residuals(layout, residuals):
    """Temporary uniform native-coordinate canvas; supports any channel count."""
    from diffpano.native_multidiffusion import NativePlanarFusionAccumulator
    if len(residuals) != layout.num_patches:
        raise ValueError('Residual count differs from native patch layout')
    first = residuals[0]
    canvas = first.new_zeros(first.shape[0], first.shape[1], layout.canvas_height, layout.canvas_width)
    accumulator = NativePlanarFusionAccumulator(canvas)
    for patch, residual in zip(layout.patches, residuals):
        if residual.dtype != first.dtype or residual.device != first.device:
            raise ValueError('Residual dtype/device mismatch')
        accumulator.accumulate(residual, patch)
    return accumulator.finalize()
