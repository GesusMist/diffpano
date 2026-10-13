"""Streaming ERP-grid gradient fusion, independent of models and trajectories.

D uses pixel-unit forward differences: x wraps in longitude; y has H-1 edges.
This is not a rotation-invariant spherical objective. No latitude reweighting,
clamping, latent processing, autograd guidance, or missing-edge constraints.
"""
from dataclasses import dataclass
import math
import time
from typing import Optional
import torch
from diffpano.fusion import RGBFusionAccumulator
from diffpano.projection import ERPContribution


@dataclass(frozen=True)
class GradientSettings:
    mode: str = 'rgb'
    lambda_color: float = 0.1
    max_iterations: int = 200
    relative_tolerance: float = 1e-5
    absolute_tolerance: float = 1e-7
    reference_mode: str = 'screened'
    coarse_rows: int = 16
    coarse_cols: int = 32
    coarse_eta: float = 0.1
    constrained_max_iterations: int = 400

    def __post_init__(self):
        if self.mode not in ('rgb', 'poisson_mean', 'poisson_select', 'poisson_max'):
            raise ValueError('Unsupported gradient fusion mode: ' + self.mode)
        if self.reference_mode not in ('screened', 'global_mean', 'single_pixel', 'coarse_color'):
            raise ValueError('Unsupported Poisson reference mode')
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (self.coarse_rows, self.coarse_cols, self.constrained_max_iterations)):
            raise ValueError('Positive coarse dimensions and constrained iteration budget required')
        if not math.isfinite(self.coarse_eta) or self.coarse_eta <= 0:
            raise ValueError('Positive finite coarse eta required')
        if not math.isfinite(self.lambda_color) or self.lambda_color <= 0:
            raise ValueError('lambda_color must be finite and positive')
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) or self.max_iterations < 0:
            raise ValueError('max_iterations must be a nonnegative integer')
        if any(not math.isfinite(v) or v < 0 for v in (self.relative_tolerance, self.absolute_tolerance)):
            raise ValueError('Solver tolerances must be finite and nonnegative')


class GradientSolveError(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def edges(image):
    """D I: (B,C,H,W) horizontal and (B,C,H-1,W) vertical differences."""
    return torch.roll(image, -1, -1) - image, image[..., 1:, :] - image[..., :-1, :]


def adjoint(horizontal, vertical):
    """D^T: each oriented edge subtracts at its tail and adds at its head."""
    if horizontal.ndim != 4 or vertical.shape != (*horizontal.shape[:-2], horizontal.shape[-2]-1, horizontal.shape[-1]):
        raise ValueError('Incompatible edge shapes')
    out = torch.roll(horizontal, 1, -1) - horizontal
    out[..., :-1, :] -= vertical
    out[..., 1:, :] += vertical
    return out


def apply_system(image, support, lambda_color):
    dx, dy = edges(image)
    return lambda_color * image + adjoint(dx * support[0], dy * support[1])


def jacobi_diagonal(reference, support, lambda_color):
    mx, my = [m.to(reference.dtype) for m in support]
    diagonal = torch.full_like(mx, lambda_color)
    if reference.shape[-1] > 1:
        diagonal += mx + torch.roll(mx, 1, -1)
    diagonal[..., :-1, :] += my
    diagonal[..., 1:, :] += my
    return diagonal


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def _reconstruct_screened(reference, guidance, support, settings, *, dtype=torch.float32):
    """Solve (lambda I + D^T M D) u = D^T M (g - D R), I = R+u.

    Float64 is an explicit tiny-test/reference option. Production uses FP32,
    including reductions, outside autocast. All work stays on the input device.
    """
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('Only FP32 production or FP64 reference arithmetic is supported')
    if settings.mode == 'rgb':
        raise ValueError('RGB mode must delegate directly; no solve is needed')
    device = reference.device
    with torch.autocast(device_type=device.type, enabled=False):
        _sync(device); start = time.perf_counter()
        ref = reference.to(dtype=dtype)
        if ref.ndim != 4 or ref.shape[1] != 3 or min(ref.shape[-2:]) < 1:
            raise ValueError('Expected reference [B,3,H,W]')
        expected = (ref.shape, (*ref.shape[:-2], ref.shape[-2]-1, ref.shape[-1]))
        if len(guidance) != 2 or len(support) != 2:
            raise ValueError('Two edge directions required')
        g = []; masks = []
        for raw, mask, shape in zip(guidance, support, expected):
            if raw.shape != shape or mask.shape != (shape[0], 1, *shape[-2:]):
                raise ValueError('Invalid guidance/support shape')
            if raw.device != device or mask.device != device:
                raise ValueError('Guidance/support must be on reference device')
            if not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError('Edge support must be binary')
            m = mask.bool()
            value = torch.where(m, raw.to(dtype), 0.)
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Nonfinite guidance on supported edges')
            g.append(value); masks.append(m)
        if not bool(torch.isfinite(ref).all()):
            raise ValueError('Nonfinite RGB reference')
        dr = edges(ref)
        mismatch = [torch.where(m, target - source, 0.) for m, target, source in zip(masks, g, dr)]
        rhs = adjoint(*mismatch)
        del dr, mismatch
        norm_b = float(torch.linalg.vector_norm(rhs))
        threshold = settings.absolute_tolerance + settings.relative_tolerance * norm_b
        u = torch.zeros_like(ref)
        residual = rhs.clone()
        diagonal = jacobi_diagonal(ref, masks, settings.lambda_color)
        z = residual / diagonal
        direction = z.clone()
        rz = torch.sum(residual * z)
        true_norm = norm_b
        converged = math.isfinite(norm_b) and norm_b <= threshold
        iterations = 0
        failure = None
        if not math.isfinite(norm_b):
            failure = 'nonfinite right-hand side'
        while not converged and failure is None and iterations < settings.max_iterations:
            ap = apply_system(direction, masks, settings.lambda_color)
            pap = torch.sum(direction * ap)
            if not bool(torch.isfinite(pap)) or float(pap) <= 0 or not bool(torch.isfinite(rz)):
                failure = 'nonfinite or nonpositive PCG curvature'; break
            alpha = rz / pap
            u.add_(direction * alpha)
            residual.sub_(ap * alpha)
            iterations += 1
            candidate = float(torch.linalg.vector_norm(residual)) <= threshold
            if candidate or iterations == settings.max_iterations or iterations % 25 == 0:
                true = rhs - apply_system(u, masks, settings.lambda_color)
                true_norm = float(torch.linalg.vector_norm(true))
                if not math.isfinite(true_norm):
                    failure = 'nonfinite true residual'; break
                converged = true_norm <= threshold
                if converged:break
                if candidate:
                    # Roundoff defeated the recursive residual: restart from the true one.
                    residual = true
                    z = residual / diagonal
                    rz = torch.sum(residual * z)
                    direction = z.clone()
                    continue
            z = residual / diagonal
            next_rz = torch.sum(residual * z)
            direction.mul_(next_rz / rz).add_(z)
            rz = next_rz
        true_norm = float(torch.linalg.vector_norm(apply_system(u, masks, settings.lambda_color) - rhs))
        converged = math.isfinite(true_norm) and true_norm <= threshold and failure is None
        diagnostics = dict(converged=converged, iterations=iterations, initial_true_residual=norm_b,
                           final_true_residual=true_norm, stopping_threshold=threshold,
                           dtype=str(dtype), solver='matrix-free Jacobi PCG', lambda_color=settings.lambda_color)
        if not converged:
            _sync(device)
            diagnostics.update(solve_seconds=time.perf_counter()-start, failure=failure or 'iteration budget exhausted')
            raise GradientSolveError('Gradient reconstruction failed: ' + diagnostics['failure'], diagnostics)
        image = ref + u
        if not bool(torch.isfinite(image).all()):
            diagnostics['converged'] = False
            raise GradientSolveError('Nonfinite reconstructed image', diagnostics)
        def objective(value):
            grad = edges(value)
            return float(settings.lambda_color * torch.sum((value-ref).square()) +
                         sum(torch.sum(torch.where(m, d-target, 0.).square()) for m, d, target in zip(masks, grad, g)))
        diagnostics.update(objective_reference=objective(ref), objective=objective(image),
            correction_rms=float(u.square().mean().sqrt()), correction_mae=float(u.abs().mean()),
            correction_max=float(u.abs().max()), mean_color_drift=(image-ref).mean((0,2,3)).tolist(),
            output_min=float(image.min()), output_max=float(image.max()),
            out_of_range_fraction=float(((image < -1) | (image > 1)).float().mean()),
            supported_edges=[int(m.sum()) for m in masks])
        _sync(device); diagnostics['solve_seconds'] = time.perf_counter()-start
        return image, diagnostics


class GuidanceAccumulator:
    """O(BCHW) streaming statistics. Stable integer IDs define tie order.

    `both` stores enough information for offline replay, without storing views.
    Geometric ownership is independent of RGB; the study wrapper caches its map.
    """
    def __init__(self, reference, mode):
        if mode not in ('poisson_mean', 'poisson_select', 'poisson_max', 'both'):
            raise ValueError('Unsupported guidance mode')
        self.mode = mode
        shapes = [reference.shape, (*reference.shape[:-2], reference.shape[-2]-1, reference.shape[-1])]
        self.den = [torch.zeros((s[0],1,*s[-2:]), device=reference.device, dtype=torch.float32) for s in shapes]
        self.num = [torch.zeros(s, device=reference.device, dtype=torch.float32) for s in shapes] if mode in ('poisson_mean', 'both') else None
        self.best = [torch.zeros_like(d) for d in self.den] if mode != 'poisson_mean' else None
        self.selected = [torch.zeros(s, device=reference.device, dtype=torch.float32) for s in shapes] if self.best is not None else None
        self.owners = [torch.full_like(d, torch.iinfo(torch.int32).max, dtype=torch.int32) for d in self.den] if self.best is not None else None

    @torch.no_grad()
    def accumulate(self, contribution, camera_id):
        if isinstance(camera_id, bool) or not isinstance(camera_id, int) or not 0 <= camera_id < torch.iinfo(torch.int32).max:
            raise ValueError('Stable nonnegative integer camera ID required')
        rgb, mask, weight = contribution.rgb.float(), contribution.valid_mask.float(), contribution.weight.float()
        if mask.shape != self.den[0].shape or weight.shape != mask.shape or rgb.shape != (mask.shape[0],3,*mask.shape[-2:]):
            raise ValueError('Invalid contribution shapes')
        if not bool(torch.isfinite(mask).all()) or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError('Validity mask must be finite and binary')
        valid = mask.bool()
        rgb = torch.where(valid, rgb, 0.)
        weight = torch.where(valid, weight, 0.)
        if not bool(torch.isfinite(rgb).all()) or not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
            raise ValueError('Nonfinite RGB/weight or negative confidence on valid support')
        a = mask * weight
        confidence = (torch.minimum(a, torch.roll(a,-1,-1)), torch.minimum(a[...,:-1,:], a[...,1:,:]))
        diffs = edges(rgb)
        for axis, (s, d) in enumerate(zip(confidence, diffs)):
            d = torch.where(s > 0, d, 0.)
            self.den[axis].add_(s)
            if self.num is not None:self.num[axis].add_(s*d)
            if self.best is not None:
                rank = d.square().sum(1, keepdim=True) if self.mode == 'poisson_max' else s
                choose = (s > 0) & ((rank > self.best[axis]) | ((rank == self.best[axis]) & (camera_id < self.owners[axis])))
                self.selected[axis] = torch.where(choose, d, self.selected[axis])
                self.owners[axis] = torch.where(choose, camera_id, self.owners[axis])
                self.best[axis] = torch.maximum(self.best[axis], rank)

    def guidance(self, mode=None):
        mode = mode or self.mode
        support = tuple(d > 0 for d in self.den)
        if mode == 'poisson_mean' and self.num is not None:
            return tuple(n / torch.where(m,d,torch.ones_like(d)) for n,d,m in zip(self.num,self.den,support)), support
        if mode in ('poisson_select', 'poisson_max') and self.selected is not None:
            return tuple(self.selected), support
        raise ValueError('Requested guidance was not accumulated')

    def owner_summary(self):
        if self.owners is None:return None
        result = []
        for owner, den in zip(self.owners,self.den):
            valid = den > 0
            dx = (owner != torch.roll(owner,-1,-1)) & valid & torch.roll(valid,-1,-1)
            dy = (owner[...,1:,:] != owner[...,:-1,:]) & valid[...,1:,:] & valid[...,:-1,:]
            result.append(dict(horizontal_boundary_edges=int(dx.sum()),vertical_boundary_edges=int(dy.sum()),supported=int(valid.sum())))
        return result


class GradientFusionAccumulator:
    """Generic accumulator with the exact legacy RGB reference and keep-previous policy."""
    def __init__(self, previous, fusion_config, settings):
        if fusion_config.mode != 'weighted_average' or fusion_config.uncovered_mode != 'keep_previous':
            raise ValueError('Gradient fusion requires the weighted_average keep_previous reference')
        self.settings = settings
        self.reference_accumulator = RGBFusionAccumulator(previous, fusion_config)
        self.statistics = None if settings.mode == 'rgb' else GuidanceAccumulator(previous, settings.mode)
        self.reference = None
        self.diagnostics = None

    def accumulate(self, contribution, camera_id=0):
        if self.settings.mode == 'rgb':
            return self.reference_accumulator.accumulate(contribution)
        with torch.autocast(device_type=self.reference_accumulator.previous.device.type, enabled=False):
            self.statistics.accumulate(contribution, camera_id)
            # Safe invalid-region NaNs without modifying any valid source or baseline reduction.
            valid = contribution.valid_mask.bool()
            safe = ERPContribution(torch.where(valid,contribution.rgb,0.),contribution.valid_mask,
                                   torch.where(valid,contribution.weight,0.))
            self.reference_accumulator.accumulate(safe)

    def finalize(self):
        baseline = self.reference_accumulator.finalize()
        self.reference = baseline.erp_rgb
        if self.settings.mode == 'rgb':return baseline
        guidance, support = self.statistics.guidance()
        rgb, self.diagnostics = reconstruct(self.reference, guidance, support, self.settings)
        from dataclasses import replace
        return replace(baseline, erp_rgb=rgb)


@torch.no_grad()
def reconstruct(reference, guidance, support, settings, *, dtype=torch.float32, anchor=None, connectivity=None):
    """Dispatch independent guidance and reference choices; legacy screened math is unchanged."""
    if settings.reference_mode == 'screened':
        image,diagnostics=_reconstruct_screened(reference, guidance, support, settings, dtype=dtype)
        color=float(settings.lambda_color*(image-reference.to(dtype)).square().sum())
        diagnostics.update(reference_mode='screened',objective_color=color,objective_gradient=diagnostics['objective']-color)
        return image,diagnostics
    from diffpano.poisson_reference import reconstruct_constrained
    return reconstruct_constrained(reference, guidance, support, settings, dtype=dtype, anchor=anchor, connectivity=connectivity)
