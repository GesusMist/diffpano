"""Generic angular layouts, independent of rasters and diffusion RNG streams."""
import math
from numbers import Integral
import torch
from diffpano.camera import PerspectiveCamera
from diffpano.bridge_factorial import angular_geometry, digest
from diffpano.conditioning import camera_prompt_indices, expand_directional_prompts

GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))

def checked_count(n):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError('num_cameras must be a positive integer')
    return int(n)

def fibonacci_camera_cover(view, num_cameras, *, phase=0.0):
    n = checked_count(num_cameras)
    if not math.isfinite(phase): raise ValueError('Finite phase required')
    return tuple(PerspectiveCamera((phase+i*GOLDEN_ANGLE+math.pi)%(2*math.pi)-math.pi,
        math.asin(1-2*(i+.5)/n), 0., view.fov_x, view.fov_y, view.height, view.width) for i in range(n))

def random_uniform_camera_cover(view, num_cameras, *, seed):
    n = checked_count(num_cameras)
    if isinstance(seed,bool) or not isinstance(seed,Integral): raise ValueError('Integer layout seed required')
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    # One row per camera; column 0 is u, column 1 maps to z. FP32 CPU draws.
    pairs = torch.rand((n,2), generator=generator, dtype=torch.float32, device='cpu').tolist()
    return tuple(PerspectiveCamera(2*math.pi*u-math.pi, math.asin(2*v-1), 0.,
        view.fov_x, view.fov_y, view.height, view.width) for u,v in pairs)

def cover(strategy, view, n, *, layout_seed=0, phase=0.0):
    if strategy=='fibonacci': return fibonacci_camera_cover(view,n,phase=phase)
    if strategy=='random': return random_uniform_camera_cover(view,n,seed=layout_seed)
    raise ValueError(strategy)

def stable_ids(strategy,n,*,layout_seed=0):
    n=checked_count(n)
    if strategy not in ('fibonacci','random'): raise ValueError(strategy)
    prefix=f'fibonacci:n{n:03d}' if strategy=='fibonacci' else f'random:n{n:03d}:seed{layout_seed}'
    return [f'{prefix}:{i:03d}' for i in range(n)]

def angular_hash(cameras):
    # JSON distinguishes 80 from 80.0; geometry must not.
    return digest([{k:float(v) for k,v in pose.items()} for pose in angular_geometry(cameras)])

def routing(cameras,lines):
    from collections import Counter
    bank=expand_directional_prompts(lines)
    slots=camera_prompt_indices(cameras,bank.directions).tolist()
    return dict(indices=slots,semantic_band_counts={str(i):sum(s//4==i for s in slots) for i in range(5)})
