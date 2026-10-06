"""One canonical physical layout for every authorized family/backend/prompt."""
import math
from dataclasses import asdict
from diffpano.camera import PerspectiveCamera
from diffpano.config import ViewConfig
from studies.camera_patching.cameras import fibonacci_camera_cover, random_uniform_camera_cover, angular_hash, stable_ids
from .common import COUNTS, VERSION, digest

RINGS = (90.,67.5,45.,22.5,0.,-22.5,-45.,-67.5,-90.)
RING_COUNTS = {70:(3,7,8,11,12,11,8,7,3),50:(2,5,6,7,10,7,6,5,2),30:(1,3,4,4,6,4,4,3,1)}
def build(strategy,n,seed=None,view=None):
    if n not in COUNTS:raise ValueError('Only N=70,50,30 are authorized')
    view=view or ViewConfig(height=1024,width=1024,fov_x=80.,fov_y=80.)
    assert view.fov_x==view.fov_y==80.
    if strategy=='old':
        assert sum(RING_COUNTS[n])==n
        return tuple(PerspectiveCamera(-math.pi+2*math.pi*j/m,math.radians(p),0.,80.,80.,view.height,view.width)
                     for p,m in zip(RINGS,RING_COUNTS[n]) for j in range(m))
    if strategy=='fibonacci':return fibonacci_camera_cover(view,n,phase=0.)
    if strategy=='random':
        if seed is None:raise ValueError('Explicit candidate or accepted seed required')
        return random_uniform_camera_cover(view,n,seed=seed)
    raise ValueError(strategy)
def ids(strategy,n,seed=None):
    if strategy=='old':return [f'old-rings-v2:n{n:03d}:{i:03d}' for i in range(n)]
    return stable_ids(strategy,n,layout_seed=seed or 0)
def layout_record(strategy,n,seed=None):
    cams=build(strategy,n,seed);poses=[{k:float(v) for k,v in asdict(c).items() if k not in ('height','width')} for c in cams]
    return dict(version=VERSION,strategy=strategy,num_cameras=n,
        description={'old':'user-specified reduced latitude-ring camera cover; regenerated yaws, not parent thinning',
                     'fibonacci':'existing FibonacciN, phase 0','random':'random_uniform_rejection_full_coverage'}[strategy],
        canonical_convention='+Y up; forward=[cos(pitch)sin(yaw),sin(pitch),cos(pitch)cos(yaw)]',
        ordering='north-to-south rings then increasing yaw' if strategy=='old' else 'original generator row order',
        ring_counts=list(RING_COUNTS[n]) if strategy=='old' else None,
        ring_latitudes=list(RINGS) if strategy=='old' else None,
        fibonacci_phase=0. if strategy=='fibonacci' else None,accepted_seed=seed,
        poses=poses,camera_ids=ids(strategy,n,seed),angular_geometry_sha256=angular_hash(cams),
        camera_identity_sha256=digest(dict(poses=poses,ids=ids(strategy,n,seed))),fixed_during_trajectory=True)
def restore(record,view=None):
    view=view or ViewConfig(height=1024,width=1024,fov_x=80.,fov_y=80.)
    cams=tuple(PerspectiveCamera(**p,height=view.height,width=view.width) for p in record['poses'])
    assert angular_hash(cams)==record['angular_geometry_sha256']
    assert len(cams)==record['num_cameras']
    assert ids(record['strategy'],len(cams),record['accepted_seed'])==record['camera_ids']
    return cams
