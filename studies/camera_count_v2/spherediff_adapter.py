"""Camera-only conversion to the pinned official coordinate conventions.

For canonical (yaw=a,pitch=b), inject theta=wrap(pi-a), phi=-b.
Native extraction uses R(theta,phi)^T [x,y,-1]. Final ERP pasting
uses theta_paste=pi-theta and ERP world [x,y,-z]. Both therefore
produce canonical forward/right/up. world_to_perspective's horizontal
coordinate is intentionally reversed internally; the original sampler's
reordering is left unchanged.

Polar yaw is encoded using positive cos(pi/2) from FP64 construction
(~6e-17), which remains nonzero in FP32 and BF16. Canonical yaw is stored
explicitly, never recovered from a rounded or deduplicated pole direction.
The tests verify full axes after both native and final-paste casts.
"""
import contextlib
import importlib.util
import math
from functools import lru_cache
import torch
from studies.original_spherediff.common import SPHERE
from .common import digest,sha

FACTORY='horizontal_and_vertical_view_dirs_v3_fov_xy_dense_equator'
@lru_cache(None)
def reference_geometry():
    p=SPHERE/'pipelines_ours/spherical_functions.py'
    spec=importlib.util.spec_from_file_location('camera_count_original_geometry',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.SphericalFunctions
def directions(cameras):
    angles=torch.tensor([[(math.pi-c.yaw+math.pi)%(2*math.pi)-math.pi,-c.pitch] for c in cameras],dtype=torch.float64)
    assert all(c.roll==0 and c.fov_x==c.fov_y==80. for c in cameras)
    d=reference_geometry().spherical_to_cartesian(angles[:,0],angles[:,1]).float()
    assert len(torch.unique(d,dim=0))==len(cameras), 'Pole orientation lost'
    return d
def tensor_hash(t):
    import hashlib
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
@contextlib.contextmanager
def override(cameras=None, *, sf=None, raw_directions=None):
    sf=sf or reference_geometry()
    if cameras is None and raw_directions is None:
        yield dict(factory_calls=0);return
    original=sf.__dict__[FACTORY]
    data=directions(cameras) if raw_directions is None else raw_directions.clone()
    audit=dict(factory_calls=0,num_cameras=len(data),runtime_fp32_sha256=tensor_hash(data),
               runtime_bf16_sha256=tensor_hash(data.to(torch.bfloat16)))
    def factory(*args,**kwargs):
        assert not args and not kwargs, 'Unexpected override arguments'
        audit['factory_calls']+=1
        return data.clone()
    setattr(sf,FACTORY,staticmethod(factory))
    try:yield audit
    finally:setattr(sf,FACTORY,original)

def physical_axes(view_dirs,*,native=False,sf=None):
    sf=sf or reference_geometry()
    theta,phi=sf.cartesian_to_spherical(view_dirs)
    if native:
        r=sf.rotation_matrix(theta,phi).float()
        # Raster x runs +tan to -tan, camera forward is -Z.
        return torch.stack((-r[:,0,:],r[:,1,:],-r[:,2,:]),dim=-1)
    theta=math.pi-theta
    theta=torch.where(theta>math.pi,theta-2*math.pi,theta)
    r=sf.rotation_matrix(theta,phi).float()
    q=r.new_tensor([1.,1.,-1.])
    return torch.stack((r[:,0,:]*q,r[:,1,:]*q,-r[:,2,:]*q),dim=-1)

def geometry_audit(cameras,device='cpu'):
    sf=reference_geometry();d=directions(cameras).to(device)
    want=torch.stack([c.rotation(torch.device(device)) for c in cameras])
    errors={}
    for name,v,native in [('fp32-native',d,True),('bf16-native',d.bfloat16(),True),
                          ('fp32-paste',d,False),('bf16-to-fp32-paste',d.bfloat16().float(),False)]:
        actual=physical_axes(v,native=native)
        norm=actual/actual.norm(dim=1,keepdim=True)
        wn=want/want.norm(dim=1,keepdim=True)
        e=torch.rad2deg(torch.acos((norm*wn).sum(dim=1).clamp(-1,1)))
        # Include corner rays, not just centers.
        corners=want.new_tensor([[-1.,-1.,1.],[-1.,1.,1.],[1.,-1.,1.],[1.,1.,1.]])
        corners[:,:2]*=math.tan(math.radians(40))
        a=torch.einsum('nij,kj->nki',actual,corners);b=torch.einsum('nij,kj->nki',want,corners)
        a=a/a.norm(dim=-1,keepdim=True);b=b/b.norm(dim=-1,keepdim=True)
        ce=torch.rad2deg(torch.acos((a*b).sum(-1).clamp(-1,1)))
        errors[name]=dict(axes_max_degrees=float(e.max()),corners_max_degrees=float(ce.max()))
        assert float(e.max())<.9 and float(ce.max())<.9,(name,errors[name])
    # Original semantic anchors: phi -90 means north after physical conversion.
    theta=torch.tensor([math.radians(t) for p in (-90,-10,0,10,90) for t in (0,90,180,270)],device=device,dtype=torch.bfloat16)
    phi=torch.tensor([math.radians(p) for p in (-90,-10,0,10,90) for t in (0,90,180,270)],device=device,dtype=torch.bfloat16)
    anchors=sf.spherical_to_cartesian(theta,phi)
    slots,_=sf.get_prompt_indices(d.bfloat16(),anchors,[(80,80)]*20)
    from studies.camera_patching.cameras import routing
    dp=routing(cameras,['north','upper','equator','lower','south'])
    mismatches=[i for i,(a,b) in enumerate(zip(slots.tolist(),dp['indices'])) if a//4!=b//4]
    # Preserve both systems' original ties; report all differences.
    return dict(passed=True,conversion='theta=wrap(pi-yaw),phi=-pitch; FP64 positive polar cosine',
        adapter_sha256=sha(__file__),runtime_fp32_sha256=tensor_hash(d),runtime_bf16_sha256=tensor_hash(d.bfloat16()),
        errors=errors,spherediff_prompt_slots=slots.tolist(),diffpano_prompt_slots=dp['indices'],
        spherediff_semantic_band_counts={str(j):sum(s//4==j for s in slots.tolist()) for j in range(5)},
        semantic_band_disagreements=mismatches,
        tie_policy='original SphereDiff torch.topk versus DiffPano argmax; no reassignment')
