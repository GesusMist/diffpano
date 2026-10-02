"""Chunked geometric diagnostics. Small positive weights are covered."""
import math
import numpy as np
import torch
from .cameras import GOLDEN_ANGLE, angular_hash

def target_rays(kind,start,stop,height,width,probes,device):
    i=torch.arange(start,stop,device=device)
    if kind in ('erp','cea'):
        y=torch.div(i,width,rounding_mode='floor').float();x=(i%width).float()
        lon=((x+.5)/width-.5)*(2*math.pi)
        if kind=='erp':
            lat=(.5-(y+.5)/height)*math.pi;z=lat.sin();radius=lat.cos()
        else:
            z=1-2*(y+.5)/height;radius=(1-z.square()).clamp_min(0).sqrt()
    elif kind=='probes':
        # Double angular construction avoids accumulated FP32 phase error.
        z=(1-2*(i.double()+.5)/probes).float()
        lon=torch.remainder(i.double()*GOLDEN_ANGLE+.37+math.pi,2*math.pi).float()-math.pi
        radius=(1-z.square()).clamp_min(0).sqrt()
    elif kind=='poles':
        return torch.tensor([[0.,1.,0.],[0.,-1.,0.]],device=device)[start:stop]
    else: raise ValueError(kind)
    return torch.stack((lon.sin()*radius,z,lon.cos()*radius),-1)

def ray_statistics(cameras,rays):
    """Only M × chunk rays are materialized; never M × full sphere."""
    rotation=torch.stack([c.rotation(rays.device) for c in cameras])
    xyz=torch.einsum('rj,mji->rmi',rays,rotation)
    x,y,z=xyz.unbind(-1);safe=torch.where(z.abs()>1e-12,z,torch.ones_like(z))
    tx=rays.new_tensor([math.tan(math.radians(c.fov_x)/2) for c in cameras])
    ty=rays.new_tensor([math.tan(math.radians(c.fov_y)/2) for c in cameras])
    xx=x/safe/tx;yy=y/safe/ty
    valid=(z>0)&(xx.abs()<=1)&(yy.abs()<=1)
    weight=torch.exp(-torch.sqrt(xx.square()+yy.square())/.1)*valid
    distance=torch.rad2deg(torch.acos(z.max(dim=1).values.clamp(-1,1)))
    return valid.sum(1),weight.sum(1),weight.max(1).values,distance

def percentiles(a):
    return {k:float(v) for k,v in zip(('min','p01','median','p99','max'),np.percentile(a,[0,1,50,99,100]))}

@torch.no_grad()
def analyze_camera_cover(cameras,*,height=2048,width=4096,probe_count=1000003,chunk_size=32768,device='cpu',progress=None):
    if not cameras or min(height,width,probe_count,chunk_size)<1: raise ValueError('Positive sizes and cameras required')
    result=dict(num_cameras=len(cameras),angular_geometry_sha256=angular_hash(cameras),domains={})
    for kind,total in [('erp',height*width),('cea',height*width),('probes',probe_count),('poles',2)]:
        values=[np.empty(total,dtype=np.float32) for _ in range(4)];examples=[]
        for start in range(0,total,chunk_size):
            stop=min(total,start+chunk_size)
            rays=target_rays(kind,start,stop,height,width,probe_count,device)
            stats=ray_statistics(cameras,rays)
            for a,t in zip(values,stats):a[start:stop]=t.cpu().numpy()
            if len(examples)<8:
                holes=torch.nonzero(stats[0]==0).flatten()[:8-len(examples)]
                examples.extend(dict(index=start+int(j),ray=rays[j].cpu().tolist()) for j in holes)
        counts,total_w,max_w,dist=values
        uncovered=int(np.count_nonzero(counts==0))
        result['domains'][kind]=dict(samples=total,uncovered=uncovered,uncovered_fraction=uncovered/total,
            contributor_count=percentiles(counts),total_center_weight=percentiles(total_w),
            maximum_center_weight=percentiles(max_w),nearest_center_degrees=percentiles(dist),uncovered_examples=examples,
            finite=all(bool(np.isfinite(a).all()) for a in values))
        if progress:progress(kind,result['domains'][kind])
        del values,counts,total_w,max_w,dist
    result['passed']=all(x['uncovered']==0 and x['finite'] for x in result['domains'].values())
    result['weight_convention']='analytic exp(-sqrt(x_norm^2+y_norm^2)/0.1); production bilinear raster weights unchanged'
    return result
