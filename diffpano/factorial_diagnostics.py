"""Fixed-camera boundary proxy and one-seed balanced factorial contrasts."""
import itertools
import math
import torch
from diffpano.geometry import erp_world_directions


def camera_boundary_metric(rgb,cameras):
    """Raw RGB adjacent gradients on fixed angular Voronoi camera boundaries.

    This detects boundary-associated texture/exposure change, not semantic
    geometric correctness. Owner maps depend only on the common camera poses.
    No masks are saved. The ERP wrap boundary is measured separately.
    """
    h,w=rgb.shape[-2:];device=rgb.device
    lat=(.5-(torch.arange(h,device=device,dtype=torch.float32)+.5)/h)*math.pi
    rays=erp_world_directions(h,w,device=device)
    best=torch.full((h,w),-2.,device=device);owner=torch.zeros((h,w),dtype=torch.int32,device=device)
    for i,c in enumerate(cameras):
        score=(rays*c.forward().to(device)).sum(-1);better=score>best
        owner=torch.where(better,i,owner);best=torch.maximum(best,score)
    gx=(rgb[...,1:]-rgb[...,:-1]).abs().mean((0,1));gy=(rgb[...,1:,:]-rgb[...,:-1,:]).abs().mean((0,1))
    bx=owner[:,1:]!=owner[:,:-1];by=owner[1:,:]!=owner[:-1,:]
    wx=lat.cos()[:,None].expand_as(gx);wy=((lat[:-1]+lat[1:])/2).cos()[:,None].expand_as(gy)
    def average(boundary):
        mx=bx if boundary else ~bx;my=by if boundary else ~by
        den=(wx*mx).sum()+(wy*my).sum()
        return float(((gx*wx*mx).sum()+(gy*wy*my).sum())/den.clamp_min(1e-12))
    edge=average(True);interior=average(False)
    return dict(boundary_gradient=edge,interior_gradient=interior,ratio=edge/max(interior,1e-12),boundary_edges=int(bx.sum()+by.sum()),
                definition='cosine-area-weighted adjacent raw RGB gradient on fixed nearest-camera-axis Voronoi boundaries; interior comparison; not a perceptual coherence score')


def factorial_contrasts(rows,metrics):
    """Balanced +/-1 contrast: positive-product mean minus negative-product mean."""
    if len(rows)!=16 or len({r['cell'] for r in rows})!=16:raise ValueError('Complete 16-cell backend required')
    effects={}
    terms=[(k,) for k in 'ABCD']+list(itertools.combinations('ABCD',2))
    for term in terms:
        result={}
        for metric in metrics:
            groups={1:[],-1:[]}
            for row in rows:
                sign=math.prod(2*row[k]-1 for k in term);groups[sign].append(row[metric])
            if any(len(v)!=8 for v in groups.values()):raise ValueError('Unbalanced factorial matrix')
            plus=sum(groups[1])/8;minus=sum(groups[-1])/8
            result[metric]=dict(effect=plus-minus,positive_product_mean=plus,negative_product_mean=minus)
        effects['x'.join(term)]=result
    return effects
