"""A common final-RGB boundary gradient diagnostic (not a quality score)."""
import numpy as np


def boundary_gradient_metrics(rgb, patches, nearby_radius=8):
    """Mean |RGB[b]-RGB[b-1]| at internal patch starts/ends and nearby lines.

    Input is HWC displayed RGB in [0,1], identically decoded from saved PNGs.
    Lines, then orientations, contribute equally. Nearby controls are the
    +/-1..radius gradient lines excluding all exact patch boundaries. A natural
    scene edge can raise the score: interpret together with the actual images.
    """
    rgb=np.asarray(rgb,dtype=np.float64)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<2:
        raise ValueError('Expected HWC RGB image')
    if nearby_radius<1: raise ValueError('nearby_radius must be positive')
    result={};all_boundary=[];all_nearby=[]
    for axis,label,key in ((1,'vertical','x'),(0,'horizontal','y')):
        length=rgb.shape[axis]
        lines=sorted({p[key]+offset for p in patches for offset in (0,p['size']) if 0<p[key]+offset<length})
        gradient=np.abs(np.diff(rgb,axis=axis))
        line_means=gradient.mean(axis=(0,2) if axis==1 else (1,2))
        near=sorted({b+d for b in lines for d in range(-nearby_radius,nearby_radius+1)
                     if d and 0<b+d<length and b+d not in lines})
        boundary=float(line_means[np.asarray(lines)-1].mean()) if lines else None
        nearby=float(line_means[np.asarray(near)-1].mean()) if near else None
        result[label]=dict(boundary_lines=lines,nearby_lines=near,boundary_gradient=boundary,nearby_gradient=nearby)
        if boundary is not None and nearby is not None:
            all_boundary.append(boundary);all_nearby.append(nearby)
    boundary=float(np.mean(all_boundary)) if all_boundary else None
    nearby=float(np.mean(all_nearby)) if all_nearby else None
    result.update(boundary_gradient=boundary,nearby_gradient=nearby,
                  boundary_excess=boundary-nearby if boundary is not None else None,
                  boundary_to_nearby_ratio=boundary/max(nearby,1e-12) if boundary is not None else None,
                  ratio_epsilon=1e-12,nearby_radius=nearby_radius,rgb_range='displayed PNG RGB [0,1]')
    return result
