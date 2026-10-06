"""Full feature dimension retained even when panorama counts make covariance singular."""
import numpy as np
from scipy import linalg

def frechet(x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if min(len(x),len(y))<2:raise ValueError('FID needs >=2 samples in each collection')
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1]:raise ValueError('Feature shape mismatch')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite features')
    mx,my=x.mean(0),y.mean(0);a=(x-mx)/np.sqrt(len(x)-1);b=(y-my)/np.sqrt(len(y)-1)
    # Nonzero singular values of a@b.T give exactly Tr(sqrt(Cx Cy)); no feature reduction.
    if len(x)*len(y)<=3000000:
        cross=a@b.T;root=float(linalg.svdvals(cross).sum())
    else:
        if len(a)>len(b):a,b=b,a
        covariance=b.T@b
        small=a@covariance@a.T;small=(small+small.T)*.5
        ev=np.linalg.eigvalsh(small)
        tolerance=1e-8*max(1.,np.max(np.abs(ev)))
        if ev.min() < -tolerance:raise ArithmeticError('Non-PSD numerical covariance product')
        root=float(np.sqrt(np.maximum(ev,0)).sum())
    value=float(np.sum((mx-my)**2)+np.sum(a*a)+np.sum(b*b)-2*root)
    if value < -1e-7:raise ArithmeticError('Negative FID beyond roundoff: '+str(value))
    return max(value,0.)

def kid(x,y,subsets=100,subset_size=1000,seed=2020):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    m=min(subset_size,len(x),len(y))
    if m<2:raise ValueError('Unbiased KID requires subset size >=2')
    rng=np.random.RandomState(seed);out=[];d=x.shape[1]
    for _ in range(subsets):
        a=x[rng.choice(len(x),m,replace=False)];b=y[rng.choice(len(y),m,replace=False)]
        xx=(a@a.T/d+1)**3;yy=(b@b.T/d+1)**3;xy=(a@b.T/d+1)**3
        out.append(float((xx.sum()-np.trace(xx)+yy.sum()-np.trace(yy))/(m*(m-1))-2*xy.mean()))
    return dict(mean=float(np.mean(out)),subset_std=float(np.std(out)),subset_size=m,subsets=subsets)

def inception_score(logits):
    import torch
    from torch_fidelity.metric_isc import isc_features_to_metric
    if len(logits)<2:raise ValueError('Collection too small for IS')
    return isc_features_to_metric(torch.from_numpy(np.asarray(logits)),splits=1,shuffle=False,rng_seed=2020)['inception_score_mean']
