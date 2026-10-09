"""Constrained pixel-grid Poisson reconstruction, with no full-resolution ridge.

FFT preconditioning uses an even vertical extension to exactly diagonalize the
periodic-longitude/open-latitude full-support Laplacian. Masked operators remain
matrix-free and their connected components are checked explicitly.
"""
import math
import time
import torch
from diffpano.gradient_fusion import edges, adjoint, GradientSolveError, _sync


def project(x):
    return x - x.mean((-2, -1), keepdim=True)


def block_mean(x, rows, cols):
    h, w = x.shape[-2:]
    if h % rows or w % cols:
        raise ValueError('ERP dimensions must be divisible by coarse grid dimensions')
    return x.reshape(*x.shape[:-2], rows, h//rows, cols, w//cols).mean((-3, -1))


def block_adjoint(values, height, width, *, weighted=False):
    """B^T, or B^T diag(n_b) if weighted; exact nonoverlapping partitions."""
    rows, cols = values.shape[-2:]
    if height % rows or width % cols:raise ValueError('Nondivisible block grid')
    bh, bw = height//rows, width//cols
    result = values.repeat_interleave(bh, -2).repeat_interleave(bw, -1)
    return result if weighted else result/(bh*bw)


def coarse_project(x, rows, cols):
    return block_adjoint(block_mean(x, rows, cols), *x.shape[-2:], weighted=True)


def graph_connectivity(support):
    mx, my = support; batch, _, h, w = mx.shape
    if bool(mx.all()) and bool(my.all()):
        return dict(components=[1]*batch, isolated_pixels=[int(h*w==1)]*batch, full_support=True)
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    components=[];isolated=[]
    ids=np.arange(h*w).reshape(h,w)
    for i in range(batch):
        x=mx[i,0].detach().cpu().numpy().astype(bool);y=my[i,0].detach().cpu().numpy().astype(bool)
        a=np.concatenate((ids[x],ids[:-1][y]));b=np.concatenate((np.roll(ids,-1,1)[x],ids[1:][y]))
        nonself=a!=b;a=a[nonself];b=b[nonself]
        graph=coo_matrix((np.ones(len(a),np.uint8),(a,b)),shape=(h*w,h*w)).tocsr()
        n,_=connected_components(graph,directed=False)
        degree=np.bincount(np.concatenate((a,b)),minlength=h*w)
        components.append(int(n));isolated.append(int((degree==0).sum()))
    return dict(components=components,isolated_pixels=isolated,full_support=False)


class ConnectivityCache:
    """Reuse only after an exact support equality check, including batch shape."""
    def __init__(self):self.masks=None;self.result=None
    def check(self,support):
        if self.masks is None or not all(torch.equal(a,b) for a,b in zip(self.masks,support)):
            self.result=graph_connectivity(support);self.masks=tuple(m.clone() for m in support)
        if any(n!=1 for n in self.result['components']):
            raise GradientSolveError('Disconnected supported-edge graph',dict(converged=False,connectivity=self.result))
        return self.result


class SpectralInverse:
    def __init__(self, reference, shift=0.):
        h,w=reference.shape[-2:];device=reference.device;dtype=reference.dtype
        yy=torch.arange(2*h,device=device,dtype=dtype)
        xx=torch.arange(w//2+1,device=device,dtype=dtype)
        eigen=4*torch.sin(math.pi*yy/(2*h)).square()[:,None]+4*torch.sin(math.pi*xx/w).square()[None,:]+shift
        eigen[0,0]=1.;self.inverse=eigen.reciprocal();self.inverse[0,0]=0.
        self.height=h;self.width=w
    def __call__(self, rhs):
        extended=torch.cat((rhs,torch.flip(rhs,[-2])),dim=-2)
        spectrum=torch.fft.rfft2(extended)
        spectrum.mul_(self.inverse)
        return project(torch.fft.irfft2(spectrum,s=(2*self.height,self.width))[...,:self.height,:])


def _pcg(rhs, apply, precondition, budget, atol, rtol):
    rhs=project(rhs);norm=float(torch.linalg.vector_norm(rhs));threshold=atol+rtol*norm
    value=torch.zeros_like(rhs);residual=rhs.clone();z=project(precondition(residual));p=z.clone()
    rz=(residual*z).sum();iterations=0;failure=None
    while float(torch.linalg.vector_norm(residual))>threshold and iterations<budget:
        ap=project(apply(p));pap=(p*ap).sum()
        if not bool(torch.isfinite(pap)) or float(pap)<=0 or not bool(torch.isfinite(rz)):
            failure='nonpositive/nonfinite projected curvature';break
        alpha=rz/pap;value=project(value+alpha*p);residual=project(residual-alpha*ap);iterations+=1
        candidate=float(torch.linalg.vector_norm(residual))<=threshold
        if candidate or iterations%25==0 or iterations==budget:
            true=project(rhs-apply(value))
            if float(torch.linalg.vector_norm(true))<=threshold:residual=true;break
            # Periodic reliable restart avoids optimistic recursive convergence.
            residual=true;z=project(precondition(residual));p=z.clone();rz=(residual*z).sum();continue
        z=project(precondition(residual));next_rz=(residual*z).sum();p=project(z+p*(next_rz/rz));rz=next_rz
    true_norm=float(torch.linalg.vector_norm(project(apply(value)-rhs)))
    converged=math.isfinite(true_norm) and true_norm<=threshold and failure is None
    return value,dict(converged=converged,iterations=iterations,initial_true_residual=norm,final_true_residual=true_norm,
                      stopping_threshold=threshold,failure=None if converged else failure or 'iteration budget exhausted')


@torch.no_grad()
def reconstruct_constrained(reference,guidance,support,settings,*,dtype=torch.float32,anchor=None,connectivity=None):
    if dtype not in (torch.float32,torch.float64):raise ValueError('FP32 production or FP64 validation required')
    if settings.mode=='rgb':raise ValueError('RGB requires no solve')
    device=reference.device
    with torch.autocast(device_type=device.type,enabled=False):
        _sync(device);start=time.perf_counter();ref=reference.to(dtype)
        if ref.ndim!=4 or ref.shape[1]!=3 or not bool(torch.isfinite(ref).all()):raise ValueError('Finite [B,3,H,W] reference required')
        shapes=(ref.shape,(*ref.shape[:-2],ref.shape[-2]-1,ref.shape[-1]));g=[];masks=[]
        if len(guidance)!=2 or len(support)!=2:raise ValueError('Two edge directions required')
        for raw,mask,shape in zip(guidance,support,shapes):
            if raw.shape!=shape or mask.shape!=(shape[0],1,*shape[-2:]) or raw.device!=device or mask.device!=device:raise ValueError('Invalid guidance/support')
            if not bool(((mask==0)|(mask==1)).all()):raise ValueError('Binary support required')
            m=mask.bool();v=torch.where(m,raw.to(dtype),0.)
            if not bool(torch.isfinite(v).all()):raise ValueError('Nonfinite supported guidance')
            masks.append(m);g.append(v)
        graph=(connectivity or ConnectivityCache()).check(masks)
        def lap(x):
            dx,dy=edges(x);return adjoint(dx*masks[0],dy*masks[1])
        mode=settings.reference_mode;eta=settings.coarse_eta;rows=settings.coarse_rows;cols=settings.coarse_cols
        if mode=='coarse_color':
            # Solve a zero-mean correction. B^T diag(n) B is a block-constant
            # orthogonal projection, NOT a pixelwise color attraction.
            dr=edges(ref);rhs=adjoint(*(torch.where(m,t-d,0.) for m,t,d in zip(masks,g,dr)))
            def apply(x):return lap(x)+eta*coarse_project(x,rows,cols)
            inv=SpectralInverse(ref,eta)
        else:
            rhs=adjoint(*g);apply=lap;inv=SpectralInverse(ref)
        solution,diag=_pcg(rhs,apply,inv,settings.constrained_max_iterations,settings.absolute_tolerance,settings.relative_tolerance)
        diag.update(reference_mode=mode,solver='projected PCG with exact reflected FFT preconditioner',dtype=str(dtype),connectivity=graph)
        if not diag['converged']:
            _sync(device);diag['solve_seconds']=time.perf_counter()-start
            raise GradientSolveError('Constrained gradient reconstruction failed: '+diag['failure'],diag)
        if mode=='global_mean':image=solution+ref.mean((-2,-1),keepdim=True)
        elif mode=='single_pixel':
            if anchor is None:raise ValueError('single_pixel requires a current source-prediction anchor')
            y,x=anchor['erp_y'],anchor['erp_x'];color=anchor['color'].to(device=device,dtype=dtype).reshape(ref.shape[0],3,1,1)
            if not (0<=y<ref.shape[-2] and 0<=x<ref.shape[-1]) or not bool(torch.isfinite(color).all()):raise ValueError('Invalid source anchor')
            image=solution+color-solution[...,y:y+1,x:x+1]
            diag['anchor']={k:v for k,v in anchor.items() if k!='color'}
            diag['anchor'].update(c0=color.flatten().tolist(),error=float((image[...,y:y+1,x:x+1]-color).abs().max()))
        elif mode=='coarse_color':image=ref+solution
        else:raise ValueError('Unknown constrained reference mode')
        if not bool(torch.isfinite(image).all()):raise GradientSolveError('Nonfinite output',diag)
        def objectives(value):
            gradient=sum(torch.sum(torch.where(m,d-t,0.).square()) for m,d,t in zip(masks,edges(value),g))
            color=eta*coarse_project(value-ref,rows,cols).square().sum() if mode=='coarse_color' else torch.zeros((),device=device,dtype=dtype)
            return float(gradient),float(color)
        grad,color=objectives(image);oldgrad,oldcolor=objectives(ref);u=image-ref
        # For pure modes this is the actual normal equation. For coarse, its
        # projection is the constrained normal equation (constant Lagrange mode).
        normal=adjoint(*(torch.where(m,d-t,0.) for m,d,t in zip(masks,edges(image),g)))
        if mode=='coarse_color':normal+=eta*coarse_project(u,rows,cols)
        diag.update(output_normal_residual=float(torch.linalg.vector_norm(project(normal))),
            objective_gradient=grad,objective_color=color,objective=grad+color,objective_reference=oldgrad+oldcolor,
            correction_rms=float(u.square().mean().sqrt()),correction_mae=float(u.abs().mean()),correction_max=float(u.abs().max()),
            mean_color_drift=u.mean((-2,-1)).tolist(),mean_constraint_error=float(u.mean((-2,-1)).abs().max()) if mode!='single_pixel' else None,
            output_min=float(image.min()),output_max=float(image.max()),out_of_range_fraction=float(((image<-1)|(image>1)).float().mean()),
            supported_edges=[int(m.sum()) for m in masks],coarse_grid=[rows,cols] if mode=='coarse_color' else None,coarse_eta=eta if mode=='coarse_color' else None)
        constraint_tolerance=settings.absolute_tolerance+settings.relative_tolerance*max(1.,float(image.abs().max()))
        diag['constraint_tolerance']=constraint_tolerance
        constraint_error=diag['anchor']['error'] if mode=='single_pixel' else diag['mean_constraint_error']
        if diag['output_normal_residual']>diag['stopping_threshold'] or constraint_error>constraint_tolerance:
            diag.update(converged=False,failure='Final FP32 output fails normal equation or reference constraint')
            _sync(device);diag['solve_seconds']=time.perf_counter()-start
            raise GradientSolveError(diag['failure'],diag)
        _sync(device);diag['solve_seconds']=time.perf_counter()-start
        return image,diag
