import unittest
from dataclasses import replace
import torch
from diffpano.config import FusionConfig
from diffpano.fusion import RGBFusionAccumulator
from diffpano.projection import ERPContribution
from diffpano.gradient_fusion import (GradientSettings,GradientSolveError,GradientFusionAccumulator,
    GuidanceAccumulator,edges,adjoint,apply_system,reconstruct,jacobi_diagonal)

CFG=FusionConfig(mode='weighted_average',weight_mode='spherediff_center')

def contribution(rgb,weight=1.,mask=None):
    mask=torch.ones_like(rgb[:,:1]) if mask is None else mask
    weight=torch.full_like(mask,weight) if isinstance(weight,(int,float)) else weight
    return ERPContribution(rgb,mask,weight)

def fuse(sources,mode='poisson_select',previous=None,**kw):
    if previous is None:previous=torch.zeros_like(sources[0][1].rgb)
    acc=GradientFusionAccumulator(previous,CFG,GradientSettings(mode=mode,**kw))
    for ident,c in sources:acc.accumulate(c,ident)
    return acc.finalize().erp_rgb,acc

class OperatorTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(8);torch.set_num_threads(2)
    def test_rgb_exact_and_rng_unchanged(self):
        sources=[(i,contribution(torch.randn(2,3,7,11),torch.rand(2,1,7,11))) for i in range(5)]
        previous=torch.randn(2,3,7,11);ref=RGBFusionAccumulator(previous,CFG)
        for _,c in sources:ref.accumulate(c)
        state=torch.random.get_rng_state();out,_=fuse(sources,'rgb',previous)
        self.assertTrue(torch.equal(out,ref.finalize().erp_rgb));self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
    def test_adjoint_spd_and_dense_reference(self):
        for h,w in [(1,1),(1,5),(4,1),(4,7)]:
            x=torch.randn(1,3,h,w,dtype=torch.float64);d=edges(x)
            y=tuple(torch.randn_like(t) for t in d)
            torch.testing.assert_close(sum((a*b).sum() for a,b in zip(d,y)),(x*adjoint(*y)).sum(),atol=1e-12,rtol=1e-12)
            support=tuple(torch.rand_like(t[:,:1])>.35 for t in d)
            z=torch.randn_like(x);ax=apply_system(x,support,.1);az=apply_system(z,support,.1)
            self.assertGreater(float((x*ax).sum()),0)
            torch.testing.assert_close((x*az).sum(),(ax*z).sum(),atol=1e-12,rtol=1e-12)
            basis=torch.eye(h*w,dtype=torch.float64).reshape(h*w,1,h,w)
            matrix=apply_system(basis,support,.1).reshape(h*w,h*w).T
            rhs=adjoint(*(m*(g-v) for m,g,v in zip(support,y,d)))
            exact=torch.linalg.solve(matrix,rhs.reshape(3,-1).T).T.reshape_as(x)
            out,diag=reconstruct(x,y,support,GradientSettings('poisson_select',relative_tolerance=1e-11,absolute_tolerance=1e-12),dtype=torch.float64)
            torch.testing.assert_close(out,x+exact,atol=1e-9,rtol=1e-9)
            torch.testing.assert_close(jacobi_diagonal(x,support,.1).flatten(),matrix.diag(),atol=1e-12,rtol=1e-12)
            self.assertTrue(diag['converged'])
    def test_exact_zero_rhs(self):
        x=torch.randn(1,3,9,17);g=edges(x);m=tuple(torch.ones_like(v[:,:1],dtype=torch.bool) for v in g)
        out,d=reconstruct(x,g,m,GradientSettings('poisson_mean'))
        self.assertTrue(torch.equal(x,out));self.assertEqual(d['iterations'],0);self.assertEqual(d['final_true_residual'],0)
    def test_single_identical_and_constant_sources(self):
        for x in [torch.randn(1,3,9,17),torch.full((1,3,9,17),.37)]:
            for mode in ['poisson_mean','poisson_select']:
                for sources in [[(0,contribution(x))],[(0,contribution(x,.3)),(1,contribution(x,.7))]]:
                    out,_=fuse(sources,mode);torch.testing.assert_close(out,x,atol=1e-6,rtol=1e-6)
    def test_invalid_nan_missing_edges_and_previous(self):
        x=torch.full((1,3,3,6),.8);mask=torch.zeros(1,1,3,6);mask[...,2:4]=1
        x[...,0]=float('nan');weight=torch.ones_like(mask);weight[...,0]=float('nan')
        previous=torch.full_like(x,.25)
        for mode in ['poisson_mean','poisson_select']:
            out,acc=fuse([(0,contribution(x,weight,mask))],mode,previous)
            expected=torch.where(mask.bool(),.8,.25).expand_as(out)
            torch.testing.assert_close(out,expected,atol=0,rtol=0)
            g,s=acc.statistics.guidance();self.assertFalse(bool(s[0][...,1].any()));self.assertFalse(bool(s[0][...,3].any()))
            self.assertEqual(float(g[0].abs().sum()),0.)
        x[...,2]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):fuse([(0,contribution(x,weight,mask))])
    def test_erp_boundaries(self):
        x=torch.arange(12.).reshape(1,1,3,4).expand(1,3,3,4)
        dx,dy=edges(x);self.assertEqual(dy.shape[-2],2)
        torch.testing.assert_close(dx[...,-1],x[...,0]-x[...,-1]);self.assertEqual(float(dy.min()),4.)
        t=torch.arange(31.)*2*torch.pi/31
        x=torch.sin(t)[None,None,None].expand(1,3,13,31).clone()
        for mode in ['poisson_mean','poisson_select']:
            out,_=fuse([(0,contribution(x))],mode);self.assertTrue(torch.equal(x,out))
            self.assertFalse(torch.equal(out[...,0],out[...,-1]))
    def test_order_ties_duplicate_scale_tiny_weights(self):
        a=torch.randn(1,3,8,15);b=torch.randn_like(a);w=torch.rand(1,1,8,15)+.2
        sources=[(9,contribution(a,w)),(2,contribution(b,w))]
        for mode in ['poisson_mean','poisson_select']:
            base,acc=fuse(sources,mode)
            for other in [sources[::-1],sources*2,[(i,contribution(c.rgb,c.weight*1e-12)) for i,c in sources]]:
                out,other_acc=fuse(other,mode);torch.testing.assert_close(out,base,atol=3e-6,rtol=3e-6)
                if mode=='poisson_select':
                    for owner in other_acc.statistics.owners:self.assertTrue(bool((owner==2).all()))
        out,_=fuse([(0,contribution(a,1e-20))],'poisson_mean');torch.testing.assert_close(out,a,atol=1e-6,rtol=1e-6)
    def test_same_source_all_channels_not_mosaic(self):
        a=torch.randn(1,3,5,12);b=torch.randn_like(a)
        w=torch.ones(1,1,5,12);w[...,:6]=2
        g=GuidanceAccumulator(a,'poisson_select');g.accumulate(contribution(a,w),0);g.accumulate(contribution(b,3-w),1)
        for axis,(out,own) in enumerate(zip(g.guidance()[0],g.owners)):
            expected=torch.where(own==0,edges(a)[axis],edges(b)[axis]);self.assertTrue(torch.equal(out,expected))
    def test_color_offset_and_no_clamp(self):
        a=torch.randn(1,3,5,12)*.3;b=torch.randn_like(a)*.3
        sources=[(0,contribution(a,torch.rand(1,1,5,12)+.1)),(1,contribution(b))]
        offset=torch.tensor([2.,-.4,.7])[None,:,None,None]
        for mode in ['poisson_mean','poisson_select']:
            base,_=fuse(sources,mode);out,acc=fuse([(i,contribution(c.rgb+offset,c.weight)) for i,c in sources],mode)
            torch.testing.assert_close(out,base+offset,atol=2e-6,rtol=2e-6)
            self.assertGreater(acc.diagnostics['out_of_range_fraction'],0)
            self.assertLess(max(abs(v) for v in acc.diagnostics['mean_color_drift']),2e-6)
    def test_failure_budget_and_validation(self):
        x=torch.randn(1,3,7,13);g=edges(torch.randn_like(x));m=tuple(torch.ones_like(v[:,:1],dtype=torch.bool) for v in g)
        with self.assertRaises(GradientSolveError) as ctx:reconstruct(x,g,m,GradientSettings('poisson_select',max_iterations=1))
        self.assertFalse(ctx.exception.diagnostics['converged']);self.assertEqual(ctx.exception.diagnostics['iterations'],1)
        for kwargs in [dict(mode='cea'),dict(lambda_color=0),dict(relative_tolerance=float('nan'))]:
            with self.assertRaises(ValueError):GradientSettings(**kwargs)
        bad=list(g);bad[0]=bad[0].clone();bad[0][...,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite guidance'):reconstruct(x,bad,m,GradientSettings('poisson_select'))

if __name__=='__main__':unittest.main()
