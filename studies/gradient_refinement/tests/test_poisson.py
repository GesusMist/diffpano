import unittest
from dataclasses import replace
import torch
from diffpano.gradient_fusion import (GradientSettings,GuidanceAccumulator,edges,adjoint,reconstruct,
                                     _reconstruct_screened,GradientSolveError)
from diffpano.poisson_reference import block_mean,block_adjoint,coarse_project,SpectralInverse,graph_connectivity
from diffpano.projection import ERPContribution

class PoissonTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20);self.x=torch.randn(1,3,4,8,dtype=torch.float64)
        self.support=(torch.ones(1,1,4,8,dtype=torch.bool),torch.ones(1,1,3,8,dtype=torch.bool))
    def settings(self,r):return GradientSettings('poisson_select',reference_mode=r,coarse_rows=2,coarse_cols=2,relative_tolerance=1e-10,absolute_tolerance=1e-11)
    def anchor(self,color=None):return dict(erp_y=2,erp_x=3,color=self.x[...,2,3] if color is None else color,camera_id=4)
    def test_identical_sources_and_constraints(self):
        for mode in ('poisson_select','poisson_max'):
            a=GuidanceAccumulator(self.x,mode)
            contribution=ERPContribution(self.x.float(),self.support[0].float(),self.support[0].float())
            a.accumulate(contribution,1);a.accumulate(contribution,3)
            for reference in ('screened','global_mean','single_pixel','coarse_color'):
                # Compute exact FP64 differences for numerical reference tests.
                out,diag=reconstruct(self.x,edges(self.x),self.support,replace(self.settings(reference),mode=mode),dtype=torch.float64,anchor=self.anchor())
                torch.testing.assert_close(out,self.x,atol=1e-8,rtol=1e-8)
                self.assertTrue(diag['converged'])
    def test_dense_fp64_constrained_masked_and_periodic_open(self):
        h,w=self.x.shape[-2:];n=h*w
        masks=tuple(m.clone() for m in self.support);masks[0][...,0,1]=False;masks[1][...,1,3]=False
        identity=torch.eye(n,dtype=torch.float64).reshape(n,1,h,w)
        dx,dy=edges(identity)
        lap=adjoint(dx*masks[0],dy*masks[1]).reshape(n,n).T
        g=(torch.randn_like(self.x),torch.randn(1,3,h-1,w,dtype=torch.float64));rhs=adjoint(g[0]*masks[0],g[1]*masks[1]).reshape(3,n).T
        for r in ('global_mean','single_pixel','coarse_color'):
            cfg=self.settings(r);operator=lap.clone();target=rhs.clone();mean=self.x.mean((-2,-1)).flatten()
            if r=='coarse_color':
                p=coarse_project(identity,2,2).reshape(n,n).T;operator+=.1*p;target+=.1*p@self.x.reshape(3,n).T
            constraints=torch.ones(n,1,dtype=torch.float64)/n
            if r=='single_pixel':constraints.zero_();constraints[2*w+3]=1;mean=self.x[...,2,3].flatten()
            kkt=torch.cat((torch.cat((operator,constraints),1),torch.cat((constraints.T,torch.zeros(1,1,dtype=torch.float64)),1)),0)
            expected=torch.linalg.solve(kkt,torch.cat((target,mean[None]),0))[:-1].T.reshape_as(self.x)
            actual,diag=reconstruct(self.x,g,masks,cfg,dtype=torch.float64,anchor=self.anchor())
            torch.testing.assert_close(actual,expected,atol=2e-8,rtol=2e-8)
        # Longitude seam participates; open latitude has no pole-to-pole edge.
        self.assertEqual(lap[0,w-1],-1);self.assertEqual(lap[0,(h-1)*w],0)
    def test_global_pixel_equal_gradients_and_offset(self):
        g=(torch.randn_like(self.x),torch.randn(1,3,3,8,dtype=torch.float64));color=torch.tensor([[.8,-.4,.2]],dtype=torch.float64)
        a,_=reconstruct(self.x,g,self.support,self.settings('global_mean'),dtype=torch.float64)
        b,d=reconstruct(self.x,g,self.support,self.settings('single_pixel'),dtype=torch.float64,anchor=self.anchor(color))
        for x,y in zip(edges(a),edges(b)):torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(a.mean((-2,-1)),self.x.mean((-2,-1)))
        torch.testing.assert_close(b[...,2,3],color);self.assertLess(d['anchor']['error'],1e-12)
    def test_blocks_true_adjoint_and_nullspace(self):
        z=torch.randn(1,3,2,2,dtype=torch.float64)
        torch.testing.assert_close((block_mean(self.x,2,2)*z).sum(),(self.x*block_adjoint(z,4,8)).sum())
        torch.testing.assert_close((8*block_mean(self.x,2,2)*z).sum(),(self.x*block_adjoint(z,4,8,weighted=True)).sum())
        null=self.x-coarse_project(self.x,2,2)
        self.assertLess(float(coarse_project(null,2,2).abs().max()),1e-14)
        with self.assertRaises(ValueError):block_mean(self.x,3,2)
    def test_disconnected_fails(self):
        masks=tuple(m.clone() for m in self.support);masks[1][...,1,:]=False
        self.assertEqual(graph_connectivity(masks)['components'],[2])
        with self.assertRaisesRegex(GradientSolveError,'Disconnected'):reconstruct(self.x,edges(self.x),masks,self.settings('global_mean'),dtype=torch.float64)
    def test_screened_dispatch_exact(self):
        g=edges(self.x*.6);cfg=GradientSettings('poisson_select')
        a,da=_reconstruct_screened(self.x,g,self.support,cfg);b,db=reconstruct(self.x,g,self.support,cfg)
        self.assertTrue(torch.equal(a,b));self.assertEqual(da['iterations'],db['iterations'])
    def test_max_signed_vector_validity_ties_and_changing_ownership(self):
        shape=(1,3,1,3);ref=torch.zeros(shape);mask=torch.ones(1,1,1,3)
        def contribution(v,weight=1.,valid=mask):return ERPContribution(v,valid,mask*weight)
        a=torch.tensor([[[[0.,-3.,0.]],[[0.,1.,0.]],[[0.,0.,0.]]]])
        b=torch.tensor([[[[0.,1.,0.]],[[0.,2.,0.]],[[0.,2.,0.]]]])
        acc=GuidanceAccumulator(ref,'poisson_max');acc.accumulate(contribution(b,100),1);acc.accumulate(contribution(a,.01),9)
        torch.testing.assert_close(acc.guidance()[0][0][...,0],torch.tensor([[[-3.],[1.],[0.]]]))
        self.assertEqual(int(acc.owners[0][...,0]),9)
        acc.accumulate(contribution(b*2,.001),2);self.assertEqual(int(acc.owners[0][...,0]),2)
        bad=mask.clone();bad[...,1]=0;acc.accumulate(contribution(a*100,1,bad),0);self.assertEqual(int(acc.owners[0][...,0]),2)
        zero=GuidanceAccumulator(ref,'poisson_max');zero.accumulate(contribution(ref),8);zero.accumulate(contribution(ref),3)
        self.assertTrue(bool((zero.owners[0]==3).all()))
        tie=GuidanceAccumulator(ref,'poisson_max');tie.accumulate(contribution(a),9);tie.accumulate(contribution(-a),1)
        torch.testing.assert_close(tie.guidance()[0][0],edges(-a)[0])

    def test_study_source_anchor_uses_current_warped_prediction(self):
        from studies.tt_cea.tests.test_pipeline import PipelineTests,IndependentMock,config
        from studies.gradient_blending.operator import StudyPipeline
        from studies.tt_cea.schedule import prepare_interval_table
        from studies.all_prompts.runtime import ordinary_run
        h=PipelineTests();b=IndependentMock();settings=GradientSettings('poisson_max',reference_mode='single_pixel')
        p=StudyPipeline(b,h.cams(),config(),settings,list(range(89)),size=(8,16))
        _,out,_=ordinary_run(p,h.initial(),[{'offset':torch.tensor(.1)}]*89,prepare_interval_table(b))
        anchors=[r['anchor'] for r in p.canvas.records]
        self.assertEqual(len({(a['camera_id'],a['erp_y'],a['erp_x']) for a in anchors}),1)
        self.assertTrue(all(a['error']<=r['constraint_tolerance'] for a,r in zip(anchors,p.canvas.records)))
        self.assertNotEqual(anchors[0]['c0'],anchors[-1]['c0'])
        a=anchors[-1];torch.testing.assert_close(out[...,a['erp_y'],a['erp_x']],torch.tensor([a['c0']]),atol=1e-6,rtol=1e-6)
