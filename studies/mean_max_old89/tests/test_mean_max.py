import importlib.util
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
import torch
from diffpano.gradient_fusion import GradientSettings,GuidanceAccumulator,edges,adjoint,reconstruct
from diffpano.projection import ERPContribution
from diffpano.poisson_reference import coarse_project,graph_connectivity
from studies.mean_max_old89.common import OUT,definitions,PROMPTS,ORIGINAL,NEW,prompt


class MeanMaxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(719)
        self.ref=torch.randn(1,3,4,8)
        self.inputs=[ERPContribution(torch.randn_like(self.ref),torch.ones(1,1,4,8),torch.rand(1,1,4,8)+.01) for _ in range(4)]

    def stats(self,mode='poisson_mean_max',alpha=.5,inputs=None,order=None):
        a=GuidanceAccumulator(self.ref,mode,alpha if mode=='poisson_mean_max' else None)
        inputs=self.inputs if inputs is None else inputs
        for i in (range(len(inputs)) if order is None else order):a.accumulate(inputs[i],i)
        return a

    def test_alpha_and_historical_schema(self):
        self.assertEqual(GradientSettings('poisson_mean_max').gradient_mix_alpha,.5)
        for alpha in (float('nan'),float('inf'),-.01,1.01,True,'0.5',[.5],torch.tensor(.5)):
            with self.assertRaises(ValueError):GradientSettings('poisson_mean_max',gradient_mix_alpha=alpha)
            with self.assertRaises(ValueError):GuidanceAccumulator(self.ref,'poisson_mean_max',alpha)
        for mode in ('rgb','poisson_mean','poisson_select','poisson_max'):
            s=GradientSettings(mode);self.assertNotIn('gradient_mix_alpha',s.to_record())
            self.assertTrue(s.matches_record(s.to_record()))
            with self.assertRaises(ValueError):GradientSettings(mode,gradient_mix_alpha=.5)
        old=GradientSettings.from_record(dict(mode='poisson_select',reference_mode='single_pixel'))
        self.assertEqual(old.single_pixel_source,'legacy_source_patch');self.assertIsNone(old.gradient_mix_alpha)

    def test_endpoints_and_exact_arithmetic(self):
        mean,_=self.stats('poisson_mean').guidance();maximum,_=self.stats('poisson_max').guidance()
        for alpha in (0,.25,.5,.75,1):
            a=self.stats(alpha=alpha);mix,support=a.guidance()
            self.assertEqual(a.selection_kind,'magnitude_max')
            for x,m,z in zip(mix,mean,maximum):
                expected=m if alpha==0 else z if alpha==1 else (1-alpha)*m+alpha*z
                torch.testing.assert_close(x,expected,atol=0,rtol=0)
            self.assertTrue(all(bool(m.all()) for m in support))
            with self.assertRaises(ValueError):a.guidance('poisson_select')

    def test_signed_vectors_weights_ties_invalid_and_cancellation(self):
        one=torch.ones(1,1,4,8);a=torch.zeros_like(self.ref);a[...,0,1]=torch.tensor([-3.,1.,0.])
        b=torch.zeros_like(a);b[...,0,1]=torch.tensor([1.,2.,2.])
        acc=self.stats(inputs=[ERPContribution(a,one,one*.001),ERPContribution(b,one,one*100)])
        maximum,_=acc.guidance('poisson_max')
        torch.testing.assert_close(maximum[0][...,0,0],torch.tensor([[-3.,1.,0.]]))
        self.assertEqual(int(acc.owners[0][...,0,0]),0)
        tie=self.stats(inputs=[ERPContribution(a,one,one),ERPContribution(-a,one,one)],order=[1,0])
        torch.testing.assert_close(tie.guidance('poisson_mean')[0][0],torch.zeros_like(a))
        torch.testing.assert_close(tie.guidance()[0][0],.5*edges(a)[0])
        bad=one.clone();bad[...,0,1]=0
        invalid=a*100;invalid[...,0,1]=float('nan');bad_weight=one.clone();bad_weight[...,0,1]=float('nan')
        acc.accumulate(ERPContribution(invalid,bad,bad_weight),0)
        self.assertTrue(all(bool(torch.isfinite(x).all()) for x in acc.guidance()[0]))
        for target in ('rgb','weight'):
            item=ERPContribution(torch.full_like(a,float('nan')) if target=='rgb' else a,one,torch.full_like(one,float('nan')) if target=='weight' else one)
            with self.assertRaises(ValueError):self.stats(inputs=[item])

    def test_order_duplicate_confidence_scale_small_and_identical(self):
        original=self.stats().guidance()[0]
        variants=[self.stats(order=[3,1,2,0]),self.stats(inputs=self.inputs*2),
                  self.stats(inputs=[ERPContribution(x.rgb,x.valid_mask,x.weight*1e-25) for x in self.inputs])]
        for acc in variants:
            for a,b in zip(original,acc.guidance()[0]):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
        for value in (self.ref,torch.zeros_like(self.ref)):
            item=ERPContribution(value,torch.ones(1,1,4,8),torch.ones(1,1,4,8))
            for alpha in (.25,.5,.75):
                for a,b in zip(self.stats(alpha=alpha,inputs=[item]*3).guidance()[0],edges(value)):
                    torch.testing.assert_close(a,b,atol=5e-7,rtol=5e-7)

    def test_dense_fp64_both_objectives_periodic_open_and_adjoint(self):
        g,support=self.stats(alpha=.75).guidance();g=tuple(x.double() for x in g)
        ref=self.ref.double();h,w=ref.shape[-2:];n=h*w
        eye=torch.eye(n,dtype=torch.float64).reshape(n,1,h,w)
        lap=adjoint(*edges(eye)).reshape(n,n).T
        self.assertEqual(lap[0,w-1],-1);self.assertEqual(lap[0,(h-1)*w],0)
        torch.testing.assert_close(sum((x*y).sum() for x,y in zip(edges(ref),g)),(ref*adjoint(*g)).sum())
        rhs=adjoint(*g).reshape(3,n).T
        for reference in ('screened','coarse_color'):
            s=GradientSettings('poisson_mean_max',gradient_mix_alpha=.75,reference_mode=reference,coarse_rows=2,coarse_cols=2,relative_tolerance=1e-10,absolute_tolerance=1e-11)
            if reference=='screened':expected=torch.linalg.solve(lap+.1*torch.eye(n,dtype=torch.float64),rhs+.1*ref.reshape(3,n).T).T.reshape_as(ref)
            else:
                proj=coarse_project(eye,2,2).reshape(n,n).T;c=torch.ones(n,1,dtype=torch.float64)/n
                matrix=torch.cat((torch.cat((lap+.1*proj,c),1),torch.cat((c.T,torch.zeros(1,1,dtype=torch.float64)),1)),0)
                target=torch.cat((rhs+.1*proj@ref.reshape(3,n).T,ref.mean((-2,-1))),0)
                expected=torch.linalg.solve(matrix,target)[:-1].T.reshape_as(ref)
            actual,record=reconstruct(ref,g,support,s,dtype=torch.float64)
            torch.testing.assert_close(actual,expected,atol=2e-8,rtol=2e-8)
            self.assertEqual(record['gradient_mix_alpha'],.75)
        self.assertEqual(graph_connectivity(support)['components'],[1])
        disconnected=tuple(x.clone() for x in support);disconnected[1][...,1,:]=False
        self.assertEqual(graph_connectivity(disconnected)['components'],[2])

    def test_legacy_same_source_outputs(self):
        path=OUT/'history/baseline-source/diffpano/gradient_fusion.py'
        spec=importlib.util.spec_from_file_location('meanmax_legacy_fusion',str(path));old=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=old;spec.loader.exec_module(old)
        for mode in ('poisson_mean','poisson_select','poisson_max'):
            a=old.GuidanceAccumulator(self.ref,mode);b=self.stats(mode)
            for i,x in enumerate(self.inputs):a.accumulate(x,i)
            for x,y in zip(a.guidance()[0],b.guidance()[0]):torch.testing.assert_close(x,y,atol=0,rtol=0)
            x,dx=old.reconstruct(self.ref,*a.guidance(),old.GradientSettings(mode))
            y,dy=reconstruct(self.ref,*b.guidance(),GradientSettings(mode))
            torch.testing.assert_close(x,y,atol=0,rtol=0);self.assertEqual(dx['iterations'],dy['iterations'])

    def test_capture_file_preserves_max_and_mixture(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from studies.gradient_blending.diagnostics import Capture
        acc=self.stats(alpha=.25)
        with tempfile.TemporaryDirectory() as directory:
            capture=Capture(Path(directory),[],sufficient_statistics=True);capture.view_ids=()
            result=SimpleNamespace(rgb=self.ref)
            with patch('studies.gradient_blending.diagnostics.save_rgb'):
                capture('canvas',1,result,result,acc)
            data=torch.load(Path(directory)/'intermediate/step-01/sufficient-statistics.pt',weights_only=False)
            self.assertEqual(data['selection_kind'],'magnitude_max');self.assertNotIn('selected',data)
            self.assertEqual(data['gradient_mix_alpha'],.25)
            for actual,expected in zip(data['mixed'],acc.guidance()[0]):torch.testing.assert_close(actual,expected,atol=0,rtol=0)

    def test_fp32_autocast_and_invalid_reference_safety(self):
        from diffpano.gradient_fusion import GradientFusionAccumulator
        from diffpano.config import FusionConfig
        valid=torch.ones(1,1,4,8);valid[...,0,0]=0
        bad=self.ref.clone();bad[...,0,0]=float('nan')
        weight=torch.ones_like(valid);weight[...,0,0]=float('nan')
        cfg=GradientSettings('poisson_mean_max',gradient_mix_alpha=.5)
        acc=GradientFusionAccumulator(torch.zeros_like(self.ref),FusionConfig(mode='weighted_average'),cfg)
        with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
            acc.accumulate(ERPContribution(bad,valid,weight),1)
            acc.accumulate(ERPContribution(self.ref,torch.ones_like(valid),torch.ones_like(valid)),0)
            out=acc.finalize()
        self.assertEqual(out.erp_rgb.dtype,torch.float32);self.assertTrue(bool(torch.isfinite(out.erp_rgb).all()))
        self.assertTrue(all(x.dtype==torch.float32 for x in acc.statistics.guidance()[0]))

    def test_matrix_exact_full_paths(self):
        rows=definitions();self.assertEqual(len(rows),90);self.assertEqual(len({r['case_id'] for r in rows}),90)
        self.assertEqual(len(ORIGINAL),11);self.assertEqual(len(NEW),4)
        from studies.mean_max_old89.configuration import resolved
        geometries=set()
        for name in PROMPTS:
            group=[r for r in rows if r['prompt']==name];self.assertEqual(len(group),6)
            info,_=prompt(name);self.assertEqual(info['path'],group[0]['prompt_path'])
            invariants=[]
            for r in group:
                c,cams,ids,slots,s,g,old=resolved(r);geometries.add(g['angular_sha256'])
                self.assertEqual(s.gradient_mix_alpha,r['gradient_mix_alpha']);self.assertEqual(c.global_pipeline.refinement.last_fraction,0)
                self.assertEqual(ids,slots);self.assertEqual(len(cams),89)
                invariants.append((g,old['prepared_schedule'],old['initialization']['initial_local_sha256']))
            self.assertTrue(all(x==invariants[0] for x in invariants))
        self.assertEqual(len(geometries),1)


class IntegrationTests(unittest.TestCase):
    def test_twenty_intervals_all_six_and_capture_exact(self):
        from studies.tt_cea.tests.test_pipeline import PipelineTests,IndependentMock,config
        from studies.tt_cea.schedule import prepare_interval_table
        from studies.gradient_blending.operator import StudyPipeline
        from studies.all_prompts.runtime import ordinary_run
        from diffpano.refinement import RefinementConfig
        class Backend(IndependentMock):
            def __init__(self):
                super().__init__();self.decodes=0;self.pipeline.scheduler.set_timesteps(20);self.timesteps=self.pipeline.scheduler.timesteps
            def decode_clean(self,x):self.decodes+=1;return super().decode_clean(x)
        class Observer:
            capture_statistics=True;view_ids=()
            def __init__(self):self.snapshots=[]
            def __call__(self,kind,stage,*args):
                if kind=='canvas':
                    stats=args[-1];self.assertions(stats)
                    self.snapshots.append((stage,stats.mode))
            def assertions(self,stats):
                assert stats.mode=='poisson_mean_max' and stats.selection_kind=='magnitude_max'
                m,_=stats.guidance('poisson_mean');z,_=stats.guidance('poisson_max');g,_=stats.guidance()
                for a,x,y in zip(g,m,z):assert torch.equal(a,(1-stats.gradient_mix_alpha)*x+stats.gradient_mix_alpha*y)
        helper=PipelineTests();outputs={}
        for ref in ('screened','coarse_color'):
            for alpha in (.25,.5,.75):
                for capture in (False,True):
                    backend=Backend();cfg=config();cfg.global_pipeline=SimpleNamespace(refinement=RefinementConfig(0.))
                    observer=Observer() if capture else None
                    settings=GradientSettings('poisson_mean_max',gradient_mix_alpha=alpha,reference_mode=ref,coarse_rows=2,coarse_cols=2,relative_tolerance=1e-4 if ref=='coarse_color' else 1e-5)
                    pipeline=StudyPipeline(backend,helper.cams(),cfg,settings,list(range(89)),observer=observer,size=(8,16))
                    table=tuple(replace(x,eligible=False) for x in prepare_interval_table(backend))
                    _,out,details=ordinary_run(pipeline,helper.initial(),[{'offset':torch.tensor(.1)}]*89,table)
                    self.assertEqual((backend.guided_prediction_count,len(backend.bridge_inputs),backend.decodes),(1780,3560,1869))
                    self.assertEqual(len(pipeline.canvas.records),21);self.assertTrue(pipeline.canvas.records[-1]['terminal'])
                    self.assertEqual(details['replay_count'],0);self.assertEqual(details['backward_calls'],0)
                    for r in pipeline.canvas.records:
                        self.assertEqual(r['gradient_mix_alpha'],alpha);self.assertEqual(r['reference_mode'],ref);self.assertTrue(r['converged'])
                    if capture:torch.testing.assert_close(out,outputs[(ref,alpha)],atol=0,rtol=0)
                    else:outputs[(ref,alpha)]=out
