import unittest
import torch
from diffpano.gradient_fusion import GradientSettings
from studies.gradient_blending.operator import StudyPipeline,StudyCanvas
from studies.tt_cea.tests.test_pipeline import IndependentMock,config,PipelineTests
from studies.tt_cea.pipeline import ExperimentalPipeline
from studies.tt_cea.schedule import prepare_interval_table
from studies.all_prompts.runtime import ordinary_run

class IntegrationTests(unittest.TestCase):
    def test_exact_rgb_pipeline_and_gradient_bridge_counts(self):
        helper=PipelineTests();cams=helper.cams();initial=helper.initial();conditions=[{'offset':torch.tensor(.1)}]*89
        for pixel in (False,True):
            baseline=IndependentMock(pixel);p=ExperimentalPipeline(baseline,cams,config(pixel),size=(8,16))
            _,expected,base_record=ordinary_run(p,initial,conditions,prepare_interval_table(baseline))
            for mode in ('rgb','poisson_mean','poisson_select'):
                b=IndependentMock(pixel);p=StudyPipeline(b,cams,config(pixel),GradientSettings(mode),list(range(89)),size=(8,16))
                _,image,record=ordinary_run(p,initial,conditions,prepare_interval_table(b))
                if mode=='rgb':self.assertTrue(torch.equal(image,expected))
                self.assertEqual(record['initial_state_sha256'],base_record['initial_state_sha256'])
                self.assertEqual(b.guided_prediction_count,baseline.guided_prediction_count)
                self.assertEqual(len(b.bridge_inputs),len(baseline.bridge_inputs))
                self.assertEqual(len(b.terminal_inputs),len(baseline.terminal_inputs))
                self.assertEqual([r['stage'] for r in p.canvas.records],[1,2,3,'terminal'])
                self.assertTrue(all(r['current_state_error_max']<2e-6 for r in p.interval_records))
                self.assertTrue(torch.equal(b.timesteps,baseline.timesteps))
    def test_instrumented_rgb_is_exact(self):
        class Observer:
            capture_statistics=True
            view_ids=(7,59)
            def __init__(self):self.canvases=[]
            def __call__(self,kind,stage,*args):
                if kind=='canvas' and stage==1:
                    ref,result,statistics=args
                    self.canvases.append((ref.rgb.clone(),result.rgb.clone(),statistics.mode))
        helper=PipelineTests();cams=helper.cams();initial=helper.initial();conditions=[{'offset':torch.tensor(.1)}]*89
        outputs=[];observer=Observer()
        for obs in (None,observer):
            b=IndependentMock();p=StudyPipeline(b,cams,config(),GradientSettings(),list(range(89)),observer=obs,size=(8,16))
            _,image,_=ordinary_run(p,initial,conditions,prepare_interval_table(b));outputs.append(image)
        self.assertTrue(torch.equal(*outputs));self.assertEqual(len(observer.canvases),1)
        self.assertTrue(torch.equal(observer.canvases[0][0],observer.canvases[0][1]))
        self.assertEqual(observer.canvases[0][2],'both')

    def test_cannot_change_domain_or_historical_guard(self):
        helper=PipelineTests();b=IndependentMock();c=config()
        with self.assertRaisesRegex(ValueError,'ERP only'):
            StudyPipeline(b,helper.cams(),c,GradientSettings(),list(range(89)),projection='cea',size=(8,16))
        from dataclasses import replace
        c.fusion=replace(c.fusion,weight_mode='uniform')
        with self.assertRaisesRegex(ValueError,'B0C0D1'):
            StudyPipeline(b,helper.cams(),c,GradientSettings(),list(range(89)),size=(8,16))

if __name__=='__main__':unittest.main()
