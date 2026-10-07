"""Full-length, tiny-raster integration checks for the two added backends."""
import unittest
from types import SimpleNamespace
import torch
from studies.tt_cea.tests.test_pipeline import IndependentMock,PipelineTests,config
from studies.tt_cea.pipeline import ExperimentalPipeline
from studies.all_prompts.runtime import ordinary_run
from studies.gradient_blending.operator import StudyPipeline
from studies.gradient_blending.run import benchmark_schedule
from studies.gradient_blending.common import expected_counts
from diffpano.gradient_fusion import GradientSettings

class ScheduleMock(IndependentMock):
    def __init__(self,pixel):
        super().__init__(pixel);self.decode_count=0;self.prepare_arguments=None
    def prepare(self,**kwargs):
        self.prepare_arguments=kwargs
        self.pipeline.scheduler.set_timesteps(kwargs['num_steps'])
        self.timesteps=self.pipeline.scheduler.timesteps
    def decode_clean(self,x):
        if self.pixel:raise AssertionError('PixelDiT must not decode native RGB')
        self.decode_count+=1
        return super().decode_clean(x)

class ExtensionChecks(unittest.TestCase):
    def check_backend(self,name,steps):
        torch.set_num_threads(2)
        pixel=name=='pixeldit';helper=PipelineTests();c=config(pixel)
        c.model.pipeline=name;c.generation=SimpleNamespace(num_inference_steps=steps)
        c.view=SimpleNamespace(height=4,width=4)
        cams=helper.cams();initial=helper.initial();conditions=[{'offset':torch.tensor(.1)}]*89
        baseline=ScheduleMock(pixel);table=benchmark_schedule(baseline,c)
        p=ExperimentalPipeline(baseline,cams,c,size=(8,16))
        _,reference,original=ordinary_run(p,initial,conditions,table)
        expected=expected_counts(name,steps)
        self.assertEqual(expected['denoiser'],4450 if pixel else 3560)
        self.assertEqual(expected['encode'],0 if pixel else 7120)
        self.assertEqual(expected['decode'],0 if pixel else 3649)
        for mode in ('rgb','poisson_mean','poisson_select'):
            b=ScheduleMock(pixel);intervals=benchmark_schedule(b,c)
            self.assertEqual(b.prepare_arguments,dict(num_steps=steps,view_height=4,view_width=4))
            self.assertEqual(intervals,table);self.assertFalse(any(v.eligible for v in intervals))
            p=StudyPipeline(b,cams,c,GradientSettings(mode),list(range(89)),size=(8,16))
            _,image,record=ordinary_run(p,initial,conditions,intervals)
            if mode=='rgb':self.assertTrue(torch.equal(image,reference))
            self.assertEqual(record['initial_state_sha256'],original['initial_state_sha256'])
            self.assertEqual(b.guided_prediction_count,expected['denoiser'])
            self.assertEqual(len(b.bridge_inputs),expected['encode'])
            self.assertEqual(b.decode_count,expected['decode'])
            self.assertEqual(len(p.canvas.records),steps+1)
            self.assertEqual(p.canvas.records[-1]['stage'],'terminal')
            self.assertTrue(all(r['mode']==mode for r in p.canvas.records))
            self.assertTrue(all(r['current_state_error_max']<2e-6 for r in p.interval_records))
            self.assertTrue(torch.equal(b.timesteps,baseline.timesteps))
            if mode!='rgb':self.assertTrue(all(r['converged'] for r in p.canvas.records))
    def test_pixeldit_50_intervals_without_vae(self):self.check_backend('pixeldit',50)
    def test_sd35_40_intervals_with_local_bridge(self):self.check_backend('sd35',40)

if __name__=='__main__':unittest.main()
