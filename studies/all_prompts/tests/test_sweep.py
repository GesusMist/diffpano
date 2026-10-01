import ast
import contextlib
import io
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from PIL import Image
from studies.all_prompts.common import *
from studies.all_prompts.runtime import ordinary_run
from studies.all_prompts.audit import configuration
from studies.tt_cea.pipeline import ExperimentalPipeline
from studies.tt_cea.schedule import prepare_interval_table
from studies.tt_cea.canvas import CanvasOperator,CanvasSpec
from studies.tt_cea.tests.test_pipeline import IndependentMock,config
from studies.tt_cea.cameras import load_cover

class SweepTests(unittest.TestCase):
    def test_exact_matrix_unique_outputs(self):
        r=rows();self.assertEqual(len(inventory()),21);self.assertEqual(len(r),210)
        self.assertEqual(len({x['output'] for x in r}),210)
        self.assertEqual([x['index'] for x in r],list(range(210)))
        self.assertEqual(sum(x['projection']=='erp' for x in r),84)
        self.assertEqual(sum(x['projection']=='cea' for x in r),84)
        self.assertEqual(sum(x['method']=='spherediff' for x in r),42)
        self.assertTrue(all(x['backend']!='sd2' and Path(x['output']).name=='final.png' for x in r))
        self.assertTrue(all(x['steps']==28 for x in r if (x['method'],x['backend'])==('spherediff','flux')))

    def test_one_line_adaptation_and_five_line_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'one.txt';p.write_bytes(b'  exact words!  \n')
            r,b=prompt_record(p);self.assertTrue(r['adapted'])
            self.assertEqual(b,b'  exact words!  \n'*5)
            p.write_bytes(b'north\r\nup\r\nhorizon\r\ndown\r\nsouth\r\n')
            r,b=prompt_record(p);self.assertFalse(r['adapted']);self.assertEqual(b,p.read_bytes())
            p.write_text('a\nb\n')
            with self.assertRaises(ValueError):prompt_record(p)

    def test_png_atomic_skip_and_corrupt_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'final.png';p.write_bytes(b'corrupt')
            self.assertFalse(image_valid(p))
            publish(Image.new('RGB',(4096,2048),'red'),p);before=sha(p)
            self.assertTrue(image_valid(p));publish(Image.new('RGB',(4096,2048),'blue'),p)
            self.assertEqual(sha(p),before)
            self.assertEqual([x.name for x in Path(d).iterdir()],['final.png'])
            q=Path(d)/'wrong.png';Image.new('RGB',(16,8)).save(q);self.assertFalse(image_valid(q))

    def test_all_configs_fixed_old89_arithmetic(self):
        for n in BACKENDS:
            c,_,cams,ids,_=configuration(n)
            self.assertEqual(c.fusion.mode,'weighted_average')
            self.assertEqual(c.fusion.weight_mode,'spherediff_center')
            self.assertEqual(c.fusion.spherediff_temperature,.1)
            self.assertEqual(ids,list(range(89)))
            self.assertTrue(all(cam.fov_x==cam.fov_y==80 for cam in cams))

    def test_no_replay_execution_and_numerical_equivalence(self):
        cams,_=load_cover('old89',4,4)
        for pixel in (False,True):
            for projection in ('erp','cea'):
                state=[torch.randn(1,3,4,4,generator=torch.Generator().manual_seed(i)) for i in range(89)]
                cond=[{'offset':torch.tensor(.12)}]*89
                b=IndependentMock(pixel);p=ExperimentalPipeline(b,cams,config(pixel),projection=projection,size=(8,16))
                ref_native,ref,old=p.run(state,cond,prepare_interval_table(b),time_travel=False)
                fresh=IndependentMock(pixel);p2=ExperimentalPipeline(fresh,cams,config(pixel),projection=projection,size=(8,16))
                with patch('studies.tt_cea.pipeline.OriginalNoiseBank',side_effect=AssertionError('noise bank forbidden')),patch('studies.tt_cea.pipeline.backward_original_noise',side_effect=AssertionError('backward forbidden')):
                    native,out,record=ordinary_run(p2,state,cond,prepare_interval_table(fresh))
                torch.testing.assert_close(out,ref,atol=0,rtol=0)
                self.assertEqual(record['guided_predictions'],267)
                self.assertEqual(record['replay_count'],0);self.assertEqual(record['backward_calls'],0)
                self.assertFalse(record['original_noise_bank_retained'])
                self.assertEqual(record['active_consensus_projection'],projection)
                self.assertEqual(len(fresh.bridge_inputs),0 if pixel else 2*267)

    def test_lazy_counter_matches_real_native_backend_startup(self):
        class LazyCounter(IndependentMock):
            def __init__(self):
                super().__init__(True)
                del self.guided_prediction_count
            def predict_clean_and_endpoint(self,*args,**kwargs):
                if not hasattr(self,'guided_prediction_count'):self.guided_prediction_count=0
                return super().predict_clean_and_endpoint(*args,**kwargs)
        b=LazyCounter();cams,_=load_cover('old89',4,4)
        pipeline=ExperimentalPipeline(b,cams,config(True),size=(8,16))
        states=[torch.zeros(1,3,4,4) for _ in cams]
        _,_,record=ordinary_run(pipeline,states,[{'offset':torch.tensor(.1)}]*89,prepare_interval_table(b))
        self.assertEqual(record['guided_predictions'],267)
        self.assertEqual(record['replay_count'],0)

    def test_old89_cea_constant_wrap_poles(self):
        c,_,_,_,_=configuration('sana');cams,_=load_cover('old89',16,16)
        p=CanvasOperator(CanvasSpec('cea',32,64),c.warp,c.fusion,'cpu');a=p.make_accumulator(1)
        for cam in cams:p.accumulate(a,torch.full((1,3,16,16),.3),cam)
        result=p.finalize(a)
        self.assertTrue(bool((result.weight_sum>0).all()))
        self.assertTrue(bool((result.pole_weight_sum>0).all()))
        torch.testing.assert_close(result.rgb,torch.full_like(result.rgb,.3),atol=3e-7,rtol=3e-7)
        erp=p.export_erp(result,32,64)
        self.assertTrue(bool(torch.isfinite(erp).all()))
        torch.testing.assert_close(erp,torch.full_like(erp,.3),atol=3e-7,rtol=3e-7)

    def test_center_weighted_arithmetic_actually_uses_weights(self):
        from diffpano.config import FusionConfig
        from diffpano.fusion import RGBFusionAccumulator,create_view_weight_map
        from diffpano.projection import ERPContribution
        config=FusionConfig(mode='weighted_average',weight_mode='spherediff_center',spherediff_temperature=.1)
        a=RGBFusionAccumulator(torch.zeros(1,3,1,1),config);mask=torch.ones(1,1,1,1)
        a.accumulate(ERPContribution(torch.ones(1,3,1,1),mask,mask))
        a.accumulate(ERPContribution(torch.zeros(1,3,1,1),mask,mask*3))
        torch.testing.assert_close(a.finalize().erp_rgb,torch.full((1,3,1,1),.25))
        w=create_view_weight_map(3,3,'spherediff_center',temperature=.1)
        self.assertEqual(float(w[0,0,1,1]),1.)
        self.assertLess(float(w[0,0,0,0]),.001)

if __name__=='__main__':unittest.main()
