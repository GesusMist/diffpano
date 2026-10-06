import unittest
from types import SimpleNamespace
from studies.flux_step_controls.common import *

class Controls(unittest.TestCase):
    def test_matrix(self):
        m=rows();self.assertEqual(len(m),42)
        self.assertEqual(len({r['output'] for r in m}),42)
        self.assertEqual({(r['method'],r['steps'],r['projection']) for r in m},
                         {('spherediff',20,'spherical'),('diffpano',28,'erp')})
        for group in (m[:21],m[21:]):self.assertEqual([r['prompt'] for r in group],[p['name'] for p in inventory()])
    def test_overrides(self):
        from studies.flux_step_controls.runtime import original_spec,diff_configuration
        c,_,_,old,d=diff_configuration('prompts/ruins.txt')
        self.assertEqual(c.generation.num_inference_steps,28)
        self.assertEqual([v['path'] for v in d],['generation.num_inference_steps'])
        s,d=original_spec();self.assertEqual(s['call']['num_inference_steps'],20)
        self.assertEqual([v['path'] for v in d],['call.num_inference_steps'])
    def test_schedule_fresh(self):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffpano.pipelines.flux import FluxViewDenoiser
        from scripts.dense_erp_experiment import canonical_schedule,schedule_metadata
        old=read(REPO/'outputs/gwtf-erp4k/20260923/flux/metadata.json')['prepared_schedule']
        p=SimpleNamespace(scheduler=FlowMatchEulerDiscreteScheduler.from_config(old['config']),
            vae_scale_factor=8,_execution_device='cpu')
        b=FluxViewDenoiser(p,guidance_scale=3.5,true_cfg_scale=1.)
        seen={}
        for n in (20,28,20):
            b.prepare(num_steps=n,view_height=1024,view_width=1024)
            r=safe(canonical_schedule(schedule_metadata(b)));schedule_valid(r,n)
            self.assertEqual(r['config'],old['config']);self.assertEqual(r['class_name'],old['class_name'])
            if n==20:self.assertEqual(r,old)
            seen[n]=r
        self.assertNotEqual(seen[20]['timesteps'],seen[28]['timesteps'][:20])
        self.assertEqual(b.scheduler_image_seq_len,4096)
    def test_invalid_schedule(self):
        for r in [dict(timesteps=[1,1],sigmas=[1,.5,0]),dict(timesteps=[2,1],sigmas=[1,0,.1])]:
            with self.assertRaises(AssertionError):schedule_valid(r,2)
    def test_partial_pair(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            row=dict(rows()[0],output=str(Path(d)/'final.png'),config=str(Path(d)/'config.json'))
            Path(row['output']).write_bytes(b'not a png')
            self.assertFalse(complete(row))
            Path(row['config']).write_text('{}');self.assertFalse(complete(row))

if __name__=='__main__':unittest.main()
