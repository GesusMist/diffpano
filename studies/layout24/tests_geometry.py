"""Focused exact-scope, camera/routing and legacy-select semantic tests."""
import unittest
from dataclasses import asdict
from collections import Counter
from studies.layout24.common import *
from studies.layout24.configuration import resolved


class ScopeGeometry(unittest.TestCase):
    def test_exact_matrix(self):
        rows=definitions();self.assertEqual(len(rows),216)
        self.assertEqual(Counter(r['variant'] for r in rows),{v:24 for v in VARIANTS})
        self.assertNotIn('native_control',PROMPTS)
        self.assertEqual(len(PROMPTS),24)
        for row in rows:
            info,raw=prompt(row['prompt']);self.assertEqual(len(info['effective_lines']),5)
            self.assertEqual(raw,prompt_path(row['prompt']).read_bytes())
            self.assertEqual(row['steps'],20);self.assertEqual(row['seed'],0);self.assertEqual(row['last_fraction'],0.)
            self.assertNotIn('gradient_mix_alpha',row)
            if row['gradient_mode']=='poisson_select':self.assertEqual(row['camera_count'],89)
    def test_exact_banks_and_routing(self):
        from studies.camera_patching.cameras import angular_hash,fibonacci_camera_cover
        from studies.camera_count_v2.cameras import RINGS,RING_COUNTS
        import math
        rows=definitions()[:9];banks={}
        for row in rows[1:7]:
            c,cameras,owners,slots,settings,g,_=resolved(row)
            self.assertEqual(c.global_pipeline.refinement.last_fraction,0.)
            self.assertEqual(len(cameras),row['camera_count']);self.assertEqual(owners,list(range(len(cameras))))
            self.assertEqual(len(set(slots)),len(cameras));self.assertEqual(settings.mode,'rgb')
            self.assertEqual(g['route']['indices'],[int(x) for x in g['route']['indices']])
            self.assertEqual(len(g['route']['indices']),len(cameras))
            self.assertTrue(all(cam.fov_x==cam.fov_y==80 and cam.roll==0 and cam.width==cam.height==1024 for cam in cameras))
            if row['layout'].startswith('uniform'):self.assertEqual(cameras,fibonacci_camera_cover(c.view,len(cameras),phase=0.))
            if row['layout'] in ('old70','old50'):
                actual=Counter(round(math.degrees(cam.pitch),2) for cam in cameras)
                self.assertEqual(actual,dict(zip(RINGS,RING_COUNTS[len(cameras)])))
            banks[row['layout']]=cameras
        for prefix in ('old','uniform'):
            high={(c.yaw,c.pitch) for c in banks[prefix+'89']}
            for n in (70,50):self.assertFalse({(c.yaw,c.pitch) for c in banks[prefix+str(n)]}<=high)
    def test_rgb_select_preserve_geometry(self):
        rows=definitions()[:9]
        for rgb,select in ((rows[1],rows[7]),(rows[4],rows[8])):
            a=resolved(rgb);b=resolved(select)
            self.assertEqual(safe(asdict(a[0])),safe(asdict(b[0])));self.assertEqual(a[1:4],b[1:4]);self.assertEqual(a[5],b[5])
            settings=b[4];self.assertEqual(settings.mode,'poisson_select');self.assertEqual(settings.reference_mode,'screened')
            self.assertEqual(settings.lambda_color,.1);self.assertEqual(settings.relative_tolerance,1e-5)
            self.assertEqual(settings.absolute_tolerance,1e-7);self.assertEqual(settings.max_iterations,200)
            self.assertNotIn('gradient_mix_alpha',settings.to_record())
    def test_current_native_noise_matches_each_saved_bank(self):
        import torch
        from studies.gwtf_noise.common import GeometryBackend
        from diffpano.gwtf_noise_initialization import initialize_gwtf_shared_noise,GWTFlowNoiseConfig
        from diffpano.erp_noise_initialization import primary_noise_size,native_cameras
        from studies.layout24.prepare import expected_initialization
        backend=GeometryBackend(16,8,1.)
        for row in definitions()[1:7]:
            _,cameras,owners,slots,_,_,old=resolved(row)
            h,w=primary_noise_size(native_cameras(backend,cameras))
            states,record=initialize_gwtf_shared_noise(backend,cameras,GWTFlowNoiseConfig(h,w,0),1,camera_slots=slots)
            expected=expected_initialization(row,old)
            for key in ('initial_local_sha256','source_sha256','source_shape','scaling','scaling_applications'):
                self.assertEqual(safe(record[key]),expected[key],(row['layout'],key))
            self.assertEqual(record['camera_slots'],slots)
            self.assertEqual(record['denoiser_calls'],0);self.assertEqual(record['vae_calls'],0)
            del states
    def test_counts(self):
        for n in (89,70,50):self.assertEqual(expected_counts(dict(camera_count=n)),dict(denoiser=20*n,encode=40*n,decode=21*n,initialize=0))
    def test_confidence_select_is_not_magnitude_max(self):
        import torch
        from diffpano.gradient_fusion import GuidanceAccumulator
        from diffpano.projection import ERPContribution
        rgb=torch.tensor([[[[0.,1.,2.],[0.,1.,2.]]]]).repeat(1,3,1,1)
        valid=torch.ones(1,1,2,3)
        acc=GuidanceAccumulator(torch.zeros_like(rgb),'poisson_select')
        acc.accumulate(ERPContribution(rgb,valid,valid),7)
        acc.accumulate(ERPContribution(rgb*10,valid,valid*.1),0)
        guidance,support=acc.guidance('poisson_select')
        self.assertTrue(torch.equal(guidance[0][...,:-1],torch.ones_like(guidance[0][...,:-1])))
        acc.accumulate(ERPContribution(rgb*2,valid,valid),3)
        guidance,_=acc.guidance('poisson_select')
        self.assertTrue(torch.equal(guidance[0][...,:-1],torch.full_like(guidance[0][...,:-1],2.)))


if __name__=='__main__':unittest.main()
