import math
import unittest
from dataclasses import replace
import torch
from diffpano.config import ViewConfig
from studies.camera_patching.cameras import *

class CameraTests(unittest.TestCase):
    def setUp(self):self.view=ViewConfig(height=64,width=96,fov_x=80,fov_y=80)
    def test_fibonacci_exact_generic_construction(self):
        hashes=[]
        for n in (1,2,17,64,89,128):
            cams=fibonacci_camera_cover(self.view,n);self.assertEqual(len(cams),n)
            self.assertEqual(cams,fibonacci_camera_cover(self.view,n));hashes.append(angular_hash(cams))
            for i,c in enumerate(cams):
                self.assertAlmostEqual(float(c.forward()[1]),1-2*(i+.5)/n,places=6)
                self.assertAlmostEqual(float(c.forward().norm()),1,places=6)
                self.assertAlmostEqual(c.yaw,((i*GOLDEN_ANGLE+math.pi)%(2*math.pi)-math.pi))
                self.assertTrue(-math.pi/2<c.pitch<math.pi/2);self.assertEqual(c.roll,0)
                self.assertEqual((c.height,c.width,c.fov_x,c.fov_y),(64,96,80,80))
            self.assertEqual(len({(c.yaw,c.pitch) for c in cams}),n)
            self.assertEqual(angular_hash(cams),angular_hash(fibonacci_camera_cover(replace(self.view,height=512,width=512),n)))
        self.assertEqual(len(set(hashes)),len(hashes))
        self.assertNotEqual(angular_hash(fibonacci_camera_cover(self.view,89)),angular_hash(fibonacci_camera_cover(self.view,89,phase=.1)))
    def test_bad_counts(self):
        for n in (0,-1,1.5,True,'89'):
            with self.assertRaises(ValueError):fibonacci_camera_cover(self.view,n)
            with self.assertRaises(ValueError):random_uniform_camera_cover(self.view,n,seed=0)
    def test_random_reproducibility_and_rng_isolation(self):
        for n in (1,17,89,128):
            torch.manual_seed(123);before=torch.random.get_rng_state().clone()
            cams=random_uniform_camera_cover(self.view,n,seed=0)
            self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
            self.assertEqual(len(cams),n);self.assertEqual(cams,random_uniform_camera_cover(self.view,n,seed=0))
            self.assertNotEqual(angular_hash(cams),angular_hash(random_uniform_camera_cover(self.view,n,seed=1)))
            self.assertEqual(angular_hash(cams),angular_hash(random_uniform_camera_cover(replace(self.view,height=1024,width=1024),n,seed=0)))
            self.assertEqual(len({(c.yaw,c.pitch) for c in cams}),n)
            for c in cams:
                self.assertAlmostEqual(float(c.forward().norm()),1,places=6)
                self.assertTrue(-math.pi<=c.yaw<math.pi);self.assertTrue(-math.pi/2<=c.pitch<=math.pi/2);self.assertEqual(c.roll,0.)
            source=torch.Generator().manual_seed(0);before=source.get_state().clone()
            random_uniform_camera_cover(self.view,n,seed=0)
            self.assertTrue(torch.equal(before,source.get_state()))
    def test_random_spherical_area_statistics(self):
        cams=random_uniform_camera_cover(self.view,20000,seed=1234)
        z=torch.tensor([math.sin(c.pitch) for c in cams]);yaw=torch.tensor([c.yaw for c in cams])
        self.assertLess(abs(float(z.mean())),.025);self.assertLess(abs(float(z.square().mean())-1/3),.025)
        self.assertLess(abs(float((z>0).float().mean())-.5),.025)
        self.assertLess(abs(float(yaw.sin().mean())),.025);self.assertLess(abs(float(yaw.cos().mean())),.025)
        bins=torch.histc(yaw,bins=12,min=-math.pi,max=math.pi)/len(cams)
        self.assertLess(float((bins-1/12).abs().max()),.015)
    def test_ids_and_phase_seed_independence(self):
        self.assertEqual(stable_ids('fibonacci',89)[-1],'fibonacci:n089:088')
        self.assertEqual(stable_ids('random',89)[-1],'random:n089:seed0:088')
        self.assertEqual(cover('fibonacci',self.view,89,layout_seed=0),cover('fibonacci',self.view,89,layout_seed=876))
        torch.manual_seed(77);a=fibonacci_camera_cover(self.view,89);torch.manual_seed(9876)
        self.assertEqual(a,fibonacci_camera_cover(self.view,89))

    def test_numeric_representation_does_not_change_angular_hash(self):
        for strategy in ('fibonacci','random'):
            integer_view=ViewConfig(height=1024,width=1024,fov_x=80,fov_y=80)
            float_view=replace(integer_view,fov_x=80.0,fov_y=80.0)
            self.assertEqual(angular_hash(cover(strategy,integer_view,89)),angular_hash(cover(strategy,float_view,89)))
