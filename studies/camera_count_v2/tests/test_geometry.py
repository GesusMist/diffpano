import math
import unittest
from dataclasses import replace
from unittest.mock import patch
import torch
from diffpano.config import ViewConfig
from diffpano.camera import PerspectiveCamera
from studies.camera_count_v2.common import rows,expected_counts
from studies.camera_count_v2.cameras import build,ids,RINGS,RING_COUNTS,layout_record,restore
from studies.camera_count_v2.spherediff_adapter import reference_geometry,directions,physical_axes,override,FACTORY,geometry_audit
from studies.camera_count_v2.coverage import quick_pass,special_rays
from studies.camera_patching.cameras import fibonacci_camera_cover,random_uniform_camera_cover,angular_hash

class CameraTests(unittest.TestCase):
    def test_matrix_and_call_counts(self):
        r=rows();self.assertEqual(len(r),126)
        self.assertEqual(sum(x['family']=='spherediff_camera_override' for x in r),18)
        for x in r:
            if x['family']=='spherediff_camera_override':
                self.assertEqual(x['strategy'],'old');self.assertEqual(x['steps'],20)
            self.assertEqual(expected_counts(x)['denoiser'],x['steps']*x['num_cameras'])
    def test_explicit_rings_and_polar_orientation(self):
        for n in (70,50,30):
            c=build('old',n);self.assertEqual(len(c),n)
            offset=0
            for p,m in zip(RINGS,RING_COUNTS[n]):
                ring=c[offset:offset+m];offset+=m
                self.assertTrue(all(abs(math.degrees(v.pitch)-p)<1e-9 for v in ring))
                self.assertEqual([v.yaw for v in ring],[-math.pi+2*math.pi*j/m for j in range(m)])
            self.assertEqual(len(set(ids('old',n))),n)
            self.assertTrue(all('old-rings-v2' in x for x in ids('old',n)))
            d=directions(c).bfloat16();self.assertEqual(len(torch.unique(d.float(),dim=0)),n)
            for v in layout_record('old',n)['poses']:self.assertNotIn('width',v)
            self.assertEqual(restore(layout_record('old',n)),c)
    def test_existing_generic_generators_and_rng(self):
        v=ViewConfig(height=1024,width=1024,fov_x=80.,fov_y=80.)
        before=torch.random.get_rng_state()
        for n in (70,50,30):
            self.assertEqual(build('fibonacci',n),fibonacci_camera_cover(v,n,phase=0.))
            self.assertEqual(build('random',n,7),random_uniform_camera_cover(v,n,seed=7))
            self.assertNotEqual(build('fibonacci',n),fibonacci_camera_cover(v,89)[:n])
        torch.testing.assert_close(before,torch.random.get_rng_state(),rtol=0,atol=0)
        self.assertEqual(angular_hash(fibonacci_camera_cover(v,89)),
                         'aabecc0b2d29255a67ffcc29b0e80168358b00c34671423cfe1c488bfbb74499')
    def test_override_restore_and_raw_reference(self):
        sf=reference_geometry();fn=getattr(sf,FACTORY);original=fn()
        rng=torch.random.get_rng_state()
        with override(raw_directions=original,sf=sf):
            torch.testing.assert_close(getattr(sf,FACTORY)(),original,rtol=0,atol=0)
        self.assertIs(getattr(sf,FACTORY),fn)
        with self.assertRaises(RuntimeError):
            with override(build('old',30),sf=sf):
                self.assertEqual(len(getattr(sf,FACTORY)()),30);raise RuntimeError('deliberate')
        self.assertIs(getattr(sf,FACTORY),fn)
        torch.testing.assert_close(rng,torch.random.get_rng_state(),rtol=0,atol=0)
    def test_full_physical_axes_and_semantic_bands(self):
        for n in (70,50,30):self.assertTrue(geometry_audit(build('old',n))['passed'])
        c=tuple(PerspectiveCamera(y,math.radians(p),0.,80.,80.,33,33)
                for y in (-1.2,.4,2.3) for p in (90,60,10,0,-10,-60,-90))
        self.assertEqual(geometry_audit(c)['semantic_band_disagreements'],[])
    def test_native_corner_raster_extraction(self):
        sf=reference_geometry()
        for c in [build('old',70)[i] for i in (0,1,2,10,34,67,68,69)]:
            t=math.tan(math.radians(40));x=torch.linspace(-t,t,3);y=torch.linspace(t,-t,3)
            xx,yy=torch.meshgrid(x,y,indexing='xy')
            local=torch.stack((xx,yy,torch.ones_like(xx)),-1)
            world=torch.einsum('ij,hwj->hwi',c.rotation(),local)
            world=world/world.norm(dim=-1,keepdim=True)
            d=directions([c])
            ix=sf.extract_perspective_from_spherical_rectangle_rasterize(world.reshape(1,1,9,3),d,
                output_size=(3,3),device=torch.device('cpu'),dtype=torch.float32)
            self.assertEqual(ix.tolist(),list(range(9)))
            uv=sf.world_to_perspective(world.reshape(1,9,3),d,(80,80))
            expected=torch.stack((-xx/t,-yy/t),-1).reshape(9,2)
            torch.testing.assert_close(uv,expected,rtol=1e-4,atol=1e-4)
    def test_holes_poles_wrap(self):
        self.assertFalse(quick_pass(build('old',30)[:1],8192,'cpu'))
        self.assertTrue(quick_pass(build('old',70),8192,'cpu'))
        rays=special_rays()
        self.assertTrue(torch.equal(rays[-2:],torch.tensor([[0.,1.,0.],[0.,-1.,0.]])))
        self.assertTrue(torch.isfinite(rays).all())

if __name__=='__main__':unittest.main()
