import math
import unittest
from dataclasses import replace
import torch
from diffpano.camera import camera_for_direction,PerspectiveCamera
from diffpano.config import WarpConfig,FusionConfig
from diffpano.projection import ProjectionCache,perspective_world_rays,erp_to_perspective
from diffpano.fusion import RGBFusionAccumulator
from diffpano.warp import StandardWarpOperator
from studies.tt_cea.canvas import *
from studies.tt_cea.cameras import fibonacci_centers

class CanvasTests(unittest.TestCase):
    def operator(self,projection,h=24,w=48):
        return CanvasOperator(CanvasSpec(projection,h,w),WarpConfig(mode='standard'),
            FusionConfig(mode='weighted_average',weight_mode='spherediff_center',spherediff_temperature=.1),torch.device('cpu'))
    def test_float64_reference_unit_roundtrip_area_and_edges(self):
        h,w=64,128;r=cea_rays(h,w,torch.device('cpu'))
        y=(torch.arange(h,dtype=torch.float64)+.5)/h;x=(torch.arange(w,dtype=torch.float64)+.5)/w
        yy,xx=torch.meshgrid(y,x,indexing='ij');phi=torch.asin(1-2*yy);lon=2*math.pi*(xx-.5)
        ref=torch.stack((phi.cos()*lon.sin(),phi.sin(),phi.cos()*lon.cos()),-1)
        torch.testing.assert_close(r.double(),ref,atol=4e-7,rtol=4e-7)
        torch.testing.assert_close(r.norm(dim=-1),torch.ones(h,w),atol=2e-7,rtol=2e-7)
        torch.testing.assert_close(uv_directions(cea_uv(r)),r,atol=1e-6,rtol=1e-6)
        edges=1-2*torch.arange(h+1,dtype=torch.float64)/h
        area=(edges[:-1]-edges[1:])*2*math.pi/w
        torch.testing.assert_close(area,torch.full_like(area,4*math.pi/(h*w)),atol=1e-15,rtol=1e-14)
        self.assertAlmostEqual(float(area.sum())*w,4*math.pi,places=12)
    def test_poles_longitude_independence_cap_and_constant(self):
        h,w=32,64;rgb=torch.full((1,3,h,w),.7);poles=torch.tensor([.1,.3])[None,None,:,None].expand(1,3,2,1)
        result=CanvasResult(rgb,torch.ones(1,1,h,w),torch.ones(1,1,h,w),poles,torch.ones(1,1,2,1))
        for north in (True,False):
            theta0=math.acos(1-1/h)
            for mode in ('nearest','bilinear'):
                for angle in (0,theta0/2,theta0*(1-1e-5),theta0*(1+1e-5)):
                    lon=torch.linspace(-math.pi,math.pi,17);sign=1 if north else -1
                    rays=torch.stack((lon.sin()*math.sin(angle),torch.full_like(lon,sign*math.cos(angle)),lon.cos()*math.sin(angle)),-1).reshape(17,1,3)
                    value=sample_cea(result,rays,mode)
                    expected=(.1 if north else .3)*(1-min(angle/theta0,1))+.7*min(angle/theta0,1)
                    torch.testing.assert_close(value,torch.full_like(value,expected),atol=2e-6,rtol=2e-6)
        result.pole_rgb.fill_(.7)
        torch.testing.assert_close(sample_cea(result,cea_rays(48,96,torch.device('cpu')),'bilinear'),torch.full((1,3,48,96),.7))
    def test_coordinate_fields_low_frequency_wrap_roll_and_great_circle(self):
        h,w=128,256;r=cea_rays(h,w,torch.device('cpu'));rgb=r.permute(2,0,1)[None]
        poles=torch.tensor([[0.,1.,0.],[0.,-1.,0.]]).T[None,:,:,None]
        result=CanvasResult(rgb,torch.ones(1,1,h,w),torch.ones(1,1,h,w),poles,torch.ones(1,1,2,1))
        for yaw,pitch,roll in ((180,0,0),(-180,0,37),(32,90,15),(30,-90,70),(20,45,90)):
            cam=camera_for_direction(yaw,pitch,roll_degrees=roll,height=33,width=33)
            rays=perspective_world_rays(cam,device=torch.device('cpu'))
            actual=sample_cea(result,rays,'bilinear');ref=rays.permute(2,0,1)[None]
            self.assertLess(float((actual-ref).abs().max()),.045)
        # Continuous seam and a smooth great-circle marker through the x=0 plane.
        rgb=torch.exp(-20*r[...,0].square())[None,None].expand(1,3,h,w)
        poles=torch.ones(1,3,2,1);result.rgb=rgb;result.pole_rgb=poles
        rays=torch.tensor([[[1e-6,0.,-1.]],[[-1e-6,0.,-1.]]])
        v=sample_cea(result,rays,'bilinear');torch.testing.assert_close(v[:,:,0],v[:,:,1],atol=1e-6,rtol=1e-6)
        self.assertGreater(float(v.min()),.99)
    def test_weighted_constant_and_erp_exact_delegation_cache_isolation(self):
        cams=[PerspectiveCamera(**p,height=12,width=12) for p in fibonacci_centers()]
        cache=ProjectionCache(max_entries=2,cpu_fallback=False)
        erp=self.operator('erp');cea=self.operator('cea');erp.cache=cache;erp.standard.cache=cache;cea.cache=cache;cea.standard.cache=cache
        for op in (erp,cea):
            acc=op.make_accumulator(1)
            for cam in cams:op.accumulate(acc,torch.full((1,3,12,12),.25),cam)
            result=op.finalize(acc)
            torch.testing.assert_close(result.rgb,torch.full_like(result.rgb,.25),atol=2e-7,rtol=2e-7)
            self.assertTrue(bool((result.weight_sum>0).all()))
            view=op.sample_view(result,cams[0]);torch.testing.assert_close(view,torch.full_like(view,.25),atol=2e-7,rtol=2e-7)
            self.assertLessEqual(len(cache.view_to_erp),2);self.assertLessEqual(len(cache.erp_to_view),2)
        direct=StandardWarpOperator(erp.standard.warp_config,erp.fusion_config)
        a=erp.make_accumulator(1);b=RGBFusionAccumulator(torch.zeros(1,3,24,48),erp.fusion_config)
        for i,cam in enumerate(cams):
            rgb=torch.full((1,3,12,12),i/89)
            erp.accumulate(a,rgb,cam);b.accumulate(direct.perspective_to_erp(rgb,cam,(24,48)))
        self.assertTrue(torch.equal(erp.finalize(a).rgb,b.finalize().erp_rgb))
        self.assertTrue(any(k[0]=='cea' for k in cache.erp_to_view))
    def test_uncovered_rejected(self):
        for projection in ('erp','cea'):
            op=self.operator(projection)
            with self.assertRaises(ValueError):op.finalize(op.make_accumulator(1))
