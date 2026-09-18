import unittest
import weakref
from dataclasses import replace
from unittest.mock import patch
import torch
from diffpano.camera import camera_for_direction,CubeFixedCameraSampler
from diffpano.config import ViewConfig,FusionConfig,WarpConfig
from diffpano.erp_noise_initialization import (ERPNoiseConfig,NearestERPIndexProjector,sample_source,
    initialize_erp_noise,native_cameras,primary_noise_size,map_statistics,states_digest)
from diffpano.pipelines.native_state import NativeStateMixin
from diffpano.projection import erp_to_perspective
from diffpano.dense_consensus import DenseERPLocalCurrentStatePipeline
from diffpano.warp import StandardWarpOperator
from diffpano.noise_v_diagnostics import NoiseVAudit,align_clean_views
from scripts.noise_v_preflight import covariance_check,matched_ray_pairs
from scripts.noise_v_common import shape_backend,s_config
from test_erp_local_consensus import LocalMock


class ScaleStub(NativeStateMixin):
    native_channels=7;native_spatial_factor=2;native_initial_noise_sigma=2.5;device=torch.device('cpu')
    def encode_clean(self,*a):raise AssertionError('No VAE initialization')
    def decode_clean(self,*a):raise AssertionError('No VAE initialization')


class ERPNoiseTests(unittest.TestCase):
    def cameras(self,size=12):
        return CubeFixedCameraSampler(ViewConfig(height=size,width=size,fov_x=100,fov_y=100)).sample(0,5)

    def test_arbitrary_channels_native_shapes_no_vae_scaling_once(self):
        flux,_=shape_backend('flux')
        for channels in (3,7,flux.native_channels):
            b=ScaleStub();b.native_channels=channels;cams=self.cameras();config=ERPNoiseConfig('V-shared-erp',17,34,17)
            states,record=initialize_erp_noise(b,cams,config)
            field=torch.randn(1,channels,17,34,generator=torch.Generator().manual_seed(17))
            index=NearestERPIndexProjector(17,34).index_map(native_cameras(b,cams)[0])
            self.assertTrue(torch.equal(states[0],sample_source(field,index)*2.5))
            self.assertEqual(tuple(states[0].shape),(1,channels,6,6));self.assertEqual(record['scaling_applications'],1)
            self.assertFalse(torch.equal(states[0][:,0],states[0][:,1]))

    def test_nearest_geometry_matches_rgb_at_centers_wrap_poles_roll_and_ties(self):
        g=torch.Generator().manual_seed(81)
        for h,w in [(2,4),(31,62),(240,480)]:
            field=torch.randn(1,3,h,w,generator=g);projector=NearestERPIndexProjector(h,w)
            for yaw,pitch,roll,size in [(0,0,0,1),(180,0,0,17),(-180,0,0,16),(11,90,30,17),(43,-90,0,16),(7,45,21,17)]:
                cam=camera_for_direction(yaw,pitch,roll_degrees=roll,height=size,width=size)
                index=projector.index_map(cam);actual=sample_source(field,index)
                expected=erp_to_perspective(field,cam,interpolation='nearest')
                self.assertTrue(torch.equal(actual,expected));self.assertGreaterEqual(int(index.min()),0);self.assertLess(int(index.max()),h*w)

    def test_same_cell_and_duplicate_values_are_exact(self):
        index=torch.tensor([[0,0,1],[2,2,2]],dtype=torch.int64)
        field=torch.randn(1,7,2,4,generator=torch.Generator().manual_seed(12));out=sample_source(field,index)
        self.assertTrue(torch.equal(out[...,0,0],out[...,0,1]));self.assertTrue(torch.equal(out[...,1,0],out[...,1,2]))
        m=map_statistics(index);self.assertEqual(m['unique_source_fraction'],.5);self.assertEqual(m['max_source_reuse'],3)

    def test_covariance_patterns_independent_shared_channels(self):
        left=torch.arange(64);right=left.clone();right[32:]+=100
        a=covariance_check(left,right,draws=4096);b=covariance_check(left,right,independent=True,draws=4096)
        c=covariance_check(left,left,cross_channel=True,draws=4096)
        self.assertEqual(a['expected_covariance'],.5);self.assertEqual(b['expected_covariance'],0);self.assertEqual(c['expected_covariance'],0)

    def test_rng_isolation_first_camera_pairing_and_source_release(self):
        b=ScaleStub();cams=self.cameras();original=torch.randn;refs=[]
        def tracked(*args,**kwargs):
            x=original(*args,**kwargs);refs.append(weakref.ref(x));return x
        before=torch.random.get_rng_state().clone()
        with patch('torch.randn',side_effect=tracked):
            a,ma=initialize_erp_noise(b,cams,ERPNoiseConfig('V-independent-erp',17,34,44))
        self.assertEqual(len(refs),len(cams));self.assertTrue(all(r() is None for r in refs))
        z,mz=initialize_erp_noise(b,cams,ERPNoiseConfig('V-shared-erp',17,34,44))
        covariance_check(torch.arange(10),torch.arange(10),draws=512)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(a[0],z[0]));self.assertFalse(torch.equal(a[1],z[1]));self.assertEqual(ma['map_sha256'],mz['map_sha256'])
        self.assertEqual(ma['source_draws'],6);self.assertEqual(mz['source_draws'],1);self.assertTrue(mz['source_released'])

    def test_direct_initializer_matches_existing_backend_exactly(self):
        b=ScaleStub();cams=self.cameras();states,_=initialize_erp_noise(b,cams,ERPNoiseConfig('S-direct-local',17,34,14))
        g=torch.Generator().manual_seed(14)
        for state,c in zip(states,native_cameras(b,cams)):
            expected=b.sample_initial_native_state(batch_size=1,native_height=c.height,native_width=c.width,generator=g)
            self.assertTrue(torch.equal(state,expected))

    def test_generation_no_source_reuse_same_calls_finite_order_invariant(self):
        for variant in ['S-direct-local','V-independent-erp','V-shared-erp']:
            outputs=[];initial_hashes=[]
            for reverse in (False,True):
                b=LocalMock();cams=self.cameras();op=StandardWarpOperator(WarpConfig(mode='standard'),FusionConfig(mode='weighted_average',weight_mode='spherediff_center'))
                pipe=DenseERPLocalCurrentStatePipeline(backend=b,cameras=cams,erp_size=(16,32),warp_operator=op,view_order=list(reversed(range(6))) if reverse else None)
                states,record=initialize_erp_noise(b,cams,ERPNoiseConfig(variant,17,34,8));initial_hashes.append(states_digest(states))
                if variant=='S-direct-local':self.assertEqual(states_digest(pipe.initialize_local_states(8)),states_digest(states))
                with patch('diffpano.erp_noise_initialization.sample_source',side_effect=AssertionError('One-time source only')),patch.object(b,'add_fixed_noise',side_effect=AssertionError('No reinjection')):
                    result=pipe.run_dense(states,[{'offset':torch.tensor(.3)}]*6)
                self.assertEqual(b.guided_prediction_count,30);self.assertTrue(bool(torch.isfinite(result.erp_rgb).all()))
                for seen,i in zip(b.forward_inputs[:6],pipe.view_order):self.assertTrue(torch.equal(seen,states[i]))
                outputs.append(result.erp_rgb)
            self.assertEqual(*initial_hashes);torch.testing.assert_close(*outputs,atol=2e-6,rtol=2e-6)

    def test_v_diagnostics_leave_trajectory_rng_and_calls_unchanged(self):
        # Cover includes overlapping equatorial views, upper and polar selections.
        from diffpano.camera import spherediff_camera_cover
        cams=spherediff_camera_cover(ViewConfig(height=12,width=12))
        outputs=[]
        for enabled in (False,True):
            b=LocalMock();op=StandardWarpOperator(WarpConfig(mode='standard'),FusionConfig(mode='weighted_average',weight_mode='spherediff_center'))
            pipe=DenseERPLocalCurrentStatePipeline(backend=b,cameras=cams,erp_size=(16,32),warp_operator=op)
            states=pipe.initialize_local_states(9);rng=torch.random.get_rng_state().clone()
            result=pipe.run_dense(states,[{'offset':torch.tensor(.3)}]*len(cams),stage_audit=NoiseVAudit(cams,5) if enabled else None)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()));self.assertEqual(b.guided_prediction_count,445)
            self.assertEqual(b.decode_count,534);outputs.append(result.erp_rgb)
        self.assertTrue(torch.equal(*outputs))

    def test_correspondence_uses_world_rays(self):
        a=camera_for_direction(0,0,height=32,width=32);b=camera_for_direction(24,0,height=32,width=32)
        ia,ib,angles=matched_ray_pairs(a,b)
        self.assertTrue(bool((ia!=ib).any()));self.assertLess(float(angles.max()),3.)
        i,j,angle=matched_ray_pairs(a,a);self.assertTrue(torch.equal(i,j));self.assertLess(float(angle.max()),1e-5)
        image=torch.ones(1,3,32,32);metrics,_,_=align_clean_views(image,image,a,b)
        # Four FP32 bilinear weights can accumulate a few rounding ulps.
        self.assertLessEqual(metrics['mae'],4*torch.finfo(torch.float32).eps)

    def test_native_grid_rule_and_exact_s_config(self):
        from diffpano.camera import spherediff_camera_cover
        for name,expected in [('pixeldit',(1917,3834)),('flux',(240,480))]:
            b,_=shape_backend(name);c,s=s_config(name);cams=spherediff_camera_cover(c.view)
            self.assertEqual(primary_noise_size(native_cameras(b,cams)),expected);self.assertEqual(c.to_dict(),s['config'])
            self.assertEqual((c.erp.height,c.erp.width),(1024,2048));self.assertEqual(c.fusion.spherediff_temperature,.1)
        with self.assertRaises(ValueError):ERPNoiseConfig('V-shared-erp',10,21,0).validate()

if __name__=='__main__':unittest.main()
