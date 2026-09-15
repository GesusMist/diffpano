import unittest
from dataclasses import replace
from unittest.mock import patch
import torch

from diffpano.camera import CubeFixedCameraSampler,camera_for_direction
from diffpano.config import ViewConfig,FusionConfig,WarpConfig
from diffpano.erp_local_consensus import ERPLocalCurrentStatePipeline,geometry_preflight,camera_digest
from diffpano.warp import StandardWarpOperator
from test_detail_preserving_consensus import ContextMock


class LocalMock(ContextMock):
    native_channels=4
    def encode_clean(self,rgb):
        self.encodes+=1
        return torch.cat([rgb,rgb.mean(dim=1,keepdim=True)],dim=1)
    def decode_clean(self,state):
        self.decode_count+=1
        return state[:,:3]


class ERPLocalTests(unittest.TestCase):
    def make(self,order=None):
        b=LocalMock();v=ViewConfig(height=12,width=12,fov_x=100,fov_y=100)
        cameras=CubeFixedCameraSampler(v).sample(0,5)
        operator=StandardWarpOperator(WarpConfig(mode='standard'),FusionConfig(mode='average',weight_mode='uniform'))
        p=ERPLocalCurrentStatePipeline(backend=b,cameras=cameras,erp_size=(16,32),warp_operator=operator,view_order=order)
        return b,p

    def test_fixed_slots_and_deterministic_local_gaussian_stream(self):
        b,p=self.make();a=p.initialize_local_states(18);again=p.initialize_local_states(18)
        different=p.initialize_local_states(19)
        generator=torch.Generator().manual_seed(18)
        for x,y,z in zip(a,again,different):
            self.assertTrue(torch.equal(x,y));self.assertFalse(torch.equal(x,z))
            self.assertEqual(tuple(x.shape),(1,4,12,12))
            expected=torch.randn(1,4,12,12,generator=generator)*b.native_initial_noise_sigma
            self.assertTrue(torch.equal(x,expected))
        sampler=CubeFixedCameraSampler(ViewConfig(height=12,width=12,fov_x=100,fov_y=100))
        self.assertEqual(camera_digest(sampler.sample(0,5)),camera_digest(sampler.sample(4,5)))
        p.cameras=(replace(p.cameras[0],yaw=.1),)+p.cameras[1:]
        with self.assertRaisesRegex(AssertionError,'Camera slots changed'):p.run(a,{'offset':torch.tensor(.3)})

    def test_full_cover_constant_geometry_and_gap_rejection(self):
        b,p=self.make();stats,counts=geometry_preflight(p.cameras,p.erp_size,p.operator)
        self.assertEqual(stats['coverage_percent'],100.)
        self.assertEqual(stats['minimum_contributors'],1.)
        self.assertGreater(stats['multi_contributor_percent'],0.)
        with self.assertRaisesRegex(ValueError,'gaps'):geometry_preflight(p.cameras[:1],p.erp_size,p.operator)

    def test_standard_roundtrip_and_horizontal_wrap_feature(self):
        size=64;h,w=64,128
        cameras=CubeFixedCameraSampler(ViewConfig(height=size,width=size,fov_x=100,fov_y=100)).sample(0,1)
        op=StandardWarpOperator(WarpConfig(mode='standard'),FusionConfig(mode='average',weight_mode='uniform'))
        lon=(torch.arange(w)+.5)/w*2*torch.pi-torch.pi
        lat=(torch.arange(h)+.5)/h*torch.pi-torch.pi/2
        feature=torch.stack([(-((torch.pi-lon.abs())/.4).square()).exp().expand(h,-1),
            lon.sin().expand(h,-1),lat.sin()[:,None].expand(-1,w)])[None]
        b,p=self.make();p.cameras=tuple(cameras);p.camera_sha256=camera_digest(cameras);p.erp_size=(h,w);p.operator=op
        views=[op.erp_to_perspective(feature,c) for c in cameras]
        fused,_=p._project_and_fuse(views,{})
        self.assertLess(float((fused.erp_rgb-feature).abs().mean()),.025)
        for c,v in zip(cameras,views):
            back=op.erp_to_perspective(fused.erp_rgb,c)
            self.assertLess(float((back[...,8:-8,8:-8]-v[...,8:-8,8:-8]).abs().mean()),.035)
        crossing=op.erp_to_perspective(feature,cameras[2])
        self.assertGreater(float(crossing[0,0,size//2,size//2]),.95)
        seam=fused.erp_rgb[0,0,h//3:2*h//3][:,[0,-1]]
        target=feature[0,0,h//3:2*h//3][:,[0,-1]]
        self.assertLess(float((seam-target).abs().max()),.05)

    def test_jacobi_counts_rgb_only_no_residual_or_fixed_noise(self):
        outputs=[]
        for order in (list(range(6)),list(reversed(range(6)))):
            b,p=self.make(order);initial=p.initialize_local_states(7)
            with patch('diffpano.vae_residual.vae_residual',side_effect=AssertionError('No VAE residual')), \
                 patch('diffpano.noise.FixedPatchNoiseBank',side_effect=AssertionError('No fixed noise')), \
                 patch.object(p.operator,'perspective_to_erp',wraps=p.operator.perspective_to_erp) as project:
                result=p.run(initial,{'offset':torch.tensor(.3)})
            self.assertEqual(b.guided_prediction_count,30);self.assertEqual(b.encodes,30)
            self.assertEqual(b.decode_count,36)
            for seen,i in zip(b.forward_inputs[:6],order):self.assertTrue(torch.equal(seen,initial[i]))
            for call in project.call_args_list:self.assertEqual(call.args[0].shape[1],3)
            self.assertTrue(all(tuple(s.shape)==(1,4,12,12) for s in result.local_states))
            self.assertEqual(tuple(result.erp_rgb.shape),(1,3,16,32))
            self.assertFalse(result.audit['global_native_state_persisted'])
            self.assertFalse(result.audit['vae_residual_correction'])
            self.assertFalse(result.audit['fixed_initial_noise_renoising'])
            self.assertEqual(set(result.audit['camera_hashes_per_step']),{p.camera_sha256})
            self.assertLess(max(r['i_current_state_error_max_abs'] for r in result.transition_records),2e-6)
            outputs.append(result.erp_rgb)
        self.assertTrue(torch.equal(*outputs))

    def test_k_configs_standard_geometry_and_historical_settings(self):
        from scripts.erp_standard_current_experiment import checked_config,read
        specs=read('configs/experiments/erp_later/k-all-models.json')
        for spec in specs.values():
            c=checked_config(spec)
            self.assertEqual(c.sampling.strategy,'cube6_fixed')
            self.assertFalse(c.consensus_transition.vae_residual_correction)
            invalid=replace(c,warp=WarpConfig(mode='lpw'))
            with self.assertRaises(ValueError):invalid.validate()

if __name__=='__main__':unittest.main()
