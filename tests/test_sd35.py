"""SD3.5 scheduler, native-coordinate and adapter oracles; no model downloads."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from diffpano.config import NativeMultiDiffusionConfig, load_experiment_config
from diffpano.pipelines.endpoints import flow_bounds, flow_endpoints
from diffpano.pipelines.base import reset_scheduler_step_state
from diffpano.planar import build_planar_patch_layout, extract_planar_patch
from diffpano.vae import encode_view_images, decode_view_latents
from diffpano.vae_residual import recover_identity, fuse_native_residuals


class SD35MathTests(unittest.TestCase):
    def test_checkpoint_static_shift_actual_sigmas_and_euler_equivalence(self):
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000,shift=3.0)
        scheduler.set_timesteps(40)
        self.assertFalse(scheduler.config.use_dynamic_shifting)
        self.assertEqual(float(scheduler.sigmas[-1]),0.)
        original=scheduler.sigmas.clone()
        generator=torch.Generator().manual_seed(31)
        state=torch.randn(1,16,8,8,generator=generator)
        velocity=torch.randn(1,16,8,8,generator=generator)
        for index,t in enumerate(scheduler.timesteps):
            sigma,following=flow_bounds(scheduler,t,state)
            self.assertEqual(float(sigma),float(original[index]))
            self.assertEqual(float(following),float(original[index+1]))
            pair=flow_endpoints(state,velocity,sigma,following)
            reset_scheduler_step_state(scheduler)
            native=scheduler.step(velocity,t,state,return_dict=False)[0]
            torch.testing.assert_close(pair.reconstruct_next(),native,atol=2e-6,rtol=2e-6)
        self.assertTrue(torch.equal(original,scheduler.sigmas))
        with self.assertRaises(ValueError):flow_bounds(scheduler,-1,state)

    def test_low_precision_scheduler_rounding_is_distinct_from_algebra(self):
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0);scheduler.set_timesteps(40)
        x=torch.randn(1,16,8,8);v=torch.randn_like(x).bfloat16();t=scheduler.timesteps[4]
        sigma,following=flow_bounds(scheduler,t,x)
        native=scheduler.step(v,t,x,return_dict=False)[0]
        self.assertTrue(torch.equal(native,(x+(following-sigma)*v).bfloat16()))
        reset_scheduler_step_state(scheduler)
        native32=scheduler.step(v.float(),t,x,return_dict=False)[0]
        torch.testing.assert_close(flow_endpoints(x,v.float(),sigma,following).reconstruct_next(),native32,atol=2e-6,rtol=2e-6)

    def test_sd35_deterministic_vae_scale_shift_and_identity(self):
        class VAE:
            dtype=torch.float32
            config=SimpleNamespace(scaling_factor=1.5305,shift_factor=.0609)
            def encode(self,rgb):
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:rgb*2,
                    sample=lambda:(_ for _ in ()).throw(AssertionError('posterior sampling'))))
            def decode(self,z,return_dict=False):return (z/2,)
        vae=VAE();rgb=torch.randn(1,16,8,8)
        encoded=encode_view_images(vae,rgb)
        torch.testing.assert_close(encoded,(rgb*2-.0609)*1.5305)
        torch.testing.assert_close(decode_view_latents(vae,encoded),rgb)
        z=torch.randn(1,16,8,8);rt=z*.3+.2
        torch.testing.assert_close(recover_identity(z,rt),z,atol=1e-6,rtol=1e-6)

    def test_sixteen_channel_residual_native_placement(self):
        layout=build_planar_patch_layout(128,256,128,64)
        global_field=torch.randn(1,16,128,256,generator=torch.Generator().manual_seed(9))
        proposals=[extract_planar_patch(global_field,p).clone() for p in layout.patches]
        fused=fuse_native_residuals(layout,proposals)
        torch.testing.assert_close(fused,global_field,atol=0,rtol=0)
        rgb=build_planar_patch_layout(1024,2048,1024,512)
        for native,image,proposal in zip(layout.patches,rgb.patches,proposals):
            self.assertEqual((image.x,image.y,image.size),(native.x*8,native.y*8,native.size*8))
            self.assertTrue(torch.equal(extract_planar_patch(fused,native),proposal))


class SD35AdapterTests(unittest.TestCase):
    def backend(self):
        from diffpano.pipelines.sd35 import SD35ViewDenoiser
        class VAE:
            dtype=torch.float32
            config=SimpleNamespace(latent_channels=16,scaling_factor=1.5305,shift_factor=.0609)
            def encode(self,rgb):
                native=torch.nn.functional.avg_pool2d(rgb,8).repeat(1,6,1,1)[:,:16]
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:native))
            def decode(self,z,return_dict=False):
                return (torch.nn.functional.interpolate(z[:,:3],scale_factor=8,mode='nearest'),)
        transformer=Mock(dtype=torch.bfloat16,config=SimpleNamespace(in_channels=16,patch_size=2))
        transformer.side_effect=lambda **kw:(kw['hidden_states']*.2+kw['hidden_states'].mean()*.1,)
        pipe=SimpleNamespace(_execution_device='cpu',transformer=transformer,vae=VAE(),vae_scale_factor=8,
            scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0),text_encoder=object(),text_encoder_2=object(),text_encoder_3=object())
        pipe.encode_prompt=Mock(return_value=(torch.ones(1,4,5),torch.zeros(1,4,5),torch.ones(1,6),torch.zeros(1,6)))
        backend=SD35ViewDenoiser(pipe,guidance_scale=4.5)
        backend.prepare(num_steps=3,view_height=64,view_width=64)
        return backend

    def test_registry_and_all_seven_configs(self):
        import json
        from pathlib import Path
        from diffpano.pipelines import build_view_denoiser
        protocol=json.loads(Path('configs/experiments/trajectory/sd35-ladder.json').read_text())
        configs={label:load_experiment_config(path) for label,path in protocol['configs'].items()}
        self.assertEqual(set(configs),set('ABCDEFG'))
        for c in configs.values():
            self.assertEqual(c.model.pipeline,'sd35');self.assertEqual(c.generation.num_inference_steps,40)
            self.assertEqual(c.model.id,'stabilityai/stable-diffusion-3.5-medium')
            self.assertEqual(c.model.revision,configs['A'].model.revision)
        for labels in ('ABCD','EFG'):
            geometry=[configs[k].native_multidiffusion for k in labels]
            self.assertTrue(all(n==geometry[0] for n in geometry))
        with patch('diffpano.pipelines.sd35.SD35ViewDenoiser.from_pretrained',return_value='backend') as loader:
            self.assertEqual(build_view_denoiser(configs['A']),'backend')
            self.assertEqual(loader.call_args.kwargs['torch_dtype'],torch.bfloat16)
        bad=copy.deepcopy(configs['A']);bad.model.pipeline='sd3'
        with self.assertRaises(ValueError):bad.validate()

    def test_raw_state_timestep_cfg_native_step_and_conditioning(self):
        b=self.backend();bank=b.prepare_prompt_conditioning(['same']*5)
        self.assertEqual(b.pipeline.encode_prompt.call_count,1)
        self.assertIsNone(b.pipeline.text_encoder_3)
        cond=b.conditioning_for_prompt_indices(bank,[8],batch_size=1)
        self.assertEqual(tuple(cond['embeds'].shape),(2,4,5))
        self.assertTrue(bool((cond['embeds'][0]==0).all()))
        epsilon=torch.randn(1,16,8,8);state=b.initialize_native_state(epsilon)
        self.assertTrue(torch.equal(state,epsilon))
        for t in b.timesteps:
            result,pair=b.native_step_with_endpoints(state,t,cond)
            torch.testing.assert_close(result,pair.reconstruct_next(),atol=2e-6,rtol=2e-6)
            kw=b.pipeline.transformer.call_args.kwargs
            self.assertEqual(tuple(kw['hidden_states'].shape),(2,16,8,8))
            self.assertTrue(torch.equal(kw['timestep'],t.expand(2)))
            self.assertEqual(b.last_model_prediction.dtype,torch.float32)
            state=result
        self.assertEqual(b.guided_prediction_count,3)
        with self.assertRaises(ValueError):b.predict_clean_native(torch.zeros(1,16,7,8),b.timesteps[0],cond)
        with self.assertRaises(ValueError):b.prepare(num_steps=3,view_height=65,view_width=64)

    def test_native_RGB_diagnostic_preserves_E_and_call_count(self):
        from diffpano.native_multidiffusion import NativeMultiDiffusionPipeline
        geometry=NativeMultiDiffusionConfig(canvas_height=8,canvas_width=16,patch_size=8,stride=4)
        initial=torch.randn(1,16,8,16,generator=torch.Generator().manual_seed(10))
        outputs=[]
        for enabled in (False,True):
            b=self.backend();bank=b.prepare_prompt_conditioning(['same']*5)
            result=NativeMultiDiffusionPipeline(native_config=geometry,backend=b,
                overlap_disagreement=True,clean_rgb_overlap_disagreement=enabled).run(initial,bank)
            self.assertEqual(b.guided_prediction_count,9)
            self.assertEqual(b.pipeline.transformer.call_count,9)
            outputs.append(result.native_canvas)
            if enabled:self.assertIn('pre_fusion_rgb_overlap_mae_mean',result.steps[0].state_statistics)
        torch.testing.assert_close(*outputs,atol=0,rtol=0)

    def test_G_order_initial_crops_identity_and_no_extra_forwards(self):
        from diffpano.implied_endpoint_consensus import PlanarImpliedEndpointConsensusPipeline
        from diffpano.vae_residual import single_patch_residual_trajectory
        geometry=NativeMultiDiffusionConfig(canvas_height=8,canvas_width=16,patch_size=8,stride=4)
        initial=torch.randn(1,16,8,16,generator=torch.Generator().manual_seed(9))
        outputs=[]
        for order in ([0,1,2],[2,1,0]):
            b=self.backend();bank=b.prepare_prompt_conditioning(['same']*5)
            pipe=PlanarImpliedEndpointConsensusPipeline(native_config=geometry,backend=b,patch_order=order,residual_correction=True)
            local=pipe.initialize_local_states(initial)
            for p,s in zip(pipe.native_layout.patches,local):
                self.assertTrue(torch.equal(s,extract_planar_patch(initial,p)))
            result=pipe.run(local,bank);outputs.append(result.canvas_rgb)
            self.assertEqual(b.pipeline.transformer.call_count,9)
            self.assertEqual(result.audit['guided_predictions'],9)
            self.assertEqual(result.audit['diagnostic_extra_denoiser_evaluations'],0)
        torch.testing.assert_close(*outputs,atol=0,rtol=0)
        b=self.backend();bank=b.prepare_prompt_conditioning(['same']*5)
        cond=b.conditioning_for_prompt_indices(bank,[8],batch_size=1)
        _,steps,count=single_patch_residual_trajectory(b,initial[...,:8],cond,correction=True)
        self.assertEqual(count,3)
        self.assertLess(max(row['correction_recovery_max_abs'] for row in steps),2e-6)


if __name__=='__main__':unittest.main()

