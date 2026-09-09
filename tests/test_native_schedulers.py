"""Real first-order scheduler controls without downloading or loading weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from diffpano.pipelines.flux import FluxViewDenoiser, calculate_shift
from diffpano.pipelines.sana import SanaViewDenoiser
from diffpano.pipelines.sd2 import SD2ViewDenoiser


class NativeSchedulerTests(unittest.TestCase):
    def test_reset_steps_match_uninterrupted_order_one_sampling(self):
        from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
        setups = [
            (SD2ViewDenoiser, '_predict_noise', DDIMScheduler(clip_sample=False)),
            (SanaViewDenoiser, '_predict_flow', DPMSolverMultistepScheduler(
                prediction_type='flow_prediction', use_flow_sigmas=True, solver_order=1)),
        ]
        for cls, predictor, scheduler in setups:
            reference = type(scheduler).from_config(scheduler.config)
            scheduler.set_timesteps(5)
            reference.set_timesteps(5)
            adapter = cls(SimpleNamespace(scheduler=scheduler), guidance_scale=1)
            prediction = lambda state, t, c: state * .1
            setattr(adapter, predictor, prediction)
            actual = torch.randn(2, 4, 8, 8)
            expected = actual.clone()
            for t in scheduler.timesteps:
                expected = reference.step(prediction(expected, t, None), t, expected, return_dict=False)[0]
                actual = adapter.denoise_native_step(actual, t, None)
                torch.testing.assert_close(actual, expected)

    def test_flux_real_packing_image_ids_and_single_dynamic_shift(self):
        from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline
        scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
        observed = []
        def forward(**kwargs):
            observed.append(kwargs)
            return (torch.zeros_like(kwargs['hidden_states']),)
        pipeline = SimpleNamespace(
            scheduler=scheduler, vae_scale_factor=8, _execution_device='cpu',
            transformer=SimpleNamespace(dtype=torch.float32,
                config=SimpleNamespace(in_channels=16, guidance_embeds=True), __call__=forward),
            _pack_latents=FluxPipeline._pack_latents,
            _unpack_latents=FluxPipeline._unpack_latents,
            _prepare_latent_image_ids=FluxPipeline._prepare_latent_image_ids,
        )
        # Special methods are resolved on the type, not the instance namespace.
        class Transformer:
            dtype = torch.float32
            config = SimpleNamespace(in_channels=16, guidance_embeds=True)
            def __call__(self, **kwargs):
                return forward(**kwargs)
        pipeline.transformer = Transformer()
        adapter = FluxViewDenoiser(pipeline, guidance_scale=3.5)
        adapter.prepare(num_steps=4, view_height=32, view_width=64)
        reference = FlowMatchEulerDiscreteScheduler.from_config(scheduler.config)
        reference.set_timesteps(sigmas=np.linspace(1, .25, 4), mu=calculate_shift(8))
        torch.testing.assert_close(scheduler.sigmas, reference.sigmas)
        condition = {'embeds': torch.zeros(1, 2, 4), 'pooled': torch.zeros(1, 4), 'text_ids': torch.zeros(2, 3)}
        native = torch.randn(1, 4, 4, 8)
        for _ in range(2):
            result = adapter.denoise_native_step(native, adapter.timesteps[1], condition)
            torch.testing.assert_close(result, native)
        self.assertEqual(observed[0]['img_ids'].shape, (8, 3))
        self.assertEqual(observed[0]['hidden_states'].shape, (1, 8, 16))
        self.assertTrue(torch.equal(observed[0]['img_ids'], observed[1]['img_ids']))
        self.assertEqual(adapter.scheduler_image_seq_len, 8)

    def test_pixel_native_step_uses_same_official_first_update(self):
        from test_x0_consensus import pixel_adapter
        adapter = pixel_adapter()
        adapter.prepare(num_steps=3, view_height=4, view_width=4)
        reference = SimpleNamespace(
            model_fn=Mock(side_effect=lambda state, t: torch.ones_like(state)),
            dpm_solver_first_update=Mock(side_effect=lambda state, a, b, model_s: state*.5 + model_s),
        )
        adapter._official_solver = lambda state, condition: reference
        adapter.predict_clean_native = Mock(side_effect=AssertionError('consensus called'))
        state = torch.randn(1, 3, 4, 4)
        native = adapter.denoise_native_step(state, adapter.timesteps[0], None)
        rgb = adapter.denoise_step(state, adapter.timesteps[0], None)
        self.assertTrue(torch.equal(native, rgb))
        self.assertEqual(reference.model_fn.call_count, 2)
        self.assertEqual(reference.dpm_solver_first_update.call_count, 2)
        self.assertEqual(adapter.native_spatial_factor, 1)


if __name__ == '__main__':
    unittest.main()
