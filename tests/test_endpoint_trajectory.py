"""Independent endpoint oracles and three-way control invariants."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler

from diffpano.pipelines.endpoints import ddim_endpoints, flow_endpoints, flow_bounds, pixel_endpoints
from diffpano.pipelines.base import reset_scheduler_step_state
from diffpano.pipelines.native_state import NativeStateMixin
from diffpano.trajectory import compare_three_way, compare_single_patch


class OracleTests(unittest.TestCase):
    def test_ddim_epsilon_clipping_thresholding_and_final_alpha(self):
        torch.manual_seed(31)
        for options in ({'clip_sample': False}, {'clip_sample': True},
                        {'clip_sample': False, 'thresholding': True},
                        {'clip_sample': False, 'set_alpha_to_one': False}):
            scheduler = DDIMScheduler(num_train_timesteps=100, **options)
            scheduler.set_timesteps(7)
            state, prediction, fixed = [torch.randn(2, 4, 5, 5) for _ in range(3)]
            for t in scheduler.timesteps:
                endpoints = ddim_endpoints(scheduler, state, prediction, t)
                ordinary = scheduler.step(prediction, t, state, eta=0).prev_sample
                torch.testing.assert_close(endpoints.reconstruct_next(), ordinary, atol=2e-6, rtol=2e-6)
                if float(endpoints.next_sigma) > 0:
                    self.assertGreater(float((endpoints.reconstruct_next(fixed)-ordinary).abs().mean()), 1e-4)
            expected = scheduler.final_alpha_cumprod.sqrt()
            torch.testing.assert_close(endpoints.next_alpha.cpu(), expected)

    def test_straight_flow_oracle(self):
        x0, x1 = torch.randn(2, 7, 4, 5), torch.randn(2, 7, 4, 5)
        for sigma, following in ((1., .8), (.91, .52), (.15, 0.)):
            state = (1-sigma)*x0 + sigma*x1
            endpoints = flow_endpoints(state, x1-x0, sigma, following)
            torch.testing.assert_close(endpoints.clean, x0, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(endpoints.endpoint, x1, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(endpoints.reconstruct_next(), state+(following-sigma)*(x1-x0), atol=1e-6, rtol=1e-6)

    def test_actual_shifted_flow_schedulers_all_steps(self):
        schedulers = [FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
                      DPMSolverMultistepScheduler(solver_order=1, prediction_type='flow_prediction', use_flow_sigmas=True, flow_shift=3)]
        for scheduler in schedulers:
            if isinstance(scheduler, FlowMatchEulerDiscreteScheduler): scheduler.set_timesteps(20, mu=1.15)
            else: scheduler.set_timesteps(20)
            snapshot = scheduler.sigmas.clone()
            for t in scheduler.timesteps:
                state, velocity = torch.randn(2, 4, 5, 5), torch.randn(2, 4, 5, 5)
                sigma, following = flow_bounds(scheduler, t, state)
                endpoints = flow_endpoints(state, velocity, sigma, following)
                reset_scheduler_step_state(scheduler)
                ordinary = scheduler.step(velocity, t, state).prev_sample
                torch.testing.assert_close(endpoints.reconstruct_next(), ordinary, atol=3e-6, rtol=3e-6)
            self.assertTrue(torch.equal(snapshot, scheduler.sigmas))
            with self.assertRaises(ValueError): flow_bounds(scheduler, -99, state)

    def test_flux_bfloat16_scheduler_rounding_explains_precision_gap(self):
        scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
        scheduler.set_timesteps(20, mu=1.15)
        torch.manual_seed(9)
        state = torch.randn(1,4,8,8)
        velocity = torch.randn_like(state).to(torch.bfloat16)
        errors = []
        for t in scheduler.timesteps:
            sigma, following = flow_bounds(scheduler, t, state)
            endpoints = flow_endpoints(state, velocity, sigma, following)
            ideal = state + (following-sigma)*velocity.float()
            torch.testing.assert_close(endpoints.reconstruct_next(), ideal, atol=1e-6, rtol=1e-6)
            reset_scheduler_step_state(scheduler)
            ordinary = scheduler.step(velocity, t, state).prev_sample.float()
            # Installed scheduler multiplies in prediction dtype and rounds its
            # entire result to that dtype. This is separate from the flow algebra.
            native_precision = (state + (following-sigma)*velocity).to(velocity.dtype).float()
            self.assertTrue(torch.equal(ordinary, native_precision))
            errors.append(float((ordinary-endpoints.reconstruct_next()).abs().mean()))
        self.assertGreater(sum(errors)/len(errors), 1e-5)

    def test_pixel_endpoint_and_near_zero(self):
        clean, endpoint = torch.randn(2, 3, 5, 5), torch.randn(2, 3, 5, 5)
        for sigma in (.99975, .2, 1e-7):
            state = (1-sigma)*clean + sigma*endpoint
            result = pixel_endpoints(state, clean, sigma, sigma/2)
            torch.testing.assert_close((1-result.sigma)*clean+result.sigma*result.endpoint, state, atol=2e-7, rtol=2e-7)
            self.assertTrue(bool(torch.isfinite(result.reconstruct_next()).all()))
        result = pixel_endpoints(clean, clean, 0, 0)
        self.assertTrue(torch.equal(result.reconstruct_next(), clean))
        with self.assertRaises(ValueError): pixel_endpoints(clean, clean, 0, .1)

    def test_pixel_official_first_update(self):
        import sys
        from pathlib import Path
        from diffpano.pipelines.pixeldit_solver import PixelDiTFirstOrderSolver
        repository = Path('third_party/PixelDiT')
        if not repository.is_dir(): self.skipTest('Pinned PixelDiT checkout unavailable')
        for directory in (repository.resolve(), (repository/'t2i').resolve()):
            if str(directory) not in sys.path: sys.path.insert(0, str(directory))
        from diffusion.model.flow_dpm import DPMS
        reference = DPMS(lambda state, t, **kw: torch.full_like(state, .125),
                         condition=None, uncondition=None, cfg_scale=1.,
                         guidance_type='uncond', model_type='flow', schedule='FLOW')
        solver = PixelDiTFirstOrderSolver(4.)
        for t in solver.prepare(5, device=torch.device('cpu')):
            state = torch.randn(1,3,4,4)
            current, following = solver.bounds_for(t)
            clean = reference.model_fn(state, current)
            result = pixel_endpoints(state, clean, current, following)
            ordinary = reference.dpm_solver_first_update(state, current, following, model_s=clean)
            torch.testing.assert_close(result.reconstruct_next(), ordinary, atol=2e-6, rtol=2e-6)


class EndpointMock(NativeStateMixin):
    device = torch.device('cpu')
    native_initial_noise_sigma = 1.
    dtype = torch.float32
    def __init__(self):
        self.pipeline = SimpleNamespace(scheduler=FlowMatchEulerDiscreteScheduler())
        self.pipeline.scheduler.set_timesteps(5)
        self.timesteps = self.pipeline.scheduler.timesteps
        self.starts = []
        self.forward_inputs = []
        self.decode_count = 0
    def initialize_native_state(self, epsilon):
        result = super().initialize_native_state(epsilon); self.starts.append(result.clone()); return result
    def make_initial_noisy_state(self, epsilon, timestep):
        self.starts.append(epsilon.clone()); return epsilon.clone()
    def _predict(self, state, conditioning):
        self.record_guided_prediction()
        self.forward_inputs.append(state.clone())
        self.last_model_prediction = state*.2 + conditioning['offset']
        return self.last_model_prediction
    def denoise_native_step(self, state, timestep, conditioning):
        velocity = self._predict(state, conditioning)
        reset_scheduler_step_state(self.pipeline.scheduler)
        return self.pipeline.scheduler.step(velocity, timestep, state).prev_sample
    def predict_clean_native(self, state, timestep, conditioning):
        velocity = self._predict(state, conditioning)
        return flow_endpoints(state, velocity, *flow_bounds(self.pipeline.scheduler, timestep, state)).clean
    def _endpoints_from_last_prediction(self, state, timestep, clean=None):
        return flow_endpoints(state, self.last_model_prediction, *flow_bounds(self.pipeline.scheduler, timestep, state), clean=clean)
    def add_fixed_noise(self, clean, epsilon, timestep):
        sigma, _ = flow_bounds(self.pipeline.scheduler, timestep, clean)
        return (1-sigma)*clean + sigma*epsilon
    def decode_clean(self, value):
        self.decode_count += 1
        return value
    def encode_clean(self, value):
        raise AssertionError('Unexpected VAE roundtrip')


class ThreeWayTests(unittest.TestCase):
    def test_reuse_fairness_counts_and_old_ab_unchanged(self):
        backend = EndpointMock()
        epsilon = torch.randn(1,3,4,4)
        condition = {'offset': torch.ones(1,3,4,4)*.3}
        result = compare_three_way(backend, condition, epsilon)
        self.assertEqual(result.model_evaluations, dict(native=5, fixed_initial_noise=5, model_implied_endpoint=5))
        self.assertEqual(backend.guided_prediction_count, 15)
        self.assertEqual(backend.decode_count, 3)
        self.assertTrue(all(torch.equal(x, epsilon) for x in backend.starts))
        self.assertTrue(all(torch.equal(backend.forward_inputs[0], x) for x in backend.forward_inputs[:3]))
        torch.testing.assert_close(result.finals['native'], result.finals['model_implied_endpoint'])
        old = compare_single_patch(EndpointMock(), condition, epsilon)
        self.assertTrue(torch.equal(old.native_final, result.finals['native']))
        self.assertTrue(torch.equal(old.x0_renoise_final, result.finals['fixed_initial_noise']))
        self.assertLess(max(s['native_vs_implied_mae'] for s in result.steps), 1e-6)
        self.assertGreater(max(s['native_vs_fixed_mae'] for s in result.steps), 1e-3)

    def test_extra_forward_fails(self):
        backend = EndpointMock()
        original = backend.predict_clean_and_endpoint
        def duplicate(*args):
            original(*args)
            return original(*args)
        backend.predict_clean_and_endpoint = duplicate
        with self.assertRaisesRegex(AssertionError, '2 guided'):
            compare_three_way(backend, {'offset': torch.tensor(.3)}, torch.randn(1,3,4,4))

    def test_changed_initial_or_conditioning_fails(self):
        backend = EndpointMock()
        backend.make_initial_noisy_state = lambda epsilon, t: epsilon+1
        with self.assertRaisesRegex(ValueError, 'bit-identical'):
            compare_three_way(backend, {}, torch.randn(1,3,4,4))
        backend = EndpointMock()
        original = backend._predict
        def mutating(state, conditioning):
            conditioning['offset'].add_(.1)
            return original(state, conditioning)
        backend._predict = mutating
        with self.assertRaisesRegex(AssertionError, 'conditioning'):
            compare_three_way(backend, {'offset': torch.tensor(.3)}, torch.randn(1,3,4,4))


class AdapterEndpointTests(unittest.TestCase):
    def test_sd2_sana_flux_real_adapter_prediction_routes(self):
        from diffusers import FluxPipeline
        from diffpano.pipelines.sd2 import SD2ViewDenoiser
        from diffpano.pipelines.sana import SanaViewDenoiser
        from diffpano.pipelines.flux import FluxViewDenoiser
        class Network:
            dtype = torch.float32
            def __init__(self, channels):
                self.config = SimpleNamespace(in_channels=channels, out_channels=channels, guidance_embeds=True)
                self.calls = 0
            def __call__(self, *args, **kwargs):
                self.calls += 1
                value = args[0] if args else kwargs['hidden_states']
                return (value*.1,)
        for name in ('sd2', 'sana', 'flux'):
            network = Network(16 if name == 'flux' else 4)
            if name == 'sd2': scheduler = DDIMScheduler(clip_sample=False)
            elif name == 'sana': scheduler = DPMSolverMultistepScheduler(solver_order=1, prediction_type='flow_prediction', use_flow_sigmas=True)
            else: scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
            pipeline = SimpleNamespace(scheduler=scheduler, _execution_device='cpu', vae_scale_factor=8,
                unet=network, transformer=network, _pack_latents=FluxPipeline._pack_latents,
                _unpack_latents=FluxPipeline._unpack_latents, _prepare_latent_image_ids=FluxPipeline._prepare_latent_image_ids)
            cls = {'sd2': SD2ViewDenoiser, 'sana': SanaViewDenoiser, 'flux': FluxViewDenoiser}[name]
            backend = cls(pipeline, guidance_scale=1.)
            backend.prepare(num_steps=4, view_height=32, view_width=32)
            backend.decode_clean = lambda value: value
            condition = torch.zeros(1,2,3) if name == 'sd2' else dict(embeds=torch.zeros(1,2,3), mask=torch.ones(1,2), pooled=torch.zeros(1,3), text_ids=torch.zeros(2,3))
            result = compare_three_way(backend, condition, torch.randn(1,4,4,4))
            self.assertEqual(network.calls, 12)
            self.assertEqual(sum(result.model_evaluations.values()), 12)
            self.assertLess(max(step['native_vs_implied_mae'] for step in result.steps), 2e-6)
            torch.testing.assert_close(result.finals['native'], result.finals['model_implied_endpoint'], atol=2e-5, rtol=2e-5)

    def test_pixel_real_adapter_cache_and_counter(self):
        from test_pixeldit import make_adapter, conditioning
        backend = make_adapter()
        backend.prepare(num_steps=4, view_height=4, view_width=4)
        result = compare_three_way(backend, conditioning(), torch.randn(1,3,4,4))
        self.assertEqual(result.model_evaluations, dict(native=4, fixed_initial_noise=4, model_implied_endpoint=4))
        torch.testing.assert_close(result.finals['native'], result.finals['model_implied_endpoint'], atol=2e-5, rtol=2e-5)


if __name__ == '__main__': unittest.main()
