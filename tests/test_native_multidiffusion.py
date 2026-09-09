import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from diffpano.config import ExperimentConfig, NativeMultiDiffusionConfig
from diffpano.native_multidiffusion import NativeMultiDiffusionPipeline, NativePlanarFusionAccumulator
from diffpano.overlap import OverlapDisagreement
from diffpano.pipelines.native_state import NativeStateMixin
from diffpano.pipelines.sd2 import SD2ViewDenoiser
from diffpano.pipelines.sana import SanaViewDenoiser
from diffpano.pipelines.flux import FluxViewDenoiser
from diffpano.pipelines.clean_prediction import validate_sana_flow_scheduler, flow_add_noise, flow_sigma
from diffpano.planar import build_planar_patch_layout, extract_planar_patch
from diffpano.trajectory import compare_single_patch
from scripts.generate import _generate_with_selected_global_pipeline


class NativeMock(NativeStateMixin):
    native_channels = 7
    native_spatial_factor = 1
    native_initial_noise_sigma = 1.5
    device = torch.device('cpu')
    timesteps = torch.tensor([3., 2., 1.])

    def __init__(self):
        self.decode_count = 0
        self.native_calls = []
        self.clean_calls = []
        self.initial_inputs = []
        self.conditionings = []

    def initialize_native_state(self, epsilon):
        self.initial_inputs.append(epsilon.clone())
        return super().initialize_native_state(epsilon)

    def conditioning_for_prompt_indices(self, prepared, indices, *, batch_size):
        assert indices == [8]
        return prepared

    def denoise_native_step(self, state, timestep, conditioning):
        self.native_calls.append(state.clone())
        self.conditionings.append(conditioning)
        return state * .5

    def make_initial_noisy_state(self, epsilon, timestep):
        self.initial_inputs.append(epsilon.clone())
        return epsilon * self.native_initial_noise_sigma

    def add_fixed_noise(self, clean, epsilon, timestep):
        return clean * .25 + epsilon * .75

    def predict_clean_native(self, state, timestep, conditioning):
        self.clean_calls.append(state.clone())
        self.conditionings.append(conditioning)
        return state * .3

    def encode_clean(self, value):
        raise AssertionError('native loop must not encode')

    def decode_clean(self, value):
        self.decode_count += 1
        return value[:, :3]


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.layout = build_planar_patch_layout(11, 17, 6, 4)
        self.source = torch.randn(2, 7, 11, 17)

    def test_arbitrary_channel_identity_and_order(self):
        results = []
        for patches in (self.layout.patches, tuple(reversed(self.layout.patches))):
            acc = NativePlanarFusionAccumulator(self.source)
            for p in patches:
                acc.accumulate(extract_planar_patch(self.source, p), p)
            results.append(acc.finalize())
        torch.testing.assert_close(results[0], self.source, atol=5e-7, rtol=0)
        torch.testing.assert_close(results[0], results[1], atol=5e-7, rtol=0)

    def test_native_loop_no_vae_no_x0_helpers_final_decode_only(self):
        cfg = NativeMultiDiffusionConfig(11, 17, 6, 4)
        results = []
        for order in (None, list(reversed(range(self.layout.num_patches)))):
            backend = NativeMock()
            backend.add_fixed_noise = Mock(side_effect=AssertionError('renoise called'))
            backend.predict_clean_native = Mock(side_effect=AssertionError('x0 called'))
            pipe = NativeMultiDiffusionPipeline(native_config=cfg, backend=backend,
                    patch_order=order, overlap_disagreement=True)
            result = pipe.run(self.source, object())
            torch.testing.assert_close(result.native_canvas, self.source / 8)
            self.assertEqual(backend.decode_count, 1)
            self.assertEqual(len(backend.native_calls), 3 * self.layout.num_patches)
            self.assertEqual(result.steps[0].state_statistics['overlap_mae_mean'], 0)
            results.append(result.native_canvas)
        torch.testing.assert_close(*results)

    def test_same_epsilon_conditioning_and_evaluation_counts(self):
        backend = NativeMock()
        epsilon = torch.randn(2, 7, 6, 6)
        conditioning = object()
        result = compare_single_patch(backend, conditioning, epsilon)
        self.assertTrue(torch.equal(backend.initial_inputs[0], backend.initial_inputs[1]))
        self.assertTrue(torch.equal(backend.native_calls[0], backend.clean_calls[0]))
        self.assertTrue(torch.equal(result.initial_epsilon, epsilon))
        self.assertEqual(result.model_evaluations, {'native': 3, 'x0_renoise': 3})
        self.assertEqual(len(backend.native_calls), 3)
        self.assertEqual(len(backend.clean_calls), 3)
        self.assertTrue(all(c is conditioning for c in backend.conditionings))
        self.assertEqual(backend.decode_count, 2)

    def test_neighbor_diagnostics_known_difference_order_and_nonoverlap(self):
        layout = build_planar_patch_layout(4, 7, 4, 3)
        for patches in (layout.patches, tuple(reversed(layout.patches))):
            metric = OverlapDisagreement(layout)
            for p in patches:
                metric.add(p, torch.full((1, 7, 4, 4), float(p.index * 2)))
            self.assertEqual(metric.values()['overlap_pair_count'], 1)
            self.assertEqual(metric.values()['overlap_mae_mean'], 2)
            self.assertEqual(metric.values()['overlap_rmse_mean'], 2)
            self.assertFalse(metric.pending)
        layout = build_planar_patch_layout(4, 8, 4, 4)
        self.assertEqual(OverlapDisagreement(layout).values()['overlap_pair_count'], 0)

    def test_native_config_route_and_erp_rejection(self):
        config = ExperimentConfig()
        config.canvas.mode = 'planar'
        config.warp.mode = 'standard'
        config.fusion.mode = 'average'
        config.fusion.weight_mode = 'uniform'
        config.global_pipeline.mode = 'native_multidiffusion'
        config.validate()
        with patch('scripts.generate.generate_planar_native_multidiffusion', return_value='native') as call:
            self.assertEqual(_generate_with_selected_global_pipeline(config, None, None), 'native')
            call.assert_called_once()
        config.canvas.mode = 'erp'
        with self.assertRaisesRegex(ValueError, 'requires canvas'):
            config.validate()

    def test_all_phase_a_configs_load_and_use_supported_local_sizes(self):
        from pathlib import Path
        from diffpano.config import load_experiment_config
        for group in ('native_multidiffusion', 'trajectory'):
            for path in (Path(__file__).resolve().parents[1] / 'configs' / 'experiments' / group).glob('*.yaml'):
                config = load_experiment_config(str(path))
                self.assertEqual(config.global_pipeline.mode, 'native_multidiffusion')
                self.assertEqual(config.native_multidiffusion.prompt_assignment, 'global')
                if config.model.pipeline == 'pixeldit':
                    self.assertEqual(config.native_multidiffusion.patch_size, 1024)

    def test_fractional_rgb_native_dimensions_rejected(self):
        backend = NativeMock()
        backend.native_spatial_factor = 2
        with self.assertRaises(ValueError):
            backend.native_spatial_shape_for_rgb(5, 8)


class StatefulScheduler:
    order = 1
    config = SimpleNamespace(solver_order=1)
    init_noise_sigma = 1

    def __init__(self):
        self._step_index = 9
        self.model_outputs = [torch.tensor(99.)]
        self.lower_order_nums = 1
        self.indices = []

    def step(self, prediction, timestep, state, return_dict=False):
        if self._step_index is None:
            self._step_index = int(timestep)
        self.indices.append(self._step_index)
        assert self.model_outputs == [None]
        assert self.lower_order_nums == 0
        self.model_outputs = [prediction]
        self.lower_order_nums += 1
        self._step_index += 1
        return (state - prediction,)


class AdapterTests(unittest.TestCase):
    def test_each_same_timestep_patch_resets_real_adapter_scheduler_calls(self):
        for cls, predictor in ((SD2ViewDenoiser, '_predict_noise'), (SanaViewDenoiser, '_predict_flow')):
            scheduler = StatefulScheduler()
            backend = cls(SimpleNamespace(scheduler=scheduler), guidance_scale=1)
            setattr(backend, predictor, lambda state, t, c: torch.ones_like(state))
            backend.decode_clean = Mock(side_effect=AssertionError('VAE called'))
            for timestep in (3, 2, 1):
                for _ in range(4):
                    backend.denoise_native_step(torch.zeros(1, 7, 4, 4), timestep, None)
            self.assertEqual(scheduler.indices, [3]*4 + [2]*4 + [1]*4)

    def test_flux_actual_raw_geometry_and_scheduler_reset(self):
        scheduler = StatefulScheduler()
        seen = []
        pipeline = SimpleNamespace(scheduler=scheduler, vae_scale_factor=8, _execution_device='cpu')
        backend = FluxViewDenoiser(pipeline, guidance_scale=1)
        backend._view_size = (32, 64)
        backend._pack = lambda value: value.permute(0, 2, 3, 1).reshape(1, 32, 4)
        backend._unpack = lambda value, h, w: value.reshape(1, 4, 8, 4).permute(0, 3, 1, 2)
        def predict(value, timestep, conditioning, *, native_shape):
            seen.append(tuple(native_shape))
            return torch.zeros_like(value)
        backend._guided_prediction = predict
        for _ in range(3):
            backend.denoise_native_step(torch.ones(1, 4, 4, 8), 2, None)
        self.assertEqual(seen, [(4, 8)] * 3)
        self.assertEqual(scheduler.indices, [2] * 3)
        with self.assertRaisesRegex(ValueError, 'resolution'):
            backend.denoise_native_step(torch.ones(1, 4, 8, 4), 2, None)
        with self.assertRaisesRegex(ValueError, 'even'):
            backend._validate_native_geometry((3, 8))

    def test_sana_rejects_incompatible_clean_semantics(self):
        for config in (SimpleNamespace(prediction_type='epsilon', use_flow_sigmas=True),
                       SimpleNamespace(prediction_type='flow_prediction', use_flow_sigmas=False)):
            with self.assertRaises(ValueError):
                validate_sana_flow_scheduler(SimpleNamespace(config=config))

    def test_real_diffusers_flow_schedule_and_forward_noise(self):
        from diffusers import FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler
        schedulers = [FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
                      DPMSolverMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, solver_order=1)]
        for scheduler in schedulers:
            if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
                scheduler.set_timesteps(4, mu=.7)
            else:
                scheduler.set_timesteps(4)
                validate_sana_flow_scheduler(scheduler)
            clean, noise = torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4)
            for timestep in scheduler.timesteps:
                scheduler._step_index = 3  # a previous independent patch/trajectory
                value = flow_add_noise(scheduler, clean, noise, timestep)
                sigma = flow_sigma(scheduler, timestep, device=clean.device, dtype=clean.dtype)
                torch.testing.assert_close(value, (1-sigma)*clean + sigma*noise)


if __name__ == '__main__':
    unittest.main()
