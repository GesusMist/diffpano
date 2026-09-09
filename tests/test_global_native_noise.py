import unittest
from pathlib import Path

import torch

from diffpano.config import CleanConsensusConfig, ExperimentConfig, FusionConfig, PlanarConfig, load_experiment_config
from diffpano.noise import GlobalNativeNoiseBank
from diffpano.pipelines.native_state import NativeStateMixin
from diffpano.planar import build_planar_patch_layout, extract_planar_patch
from diffpano.planar_pipeline import PlanarX0ConsensusPipeline
from test_planar import MockCleanBackend


class NoiseGeometry(NativeStateMixin):
    device = torch.device('cpu')
    native_channels = 7

    def __init__(self, factor):
        self.native_spatial_factor = factor


class PixelNoiseBackend(NativeStateMixin, MockCleanBackend):
    native_spatial_factor = 1
    native_channels = 3


class GlobalNoiseTests(unittest.TestCase):
    def test_all_actual_factors_have_bit_identical_overlap_and_storage_modes(self):
        for factor in (1, 8, 32):
            layout = build_planar_patch_layout(16*factor, 24*factor, 16*factor, 8*factor)
            outputs = []
            for storage in ('cpu', 'gpu', 'seed'):
                bank = GlobalNativeNoiseBank(CleanConsensusConfig(noise_binding='global_native_canvas', noise_storage=storage),
                    backend=NoiseGeometry(factor), layout=layout, stride=8*factor, batch_size=2, seed=17)
                a = bank.get([0], device=torch.device('cpu'))
                b = bank.get([1], device=torch.device('cpu'))
                self.assertTrue(torch.equal(a[..., :, 8:], b[..., :, :8]))
                self.assertTrue(torch.equal(a, bank.get([0], device=torch.device('cpu'))))
                outputs.append(bank.get([0, 1], device=torch.device('cpu')))
            self.assertTrue(all(torch.equal(outputs[0], value) for value in outputs[1:]))

    def test_fractional_stride_canvas_patch_and_edge_origin_are_rejected(self):
        backend = NoiseGeometry(8)
        config = CleanConsensusConfig(noise_binding='global_native_canvas')
        cases = [(256, 512, 160, 50), (255, 512, 160, 48), (256, 512, 159, 48)]
        for h, w, p, stride in cases:
            with self.subTest(case=(h,w,p,stride)), self.assertRaises(ValueError):
                GlobalNativeNoiseBank(config, backend=backend,
                    layout=build_planar_patch_layout(h,w,p,stride), stride=stride, batch_size=1, seed=0)
        # Validate even externally supplied layouts, including a fractional edge origin.
        from diffpano.planar import PlanarPatch, PlanarPatchLayout
        malformed = PlanarPatchLayout(256, 512, 160, (PlanarPatch(0,0,1,160),))
        with self.assertRaisesRegex(ValueError, 'origin'):
            GlobalNativeNoiseBank(config, backend=backend, layout=malformed, stride=48, batch_size=1, seed=0)

    def test_aligned_stride_48_exact_edge_crop(self):
        layout = build_planar_patch_layout(256, 512, 160, 48)
        bank = GlobalNativeNoiseBank(CleanConsensusConfig(noise_binding='global_native_canvas'),
            backend=NoiseGeometry(8), layout=layout, stride=48, batch_size=1, seed=0)
        last = bank.patches[layout.patches[-1].index]
        self.assertEqual((last.y,last.x), (12,44))
        self.assertTrue(torch.equal(bank.get([last.index], device=torch.device('cpu')),
                                    bank._value[..., 12:32,44:64]))

    def test_planar_global_binding_pre_fusion_disagreement_zero_independent_nonzero(self):
        records = {}
        for binding in ('camera_index', 'global_native_canvas'):
            backend = PixelNoiseBackend(num_steps=2)
            result = PlanarX0ConsensusPipeline(
                planar_config=PlanarConfig(height=6,width=10,patch_size=6,stride=4,prompt_assignment='global'),
                fusion_config=FusionConfig(mode='average',weight_mode='uniform'),
                consensus_config=CleanConsensusConfig(noise_binding=binding), backend=backend,
                overlap_disagreement=True, seed=19,
            ).run(None, batch_size=1)
            records[binding] = result.steps[0].state_statistics['overlap_mae_mean']
            self.assertEqual(backend.prediction_calls, 4)
        self.assertEqual(records['global_native_canvas'], 0)
        self.assertGreater(records['camera_index'], 0)

    def test_global_noise_pipeline_order_and_chunk_invariance(self):
        outputs = []
        for order, chunk in ((None, 1), ([2, 1, 0], 2)):
            backend = PixelNoiseBackend(num_steps=2)
            result = PlanarX0ConsensusPipeline(
                planar_config=PlanarConfig(height=6,width=14,patch_size=6,stride=4,prompt_assignment='global'),
                fusion_config=FusionConfig(mode='average'),
                consensus_config=CleanConsensusConfig(noise_binding='global_native_canvas'),
                backend=backend, patch_batch_size=chunk, patch_order=order, seed=5,
            ).run(None, batch_size=2)
            outputs.append(result.canvas_rgb)
        self.assertTrue(torch.equal(outputs[0], outputs[1]))

    def test_clean_native_noise_shape_mismatch_rejected_without_resize(self):
        backend = PixelNoiseBackend(num_steps=2)
        backend.encode_clean = lambda rgb: rgb[..., :-1, :]
        pipe = PlanarX0ConsensusPipeline(
            planar_config=PlanarConfig(height=6,width=10,patch_size=6,stride=4),
            fusion_config=FusionConfig(mode='average'),
            consensus_config=CleanConsensusConfig(noise_binding='global_native_canvas'), backend=backend,
        )
        with self.assertRaisesRegex(ValueError, 'exactly match'):
            pipe.run(None, batch_size=1)

    def test_global_binding_rejects_erp(self):
        config = ExperimentConfig()
        config.global_pipeline.mode = 'erp_x0_consensus'
        config.global_pipeline.clean_consensus.noise_binding = 'global_native_canvas'
        with self.assertRaisesRegex(ValueError, 'planar x0'):
            config.validate()

    def test_phase_b_configs_keep_controls_fixed(self):
        base = Path(__file__).resolve().parents[1] / 'configs/experiments'
        for backend in ('sana','sd2','flux','pixeldit'):
            configs = [load_experiment_config(str(base/'planar_fusion'/f'{backend}-{label}.yaml')) for label in 'abcd']
            expected = configs[0].to_dict()
            for config in configs:
                actual = config.to_dict()
                actual.pop('fusion')
                actual['experiment'].pop('name')
                compare = {key:value for key,value in expected.items() if key != 'fusion'}
                compare['experiment'] = {key:value for key,value in expected['experiment'].items() if key != 'name'}
                self.assertEqual(actual, compare)
            for binding in ('camera_index','global_native_canvas'):
                config = load_experiment_config(str(base/'planar_x0_noise'/f'{backend}-{binding}.yaml'))
                self.assertEqual(config.planar.geometry_preset, 'custom')


if __name__ == '__main__':
    unittest.main()
