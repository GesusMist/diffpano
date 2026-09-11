import copy
import unittest
from pathlib import Path

import torch

from diffpano.config import FusionConfig, NativeMultiDiffusionConfig, load_experiment_config
from diffpano.implied_endpoint_consensus import PlanarImpliedEndpointConsensusPipeline
from diffpano.planar import extract_planar_patch
from scripts.detail_preserving_experiment import check_configs, read
from test_implied_endpoint_consensus import ConsensusMock


class ContextMock(ConsensusMock):
    def _predict(self, state, conditioning):
        self.record_guided_prediction()
        self.last_model_prediction = state*.2 + state.mean()*.4
        return self.last_model_prediction

    def encode_clean(self, rgb):
        self.encodes += 1
        return rgb + .03*rgb.square()


class DetailConsensusTests(unittest.TestCase):
    def setUp(self):
        self.geometry = NativeMultiDiffusionConfig(canvas_height=4, canvas_width=8, patch_size=4, stride=2)
        self.fusion = FusionConfig(mode='detail_preserving_average', weight_mode='uniform')
        self.initial = torch.randn(1, 3, 4, 8, generator=torch.Generator().manual_seed(13))
        self.conditioning = {'offset': torch.tensor(.3)}

    def pipe(self, backend, **kwargs):
        return PlanarImpliedEndpointConsensusPipeline(native_config=self.geometry, backend=backend,
            fusion_config=self.fusion, **kwargs)

    def oracle_fusion(self, pipe, proposals):
        numerator = torch.zeros_like(self.initial)
        denominator = torch.zeros_like(self.initial)
        for patch, rgb in zip(pipe.rgb_layout.patches, proposals):
            weight = rgb.abs() + 1e-6
            extract_planar_patch(numerator, patch).add_(rgb*weight)
            extract_planar_patch(denominator, patch).add_(weight)
        return numerator/denominator

    def test_signed_channel_magnitude_formula_without_clamping(self):
        pipe = self.pipe(ConsensusMock())
        proposals = [self.initial[..., p.x:p.x+4].clone()*(i+1) for i,p in enumerate(pipe.rgb_layout.patches)]
        actual, weights = pipe.fuse_rgb(proposals)
        torch.testing.assert_close(actual, self.oracle_fusion(pipe, proposals), atol=1e-6, rtol=1e-6)
        self.assertEqual(float(weights.min()), 1.)
        self.assertEqual(float(weights.max()), 2.)
        self.assertGreater(float(actual.abs().max()), 1.)
        average = PlanarImpliedEndpointConsensusPipeline(native_config=self.geometry, backend=ConsensusMock()).fuse_rgb(proposals)[0]
        self.assertGreater(float((actual-average).abs().max()), .1)

    def test_latent_and_pixel_jacobi_against_independent_full_trajectory(self):
        for pixel in (False, True):
            results = []
            for order in ([0,1,2], [2,1,0]):
                backend = ContextMock()
                pipe = self.pipe(backend, pixel_native=pixel, residual_correction=not pixel, patch_order=order)
                local = pipe.initialize_local_states(self.initial)
                result = pipe.run(local, self.conditioning)
                reference = ContextMock()
                states = [s.clone() for s in local]
                for t in reference.timesteps:
                    pairs = [reference.predict_clean_and_endpoint(s,t,self.conditioning) for s in states]
                    rgbs = [p.clean if pixel else reference.decode_clean(p.clean) for p in pairs]
                    fused = self.oracle_fusion(pipe, rgbs)
                    residual = torch.zeros_like(self.initial)
                    coverage = torch.zeros_like(self.initial)
                    if not pixel:
                        for patch, pair, rgb in zip(pipe.native_layout.patches,pairs,rgbs):
                            extract_planar_patch(residual,patch).add_(pair.clean-reference.encode_clean(rgb))
                            extract_planar_patch(coverage,patch).add_(1.)
                        residual /= coverage
                    states = [pair.next_alpha*(extract_planar_patch(fused,patch) if pixel else
                        reference.encode_clean(extract_planar_patch(fused,patch))+extract_planar_patch(residual,patch))
                        +pair.next_sigma*pair.endpoint for pair,patch in zip(pairs,pipe.rgb_layout.patches)]
                for actual, expected in zip(result.local_states, states):
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
                terminal = [s if pixel else reference.decode_native_canvas(s) for s in states]
                torch.testing.assert_close(result.canvas_rgb, self.oracle_fusion(pipe,terminal), atol=2e-6, rtol=2e-6)
                self.assertEqual(backend.guided_prediction_count, 15)
                self.assertEqual(backend.encodes, 0 if pixel else 30)
                self.assertFalse(result.audit['bridge_enabled'])
                self.assertEqual(result.audit['training_free_residual_correction'], not pixel)
                results.append(result)
            torch.testing.assert_close(results[0].canvas_rgb, results[1].canvas_rgb, atol=0, rtol=0)

    def test_all_five_configs_change_only_rgb_fusion(self):
        specs = read('configs/experiments/vae_residual/h-all-models.json')
        self.assertEqual(set(specs), {'sd2','sana','flux','sd35','pixeldit'})
        for spec in specs.values():
            config, _ = check_configs(spec)
            self.assertEqual(config.fusion, self.fusion)
            invalid = copy.deepcopy(config)
            invalid.global_pipeline.mode = 'native_multidiffusion'
            with self.assertRaises(ValueError):
                invalid.validate()


if __name__ == '__main__':
    unittest.main()
