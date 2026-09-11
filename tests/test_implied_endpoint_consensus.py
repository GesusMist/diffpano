import copy
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from diffpano.config import NativeMultiDiffusionConfig, load_experiment_config
from diffpano.implied_endpoint_consensus import PlanarImpliedEndpointConsensusPipeline
from diffpano.pipelines.endpoints import flow_endpoints
from diffpano.planar import extract_planar_patch
from test_endpoint_trajectory import EndpointMock


class ConsensusMock(EndpointMock):
    native_channels = 3
    native_spatial_factor = 1
    def __init__(self, projection=0.):
        super().__init__()
        self.projection = projection
        self.encodes = 0
    def conditioning_for_prompt_indices(self, prepared, indices, batch_size):
        return prepared
    def encode_clean(self, rgb):
        self.encodes += 1
        return rgb + self.projection
    def add_fixed_noise(self, *args):
        raise AssertionError('Fixed-noise renoising must never occur')


class ConsensusTests(unittest.TestCase):
    def setUp(self):
        self.geometry = NativeMultiDiffusionConfig(canvas_height=4, canvas_width=8, patch_size=4, stride=2)
        self.conditioning = {'offset': torch.tensor(.3)}
        self.initial = torch.randn(1,3,4,8, generator=torch.Generator().manual_seed(13))
    def pipe(self, backend, **kwargs):
        return PlanarImpliedEndpointConsensusPipeline(native_config=self.geometry, backend=backend, **kwargs)

    def test_initialization_matches_native_global_sampling_and_isolated_crops(self):
        backend = ConsensusMock()
        initial = backend.sample_initial_native_state(batch_size=1, native_height=4, native_width=8,
            generator=torch.Generator().manual_seed(13))
        self.assertTrue(torch.equal(initial, self.initial))
        pipe = self.pipe(backend)
        local = pipe.initialize_local_states(initial)
        for p, s in zip(pipe.native_layout.patches, local):
            self.assertTrue(torch.equal(s, extract_planar_patch(initial,p)))
        local[0].zero_()
        self.assertTrue(torch.equal(initial, self.initial))

    def test_identical_overlaps_fuse_and_reextract_exactly(self):
        pipe = self.pipe(ConsensusMock())
        proposals = [extract_planar_patch(self.initial,p) for p in pipe.rgb_layout.patches]
        fused, weights = pipe.fuse_rgb(proposals)
        self.assertTrue(torch.equal(fused,self.initial))
        self.assertTrue(bool((weights>0).all()))
        for p, proposal in zip(pipe.rgb_layout.patches,proposals):
            self.assertTrue(torch.equal(extract_planar_patch(fused,p),proposal))

    def test_jacobi_order_counts_no_fixed_noise_and_roundtrip(self):
        results = []
        for order in ([0,1,2],[2,1,0]):
            backend = ConsensusMock(projection=.02)
            pipe = self.pipe(backend, patch_order=order)
            local = pipe.initialize_local_states(self.initial)
            result = pipe.run(local,self.conditioning)
            # The first three calls see the frozen, bit-identical initial crops.
            for seen, i in zip(backend.forward_inputs[:3],order):
                self.assertTrue(torch.equal(seen,local[i]))
            self.assertEqual(backend.guided_prediction_count,15)
            self.assertEqual(backend.encodes,30)
            self.assertEqual(result.audit['expected_guided_predictions'],15)
            self.assertIsNone(backend.last_model_prediction)
            self.assertTrue(all(abs(s.state_statistics['clean_native_roundtrip_mae']-.02)<1e-6 for s in result.steps))
            self.assertFalse(result.audit['global_native_state_persisted'])
            results.append(result)
        torch.testing.assert_close(results[0].canvas_rgb,results[1].canvas_rgb,rtol=0,atol=0)
        for a,b in zip(results[0].local_states,results[1].local_states):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_clean_replacement_preserves_original_endpoint_not_velocity(self):
        x0,x1 = torch.full((1,3,4,4),.2),torch.full((1,3,4,4),.9)
        state=.4*x0+.6*x1
        pair=flow_endpoints(state,x1-x0,.6,.3)
        replacement=x0+.5
        actual=pair.reconstruct_next(clean=replacement)
        torch.testing.assert_close(actual,.7*replacement+.3*x1)
        self.assertFalse(torch.allclose(actual,.7*replacement+.3*(replacement+x1-x0)))
        torch.testing.assert_close(pair.clean,x0)
        torch.testing.assert_close(pair.endpoint,x1)
        with self.assertRaises(ValueError): pair.reconstruct_next(clean=torch.zeros(1,3,2,2))

    def test_one_patch_identity_projection_equals_independent_endpoint_trajectory(self):
        geometry=copy.deepcopy(self.geometry); geometry.canvas_width=4
        backend=ConsensusMock()
        pipe=PlanarImpliedEndpointConsensusPipeline(native_config=geometry,backend=backend,pixel_native=True)
        initial=self.initial[...,:4].clone()
        actual=pipe.run([initial],self.conditioning)
        reference=ConsensusMock(); state=initial.clone()
        for t in reference.timesteps:
            state=reference.predict_clean_and_endpoint(state,t,self.conditioning).reconstruct_next()
        torch.testing.assert_close(actual.canvas_rgb,state)
        self.assertEqual(backend.encodes,0)
        self.assertEqual(backend.decode_count,0)

    def test_actual_cross_patch_correction_matches_independent_jacobi_oracle(self):
        class ContextMock(ConsensusMock):
            def _predict(self, state, conditioning):
                self.record_guided_prediction()
                self.last_model_prediction = state*.2 + state.mean()*.4
                return self.last_model_prediction
        backend=ContextMock(projection=.03)
        pipe=self.pipe(backend)
        initial=pipe.initialize_local_states(self.initial)
        actual=pipe.run(initial,self.conditioning)
        reference=ContextMock(projection=.03)
        states=[s.clone() for s in initial]
        for t in reference.timesteps:
            pairs=[reference.predict_clean_and_endpoint(s,t,self.conditioning) for s in states]
            numerator=torch.zeros_like(self.initial);denominator=torch.zeros_like(self.initial[:,:1])
            for p,pair in zip(pipe.rgb_layout.patches,pairs):
                extract_planar_patch(numerator,p).add_(pair.clean)
                extract_planar_patch(denominator,p).add_(1)
            fused=numerator/denominator
            states=[pair.next_alpha*(extract_planar_patch(fused,p)+.03)+pair.next_sigma*pair.endpoint
                    for pair,p in zip(pairs,pipe.rgb_layout.patches)]
        for a,b in zip(actual.local_states,states):torch.testing.assert_close(a,b)
        self.assertGreater(actual.steps[0].state_statistics['consensus_correction_mae_mean'],0.)

    def test_noninteger_factor_and_invalid_initialization_fail(self):
        backend=ConsensusMock(); backend.native_spatial_factor=1.5
        with self.assertRaises(ValueError): self.pipe(backend)
        pipe=self.pipe(ConsensusMock())
        with self.assertRaises(ValueError): pipe.initialize_local_states(torch.zeros(1,4,4,8))

    def test_duplicate_model_prediction_fails(self):
        backend=ConsensusMock(); original=backend.predict_clean_and_endpoint
        def duplicate(*args):
            original(*args)
            return original(*args)
        backend.predict_clean_and_endpoint=duplicate
        pipe=self.pipe(backend)
        with self.assertRaisesRegex(AssertionError,'one guided'):
            pipe.run(pipe.initialize_local_states(self.initial),self.conditioning)

    def test_configs_exactly_match_native_controls_and_route(self):
        from scripts.generate import _generate_with_selected_global_pipeline
        for name in ('sd2','sana','flux','pixeldit'):
            native=load_experiment_config('configs/experiments/native_multidiffusion/'+name+'.yaml')
            new=load_experiment_config('configs/experiments/implied_endpoint_consensus/'+name+'.yaml')
            a,b=native.to_dict(),new.to_dict()
            for key in ('experiment','output','global_pipeline'):
                del a[key]; del b[key]
            self.assertEqual(a,b)
            self.assertEqual(native.experiment.seed,new.experiment.seed)
            with patch('scripts.generate.generate_planar_implied_endpoint_consensus',return_value='new'):
                self.assertEqual(_generate_with_selected_global_pipeline(new,None,None),'new')
            invalid=copy.deepcopy(new); invalid.canvas.mode='erp'
            with self.assertRaises(ValueError): invalid.validate()
            for mode,weight in (('lpw','uniform'),('average','cosine')):
                invalid=copy.deepcopy(new); invalid.fusion.mode=mode; invalid.fusion.weight_mode=weight
                with self.assertRaises(ValueError): invalid.validate()
            invalid=copy.deepcopy(new); invalid.performance.view_batch_size=2
            with self.assertRaisesRegex(ValueError,'individually'): invalid.validate()
            invalid=copy.deepcopy(new); invalid.planar.patch_strategy='dynamic'
            with self.assertRaises(ValueError): invalid.validate()

if __name__=='__main__': unittest.main()
