from historical_configs import retained_reference
import unittest
from unittest.mock import patch
import torch
from diffpano.config import NativeMultiDiffusionConfig
from diffpano.implied_endpoint_consensus import PlanarImpliedEndpointConsensusPipeline
from scripts.no_residual_planar_experiment import check_configs,read
from test_detail_preserving_consensus import ContextMock


class NoResidualPlanarTests(unittest.TestCase):
    def test_j_has_only_fused_encode_no_residual_and_is_jacobi(self):
        initial=torch.randn(1,3,4,8,generator=torch.Generator().manual_seed(82))
        results=[]
        for order in ([0,1,2],[2,1,0]):
            b=ContextMock()
            p=PlanarImpliedEndpointConsensusPipeline(native_config=NativeMultiDiffusionConfig(
                canvas_height=4,canvas_width=8,patch_size=4,stride=2),backend=b,
                transition_mode='preserve_current_state',residual_correction=False,
                roundtrip_diagnostics=False,patch_order=order)
            local=p.initialize_local_states(initial)
            with patch('diffpano.implied_endpoint_consensus.vae_residual',side_effect=AssertionError('No residual')), \
                 patch('diffpano.implied_endpoint_consensus.fuse_native_residuals',side_effect=AssertionError('No residual fusion')):
                result=p.run(local,{'offset':torch.tensor(.3)})
            self.assertEqual(b.guided_prediction_count,15)
            self.assertEqual(b.encodes,15)
            for seen,i in zip(b.forward_inputs[:3],order):self.assertTrue(torch.equal(seen,local[i]))
            self.assertFalse(result.audit['training_free_residual_correction'])
            self.assertNotIn('clean_native_roundtrip_mae',result.steps[0].state_statistics)
            self.assertLess(max(r['current_state_interpolation_reconstruction_error'] for r in result.transition_diagnostics),1e-6)
            self.assertIn('rgb_consensus_delta_mae_mean',result.steps[0].state_statistics)
            results.append(result.canvas_rgb)
        self.assertTrue(torch.equal(*results))

    def test_all_j_configs_pair_exactly_with_saved_f(self):
        specs=read('configs/experiments/implied_endpoint_consensus/j-all-models.json')
        self.assertEqual(list(specs),['sd35','flux','sana','sd2'])
        for spec in specs.values():
            c,_=check_configs(retained_reference(spec))
            self.assertFalse(c.consensus_transition.vae_residual_correction)
            self.assertEqual(c.fusion.mode,'average')
        from scripts.current_state_experiment import check_configs as check_i
        for spec in read('configs/experiments/vae_residual/i-all-models.json').values():check_i(retained_reference(spec))
