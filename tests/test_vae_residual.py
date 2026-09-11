import unittest
from unittest.mock import Mock,patch
import torch

from diffpano.vae_residual import recover_identity,residual_metrics,single_patch_residual_trajectory,vae_residual
from test_implied_endpoint_consensus import ConsensusMock


class ResidualSingleTests(unittest.TestCase):
    def test_identity_recovery_random_channels_and_dtype(self):
        for channels in (4,7,16,32):
            z=torch.randn(2,channels,8,8)
            rt=z*.4+.3
            recovered=recover_identity(z,rt)
            torch.testing.assert_close(recovered,z,atol=3e-7,rtol=3e-7)
            self.assertLess(residual_metrics(z,rt,recovered)['correction_recovery_mae'],1e-7)
        with self.assertRaises(ValueError):vae_residual(z,rt.double())

    def test_C_decode_encode_once_no_fusion_and_D_reduces_to_B(self):
        initial=torch.randn(1,3,4,4);condition={'offset':torch.tensor(.3)}
        finals=[]
        for correction in (False,True):
            backend=ConsensusMock(projection=.17)
            endpoints=[]
            original=backend.predict_clean_and_endpoint
            def wrapped(*args):
                pair=original(*args);endpoints.append((pair,pair.endpoint.clone()));return pair
            backend.predict_clean_and_endpoint=wrapped
            with patch('diffpano.native_multidiffusion.NativePlanarFusionAccumulator',side_effect=AssertionError('Fusion in single patch')):
                image,steps,count=single_patch_residual_trajectory(backend,initial,condition,correction=correction)
            self.assertEqual(count,5);self.assertEqual(backend.encodes,5);self.assertEqual(backend.decode_count,6)
            self.assertEqual(backend.guided_prediction_count,5)
            self.assertTrue(all(torch.equal(pair.endpoint,snapshot) for pair,snapshot in endpoints))
            finals.append(image)
        reference=ConsensusMock();state=initial.clone()
        for t in reference.timesteps:state=reference.predict_clean_and_endpoint(state,t,condition).reconstruct_next()
        torch.testing.assert_close(finals[1],state,atol=1e-6,rtol=1e-6)
        self.assertFalse(torch.allclose(finals[0],state))

    def test_duplicate_denoiser_prediction_rejected(self):
        b=ConsensusMock();original=b.predict_clean_and_endpoint
        def twice(*args):original(*args);return original(*args)
        b.predict_clean_and_endpoint=twice
        with self.assertRaises(AssertionError):single_patch_residual_trajectory(b,torch.zeros(1,3,4,4),{'offset':torch.tensor(.3)})

class ResidualConsensusTests(unittest.TestCase):
    def setUp(self):
        from diffpano.config import NativeMultiDiffusionConfig
        self.geometry=NativeMultiDiffusionConfig(canvas_height=4,canvas_width=8,patch_size=4,stride=2)
        self.initial=torch.randn(1,3,4,8,generator=torch.Generator().manual_seed(83))
        self.condition={'offset':torch.tensor(.3)}

    def pipeline(self,backend,**kwargs):
        from diffpano.implied_endpoint_consensus import PlanarImpliedEndpointConsensusPipeline
        return PlanarImpliedEndpointConsensusPipeline(native_config=self.geometry,backend=backend,**kwargs)

    def test_zero_residual_G_equals_F_and_single_patch_G_equals_D(self):
        outputs=[]
        for enabled in (False,True):
            backend=ConsensusMock();pipe=self.pipeline(backend,residual_correction=enabled)
            result=pipe.run(pipe.initialize_local_states(self.initial),self.condition)
            outputs.append(result.canvas_rgb)
            self.assertEqual(backend.guided_prediction_count,15)
            self.assertEqual(backend.encodes,30)
        torch.testing.assert_close(*outputs,atol=0,rtol=0)
        self.geometry.canvas_width=4
        initial=self.initial[...,:4].clone()
        backend=ConsensusMock(projection=.1);pipe=self.pipeline(backend,residual_correction=True)
        result=pipe.run([initial],self.condition)
        expected,_,_=single_patch_residual_trajectory(ConsensusMock(projection=.1),initial,self.condition,correction=True)
        torch.testing.assert_close(result.canvas_rgb,expected,atol=1e-6,rtol=1e-6)

    def test_seven_channel_exact_native_placement_identical_overlap(self):
        from diffpano.planar import build_planar_patch_layout,extract_planar_patch
        from diffpano.vae_residual import fuse_native_residuals
        layout=build_planar_patch_layout(6,10,4,2)
        canvas=torch.randn(2,7,6,10,generator=torch.Generator().manual_seed(7))
        residuals=[extract_planar_patch(canvas,p).clone() for p in layout.patches]
        fused=fuse_native_residuals(layout,residuals)
        torch.testing.assert_close(fused,canvas,atol=0,rtol=0)
        for patch,residual in zip(layout.patches,residuals):
            torch.testing.assert_close(extract_planar_patch(fused,patch),residual,atol=0,rtol=0)

    def test_synchronized_residual_independent_oracle_order_endpoint_and_calls(self):
        from diffpano.planar import extract_planar_patch
        class ContextVAE(ConsensusMock):
            def _predict(self,state,conditioning):
                self.record_guided_prediction()
                self.last_model_prediction=state*.2+state.mean()*.4
                return self.last_model_prediction
            def encode_clean(self,rgb):
                self.encodes+=1
                return rgb+.1+rgb.mean().square()
        results=[]
        for order in ([0,1,2],[2,1,0]):
            backend=ContextVAE();pairs_saved=[];original=backend.predict_clean_and_endpoint
            def record(*args):
                pair=original(*args);pairs_saved.append((pair,pair.endpoint.clone()));return pair
            backend.predict_clean_and_endpoint=record
            pipe=self.pipeline(backend,patch_order=order,residual_correction=True)
            local=pipe.initialize_local_states(self.initial)
            result=pipe.run(local,self.condition);results.append(result)
            for seen,i in zip(backend.forward_inputs[:3],order):
                self.assertTrue(torch.equal(seen,local[i]))
            self.assertTrue(all(torch.equal(pair.endpoint,saved) for pair,saved in pairs_saved))
            self.assertEqual(backend.guided_prediction_count,15)
            self.assertEqual(backend.encodes,30);self.assertEqual(backend.decode_count,18)
            self.assertFalse(result.audit['global_native_state_persisted'])
        torch.testing.assert_close(results[0].canvas_rgb,results[1].canvas_rgb,atol=0,rtol=0)
        reference=ContextVAE();states=[s.clone() for s in local];local_wrong=[s.clone() for s in local]
        def average(proposals):
            numerator=torch.zeros_like(self.initial);denominator=torch.zeros_like(self.initial[:,:1])
            for patch,proposal in zip(pipe.native_layout.patches,proposals):
                extract_planar_patch(numerator,patch).add_(proposal)
                extract_planar_patch(denominator,patch).add_(1)
            return numerator/denominator
        for t in reference.timesteps:
            updated=[]
            for source,sync in ((states,True),(local_wrong,False)):
                pairs=[reference.predict_clean_and_endpoint(state,t,self.condition) for state in source]
                rgb=[pair.clean for pair in pairs]
                residuals=[pair.clean-(image+.1+image.mean().square()) for pair,image in zip(pairs,rgb)]
                fused=average(rgb);fused_r=average(residuals)
                next_states=[]
                for i,(pair,patch) in enumerate(zip(pairs,pipe.native_layout.patches)):
                    crop=extract_planar_patch(fused,patch)
                    correction=extract_planar_patch(fused_r,patch) if sync else residuals[i]
                    clean=crop+.1+crop.mean().square()+correction
                    next_states.append(pair.next_alpha*clean+pair.next_sigma*pair.endpoint)
                updated.append(next_states)
            states,local_wrong=updated
        for actual,expected in zip(results[0].local_states,states):
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        self.assertGreater(max(float((a-b).abs().max()) for a,b in zip(states,local_wrong)),1e-5)
        stats=results[0].steps[0].state_statistics
        self.assertGreater(stats['local_vae_residual_mae_max'],stats['local_vae_residual_mae_mean'])
        self.assertGreater(stats['residual_correction_magnitude'],0)

    def test_pixel_and_learned_bridge_are_excluded(self):
        with self.assertRaises(ValueError):self.pipeline(ConsensusMock(),pixel_native=True,residual_correction=True)
        with self.assertRaises(ValueError):self.pipeline(ConsensusMock(),bridge=Mock(),residual_correction=True)


if __name__=='__main__':unittest.main()

