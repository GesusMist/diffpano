from historical_configs import retained_reference
import copy
import unittest
from dataclasses import replace

import torch
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler

from diffpano.current_state_transition import interpolate_from_current_state, transition_diagnostics
from diffpano.pipelines.endpoints import flow_endpoints, ddim_endpoints, flow_bounds
from diffpano.config import NativeMultiDiffusionConfig
from diffpano.implied_endpoint_consensus import PlanarImpliedEndpointConsensusPipeline
from scripts.current_state_experiment import check_configs, read
from test_implied_endpoint_consensus import ConsensusMock
from test_detail_preserving_consensus import ContextMock


def transition(x,c,p,flow=True):
    return interpolate_from_current_state(x,c,p.alpha,p.sigma,p.next_alpha,p.next_sigma,flow=flow)


class CurrentStateTests(unittest.TestCase):
    def setUp(self):
        generator=torch.Generator().manual_seed(51)
        self.x,self.c,self.v=[torch.randn(1,3,4,8,generator=generator) for _ in range(3)]

    def test_flow_current_consistency_interpolation_and_signed_identity(self):
        for sigma,sn in ((1.,.9),(.87,.3),(.2,.01),(.01,0.)):
            p=flow_endpoints(self.x,self.v,sigma,sn)
            e=(self.x-(1-sigma)*self.c)/sigma
            torch.testing.assert_close((1-sigma)*self.c+sigma*e,self.x,atol=2e-6,rtol=2e-6)
            actual=transition(self.x,self.c,p)
            torch.testing.assert_close(actual,(1-sn)*self.c+sn*e,atol=2e-6,rtol=2e-6)
            expected=-(sn/sigma)*(1-sigma)*(self.c-p.clean)
            torch.testing.assert_close(actual-p.reconstruct_next(clean=self.c),expected,atol=2e-6,rtol=2e-6)
            d=transition_diagnostics(self.x,self.c,p,actual)
            self.assertLess(d['i_current_state_mismatch'],2e-6)
            if sn==0: self.assertTrue(torch.equal(actual,self.c))

    def test_actual_flow_schedules_no_clean_change(self):
        for scheduler,options in ((FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),{'mu':1.15}),
                                  (FlowMatchEulerDiscreteScheduler(shift=3),{}),
                                  (DPMSolverMultistepScheduler(solver_order=1,prediction_type='flow_prediction',use_flow_sigmas=True,flow_shift=3),{})):
            scheduler.set_timesteps(20,**options);snapshot=scheduler.sigmas.clone()
            for t in scheduler.timesteps:
                s,sn=flow_bounds(scheduler,t,self.x);p=flow_endpoints(self.x,self.v,s,sn)
                torch.testing.assert_close(transition(self.x,p.clean,p),p.reconstruct_next(),atol=2e-6,rtol=2e-6)
            self.assertTrue(torch.equal(snapshot,scheduler.sigmas))

    def test_ddim_consistency_no_clean_change_and_signed_identity(self):
        for terminal in (True,False):
            scheduler=DDIMScheduler(clip_sample=False,set_alpha_to_one=terminal)
            scheduler.set_timesteps(30)
            for t in scheduler.timesteps:
                p=ddim_endpoints(scheduler,self.x,self.v,t)
                e=(self.x-p.alpha*self.c)/p.sigma
                torch.testing.assert_close(p.alpha*self.c+p.sigma*e,self.x,atol=2e-6,rtol=2e-6)
                actual=transition(self.x,self.c,p,False)
                torch.testing.assert_close(actual,p.next_alpha*self.c+p.next_sigma*e,atol=2e-6,rtol=2e-6)
                torch.testing.assert_close(transition(self.x,p.clean,p,False),p.reconstruct_next(),atol=3e-6,rtol=3e-6)
                expected=-(p.next_sigma/p.sigma)*p.alpha*(self.c-p.clean)
                torch.testing.assert_close(actual-p.reconstruct_next(clean=self.c),expected,atol=3e-6,rtol=3e-6)
                transition_diagnostics(self.x,self.c,p,actual)
            self.assertEqual(float(p.next_sigma)==0,terminal)

    def test_i_ignores_old_endpoint_but_g_does_not(self):
        p=flow_endpoints(self.x,self.v,.7,.4)
        altered=replace(p,endpoint=p.endpoint+10.)
        self.assertTrue(torch.equal(transition(self.x,self.c,p),transition(self.x,self.c,altered)))
        self.assertFalse(torch.equal(p.reconstruct_next(clean=self.c),altered.reconstruct_next(clean=self.c)))

    def test_terminal_zero_and_input_guards(self):
        actual=interpolate_from_current_state(self.c,self.c,1,0,1,0,flow=True)
        self.assertTrue(torch.equal(actual,self.c))
        for x,sn in ((self.x,0),(self.c,.2)):
            with self.assertRaises(ValueError):interpolate_from_current_state(x,self.c,1,0,1-sn,sn,flow=True)
        with self.assertRaises(ValueError):interpolate_from_current_state(self.x,self.c.double(),1,.1,1,0)
        with self.assertRaises(ValueError):interpolate_from_current_state(self.x,self.c,1,float('nan'),1,0)

    def test_zero_correction_full_g_i_equivalence(self):
        geometry=NativeMultiDiffusionConfig(canvas_height=4,canvas_width=8,patch_size=4,stride=2)
        results=[]
        for mode in ('preserve_prefusion_endpoint','preserve_current_state'):
            backend=ConsensusMock()
            pipe=PlanarImpliedEndpointConsensusPipeline(native_config=geometry,backend=backend,residual_correction=True,transition_mode=mode)
            results.append(pipe.run(pipe.initialize_local_states(self.x),{'offset':torch.tensor(.3)}))
        torch.testing.assert_close(results[0].canvas_rgb,results[1].canvas_rgb,atol=2e-6,rtol=2e-6)
        self.assertLess(max(r['clean_consensus_delta_mae'] for r in results[1].transition_diagnostics),2e-6)

    def test_jacobi_order_counts_and_independent_current_state_oracle(self):
        geometry=NativeMultiDiffusionConfig(canvas_height=4,canvas_width=8,patch_size=4,stride=2)
        results=[]
        for order in ([0,1,2],[2,1,0]):
            backend=ContextMock()
            pipe=PlanarImpliedEndpointConsensusPipeline(native_config=geometry,backend=backend,residual_correction=True,
                transition_mode='preserve_current_state',patch_order=order)
            local=pipe.initialize_local_states(self.x)
            result=pipe.run(local,{'offset':torch.tensor(.3)})
            self.assertEqual(backend.guided_prediction_count,15)
            self.assertEqual(backend.encodes,30)
            for seen,i in zip(backend.forward_inputs[:3],order):self.assertTrue(torch.equal(seen,local[i]))
            ref=ContextMock();states=[s.clone() for s in local]
            for t in ref.timesteps:
                pairs=[ref.predict_clean_and_endpoint(s,t,{'offset':torch.tensor(.3)}) for s in states]
                clean_canvas=torch.zeros_like(self.x);residual_canvas=torch.zeros_like(self.x);counts=torch.zeros_like(self.x)
                for p,pair in zip(pipe.rgb_layout.patches,pairs):
                    rgb=ref.decode_clean(pair.clean)
                    region=(slice(None),slice(None),slice(p.y,p.y+p.size),slice(p.x,p.x+p.size))
                    clean_canvas[region]+=rgb;residual_canvas[region]+=pair.clean-ref.encode_clean(rgb);counts[region]+=1
                clean_canvas/=counts;residual_canvas/=counts
                following=[]
                for p,pair,x in zip(pipe.rgb_layout.patches,pairs,states):
                    region=(slice(None),slice(None),slice(p.y,p.y+p.size),slice(p.x,p.x+p.size))
                    c=ref.encode_clean(clean_canvas[region])+residual_canvas[region]
                    e=(x-pair.alpha*c)/pair.sigma
                    following.append(pair.next_alpha*c+pair.next_sigma*e)
                states=following
            for actual,expected in zip(result.local_states,states):torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
            results.append(result)
        self.assertTrue(torch.equal(results[0].canvas_rgb,results[1].canvas_rgb))

    def test_four_i_configs_and_historical_g_h_snapshots(self):
        specs=read('configs/experiments/vae_residual/i-all-models.json')
        self.assertEqual(list(specs),['sd35','flux','sana','sd2'])
        for spec in specs.values():
            config,_=check_configs(retained_reference(spec))
            self.assertEqual(config.fusion.mode,'average')
            self.assertEqual(config.consensus_transition.mode,'preserve_current_state')
        from scripts.detail_preserving_experiment import check_configs as check_h
        for spec in read('configs/experiments/vae_residual/h-all-models.json').values():check_h(retained_reference(spec))


if __name__=='__main__':unittest.main()
