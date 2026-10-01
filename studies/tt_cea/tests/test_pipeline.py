import unittest
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import replace
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffpano.pipelines.native_state import NativeStateMixin
from diffpano.pipelines.endpoints import flow_endpoints,flow_bounds
from diffpano.bridge_factorial import BridgeFactorialPipeline
from diffpano.camera import PerspectiveCamera
from diffpano.config import WarpConfig,FusionConfig
from diffpano.current_state_transition import interpolate_from_current_state
from studies.tt_cea.pipeline import ExperimentalPipeline
from studies.tt_cea.schedule import prepare_interval_table,eligible_indices
from studies.tt_cea.cameras import fibonacci_centers

class IndependentMock(NativeStateMixin):
    device=torch.device('cpu');native_channels=3;native_spatial_factor=1;native_initial_noise_sigma=1.
    def __init__(self,pixel=False):
        self.pipeline=SimpleNamespace(scheduler=FlowMatchEulerDiscreteScheduler());self.pipeline.scheduler.set_timesteps(3)
        self.timesteps=self.pipeline.scheduler.timesteps;self.guided_prediction_count=0;self.pixel=pixel
        self.inputs=[];self.proposals=[];self.bridge_inputs=[];self.terminal_inputs=[];self.poison=False
    def predict_clean_and_endpoint(self,x,t,c):
        self.guided_prediction_count+=1;self.inputs.append(x.clone())
        v=.11*x+.07*torch.sin(x)+c['offset'];p=flow_endpoints(x,v,*flow_bounds(self.pipeline.scheduler,t,x))
        self.proposals.append(p.clean.clone())
        if self.poison:p=replace(p,endpoint=torch.full_like(x,float('nan')))
        return p
    def decode_clean(self,x):return x if self.pixel else .8*x+.1
    def encode_clean(self,x):
        if self.pixel:raise AssertionError('Pixel VAE')
        self.bridge_inputs.append(x.clone());return .7*x-.2
    def decode_native_canvas(self,x):self.terminal_inputs.append(x.clone());return self.decode_clean(x)

def config(pixel=False):
    return SimpleNamespace(model=SimpleNamespace(pipeline='pixeldit' if pixel else 'flux'),warp=WarpConfig(mode='standard'),
        fusion=FusionConfig(mode='weighted_average',weight_mode='spherediff_center',spherediff_temperature=.1))

class PipelineTests(unittest.TestCase):
    def cams(self):return [PerspectiveCamera(**p,height=4,width=4) for p in fibonacci_centers()]
    def initial(self):return [torch.randn(1,3,4,4,generator=torch.Generator().manual_seed(i)) for i in range(89)]
    def test_full_noop_against_actual_frozen_loop(self):
        for pixel in (False,True):
            b=IndependentMock(pixel);c=config(pixel);cams=self.cams();states=self.initial();cond=[{'offset':torch.tensor(.12)}]*89
            p=ExperimentalPipeline(b,cams,c,size=(8,16));trace=[]
            frozen=BridgeFactorialPipeline(backend=b,cameras=cams,erp_size=(8,16),warp_operator=p.canvas.standard,backend_name=c.model.pipeline)
            def capture(*args,**kwargs):
                out=interpolate_from_current_state(*args,**kwargs);trace.append(out.clone());return out
            with patch('diffpano.dense_consensus.interpolate_from_current_state',side_effect=capture):ref=frozen.run_dense(states,cond)
            fresh=IndependentMock(pixel);new=ExperimentalPipeline(fresh,cams,c,size=(8,16));observed=[];x=states
            for interval in prepare_interval_table(fresh):
                x,summary=new.advance_interval(x,cond,interval,pass_kind='initial',diagnostics=lambda kind,i,v:observed.append(v['state'].clone()) if kind=='next' else None)
            final,erp,_=new.terminal(x)
            torch.testing.assert_close(erp,ref.erp_rgb,atol=2e-6,rtol=2e-6)
            for a,z in zip(trace,observed):torch.testing.assert_close(a,z,atol=2e-6,rtol=2e-6)
            for field in ('inputs','proposals','bridge_inputs'):
                self.assertEqual(len(getattr(b,field)),len(getattr(fresh,field)))
                for a,z in zip(getattr(b,field),getattr(fresh,field)):torch.testing.assert_close(a,z,atol=2e-6,rtol=2e-6)
    def test_replay_fresh_calls_endpoint_poison_and_noise_immutability(self):
        results=[]
        for poison in (False,True):
            b=IndependentMock();b.poison=poison;p=ExperimentalPipeline(b,self.cams(),config(),size=(8,16))
            initial=self.initial();saved=[x.clone() for x in initial];before=b.timesteps.clone()
            _,erp,record=p.run(initial,[{'offset':torch.tensor(.1)}]*89,prepare_interval_table(b),time_travel=True)
            self.assertEqual(b.guided_prediction_count,89*(3+len(eligible_indices(3))))
            self.assertEqual([x['k'] for x in record['diagnostics'] if x['pass_kind']=='replay'],list(eligible_indices(3)))
            self.assertEqual(record['original_noise_sha256_before'],record['original_noise_sha256_after'])
            self.assertTrue(torch.equal(before,b.timesteps));self.assertTrue(all(torch.equal(a,z) for a,z in zip(initial,saved)))
            results.append(erp)
        torch.testing.assert_close(*results,atol=0,rtol=0)
