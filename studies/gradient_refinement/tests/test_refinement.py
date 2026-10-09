import sys,unittest,copy
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path('tests').resolve()))
from test_endpoint_trajectory import EndpointMock
from test_x0_consensus import FullFrameWarp,FullFrameJointWarp,OrderedSampler
from diffpano.refinement import RefinementConfig,NativeRefinement,continue_native,independent_count
from diffpano.config import (GlobalPipelineConfig,FusionConfig,CleanConsensusConfig,PlanarConfig,NativeMultiDiffusionConfig,ExperimentConfig,load_experiment_config)
from diffpano.camera import PerspectiveCamera
from diffpano.erp_pipeline import ERPRGBPipeline
from diffpano.erp_x0_pipeline import ERPX0ConsensusPipeline
from diffpano.planar_pipeline import PlanarRGBPipeline,PlanarX0ConsensusPipeline
from diffpano.native_multidiffusion import NativeMultiDiffusionPipeline
from diffpano.implied_endpoint_consensus import PlanarImpliedEndpointConsensusPipeline
from diffpano.gradient_fusion import GradientSettings
from studies.tt_cea.tests.test_pipeline import IndependentMock,PipelineTests,config
from studies.tt_cea.schedule import prepare_interval_table
from studies.gradient_blending.operator import StudyPipeline
from studies.all_prompts.runtime import ordinary_run

class Backend(EndpointMock):
    native_channels=3;native_spatial_factor=1
    def __init__(self,pixel=False):
        super().__init__();self.encode_count=0;self.pixel_native=pixel;self.events=[]
    def encode_clean(self,x):
        if self.pixel_native:raise AssertionError('Pixel VAE encode')
        self.encode_count+=1;self.events.append('encode');return x.clone()
    def decode_clean(self,x):
        if self.pixel_native:raise AssertionError('Pixel VAE decode')
        self.decode_count+=1;self.events.append('decode');return x.clone()
    def decode_native_canvas(self,x):return x if self.pixel_native else self.decode_clean(x)
    def denoise_native_step(self,*a):self.events.append('native');return super().denoise_native_step(*a)
    def denoise_step(self,x,t,c):
        state=x if self.pixel_native else self.encode_clean(x)
        state=self.denoise_native_step(state,t,c)
        return state if self.pixel_native else self.decode_clean(state)
    def conditioning_for_cameras(self,bank,cameras,*,batch_size):return bank
    def conditioning_for_prompt_indices(self,bank,indices,*,batch_size):return bank
    def sample_fixed_noise(self,*,batch_size,height,width,generator):return torch.randn(batch_size,3,height,width,generator=generator)
    def add_fixed_noise(self,clean,noise,t):
        from diffpano.pipelines.clean_prediction import flow_add_noise
        return flow_add_noise(self.pipeline.scheduler,clean,noise,t)

class StudyBackend(IndependentMock):
    def denoise_native_step(self,x,t,c):
        self.guided_prediction_count+=1;self.inputs.append(x.clone())
        v=.11*x+.07*torch.sin(x)+c['offset']
        return self.pipeline.scheduler.step(v,t,x).prev_sample

class RefinementTests(unittest.TestCase):
    def test_cutoff_and_legacy_configuration(self):
        self.assertEqual(independent_count(20,.1),2);self.assertEqual(independent_count(28,.1),3)
        for n in (1,3,20):
            self.assertEqual(independent_count(n,0),0);self.assertEqual(independent_count(n,1),n)
        for f in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):RefinementConfig(f)
        c=load_experiment_config('config.yaml');self.assertEqual(c.global_pipeline.refinement.last_fraction,0)
        self.assertNotIn('refinement',c.to_dict()['global_pipeline'])
        c.global_pipeline.refinement.last_fraction=.1;self.assertEqual(c.to_dict()['global_pipeline']['refinement']['last_fraction'],.1)
    def test_direct_continuation_order_identity_schedule(self):
        conditions=[{'offset':torch.tensor(.1+i*.1)} for i in range(3)]
        initial=[torch.randn(1,3,4,4) for _ in conditions];expected=[]
        for state,cond in zip(initial,conditions):
            b=Backend()
            for t in b.timesteps[3:]:state=b.denoise_native_step(state,t,cond)
            expected.append(state)
        for order in ([0,1,2],[2,0,1]):
            b=Backend();actual,tail=continue_native(b,initial,['a','b','c'],conditions,3,order=order)
            for a,e in zip(actual,expected):torch.testing.assert_close(a,e,atol=0,rtol=0)
            self.assertEqual(b.encode_count+b.decode_count,0);self.assertEqual(len(tail.records),2)
            self.assertTrue(all(r['fusion'] is None for r in tail.records))
        b=Backend();tail=NativeRefinement(b,['a','b','c'],conditions,3)
        with self.assertRaisesRegex(AssertionError,'geometry'):tail.step(initial,3,geometry=['b','a','c'])
        with self.assertRaisesRegex(AssertionError,'conditioning'):tail.step(initial,3,conditions=list(reversed(conditions)))
    def test_erp_rgb_dynamic_freeze_and_joint_terminal(self):
        camera=PerspectiveCamera(0.,0.,0.,80.,80.,4,4);condition={'offset':torch.tensor(.2)}
        class Moving(OrderedSampler):
            def __init__(self):super().__init__([camera,camera]);self.visits=[]
            def sample(self,i,n):self.visits.append(i);return [replace(camera,yaw=i*.1),replace(camera,yaw=i*.1+.2)]
        for pixel in (False,True):
            for joint in (False,True):
                b=Backend(pixel);sampler=Moving();warp=FullFrameJointWarp() if joint else FullFrameWarp()
                pipe=ERPRGBPipeline(camera_sampler=sampler,warp_operator=warp,fusion_config=FusionConfig(mode='average',weight_mode='uniform'),view_denoiser=b,refinement_config=RefinementConfig(.4))
                result=pipe.run(torch.randn(1,3,4,4),condition)
                self.assertEqual(sampler.visits,[0,1,2,3]);self.assertEqual(len(result.steps),5)
                self.assertEqual(b.guided_prediction_count,10)
                self.assertEqual(b.encode_count,0 if pixel else 8)
                self.assertEqual(b.decode_count,0 if pixel else 8)
                self.assertEqual(len(pipe.refinement_tail.records),2)
    def test_clean_and_planar_canvas_modes_all_tail_and_partial(self):
        cond={'offset':torch.tensor(.1)};cam=PerspectiveCamera(0.,0.,0.,80.,80.,4,4)
        fusion=FusionConfig(mode='average',weight_mode='uniform');planar=PlanarConfig(height=4,width=8,patch_size=4,stride=2)
        for fraction in (.4,1.):
            for family in ('erp_clean','planar_clean','planar_rgb'):
                b=Backend();ref=RefinementConfig(fraction)
                if family=='erp_clean':
                    p=ERPX0ConsensusPipeline(camera_sampler=OrderedSampler([cam,cam]),warp_operator=FullFrameWarp(),fusion_config=fusion,consensus_config=CleanConsensusConfig(),backend=b,refinement_config=ref)
                    result=p.run(cond,batch_size=1,erp_height=4,erp_width=4);count=2
                elif family=='planar_clean':
                    p=PlanarX0ConsensusPipeline(planar_config=planar,fusion_config=fusion,consensus_config=CleanConsensusConfig(),backend=b,refinement_config=ref)
                    result=p.run(cond,batch_size=1);count=3
                else:
                    p=PlanarRGBPipeline(planar_config=planar,fusion_config=fusion,backend=b,refinement_config=ref)
                    result=p.run(torch.zeros(1,3,4,8),cond);count=3
                self.assertEqual(len(result.steps),5);self.assertEqual(b.guided_prediction_count,5*count)
                self.assertEqual(len(p.refinement_tail.records),independent_count(5,fraction))
                # Last native interval is followed exclusively by one decode per view.
                self.assertEqual(b.events[-count:],['decode']*count)
    def test_native_and_implied_terminal_contract(self):
        cfg=NativeMultiDiffusionConfig(4,8,4,2);condition={'offset':torch.tensor(.2)}
        for family in ('native','implied'):
            b=Backend();initial=torch.randn(1,3,4,8)
            if family=='native':
                p=NativeMultiDiffusionPipeline(native_config=cfg,backend=b,refinement_config=RefinementConfig(.4));result=p.run(initial,condition)
                self.assertEqual(b.decode_count,1);self.assertEqual(b.encode_count,0)
            else:
                p=PlanarImpliedEndpointConsensusPipeline(native_config=cfg,backend=b,refinement_config=RefinementConfig(.4),roundtrip_diagnostics=False)
                result=p.run(p.initialize_local_states(initial),condition)
                self.assertEqual(b.decode_count,3*3+3);self.assertEqual(b.encode_count,3*3)
            self.assertEqual(b.guided_prediction_count,15);self.assertEqual(len(result.steps),5)
    def test_actual_study_transition_all_backends_and_gradients(self):
        helper=PipelineTests();cameras=helper.cams();initial=helper.initial();conds=[{'offset':torch.tensor(.1)}]*89
        for name in ('flux','sana','sd2','sd35','pixeldit'):
            # Local adapter identity and native step protocol are shared; installed
            # adapter-specific scheduler arithmetic is covered by native_scheduler tests.
            pixel=name=='pixeldit';c=config(pixel);c.model.pipeline=name;c.global_pipeline=GlobalPipelineConfig()
            for mode in ('rgb','poisson_select','poisson_max'):
                before=[]
                for frac in (0.,1/3):
                    cfg=copy.deepcopy(c);cfg.global_pipeline.refinement=RefinementConfig(frac)
                    b=StudyBackend(pixel);p=StudyPipeline(b,cameras,cfg,GradientSettings(mode),list(range(89)),size=(8,16))
                    # SD2 needs DDIM endpoint coefficients; this mock is flow and we
                    # explicitly keep its flow arithmetic, testing routing, not DDIM.
                    p.flow=True
                    _,out,details=ordinary_run(p,initial,conds,prepare_interval_table(b))
                    before.append(b.inputs[:178])
                    self.assertEqual(b.guided_prediction_count,267)
                    self.assertEqual(len(p.canvas.records),3 if frac else 4)
                    self.assertEqual(len(b.bridge_inputs),0 if pixel else (356 if frac else 534))
                    self.assertEqual(len(b.terminal_inputs),0 if pixel else 89)
                    if frac:self.assertEqual(p.interval_records[-1]['phase'],'independent_refinement');self.assertIsNone(p.interval_records[-1]['fusion'])
                for a,b in zip(*before):self.assertTrue(torch.equal(a,b))
    def test_manifest_unique(self):
        from studies.gradient_refinement.common import rows
        data=rows();self.assertEqual(len(data),64);self.assertEqual(len({r['key'] for r in data}),64);self.assertEqual(sum(r['expected_reuse'] for r in data),4)

class LocalModesTests(unittest.TestCase):
    def test_local_dense_bridge_tail_and_all_independent(self):
        from diffpano.erp_local_consensus import ERPLocalCurrentStatePipeline
        from diffpano.dense_consensus import DenseERPLocalCurrentStatePipeline
        from diffpano.bridge_factorial import BridgeFactorialPipeline
        from diffpano.warp import StandardWarpOperator,LaplacianPyramidWarpOperator
        from diffpano.config import WarpConfig,LPWConfig
        h=PipelineTests();cams=h.cams();states=h.initial();cond={'offset':torch.tensor(.1)}
        for family in ('local','dense','bridge'):
            for fraction in (.4,1.):
                b=Backend();fusion=FusionConfig(mode='average',weight_mode='uniform') if family!='bridge' else FusionConfig(mode='weighted_average',weight_mode='spherediff_center')
                warp=StandardWarpOperator(WarpConfig(mode="standard"),fusion)
                kwargs=dict(backend=b,cameras=cams,erp_size=(8,16),warp_operator=warp,refinement_config=RefinementConfig(fraction))
                if family=='local':
                    p=ERPLocalCurrentStatePipeline(**kwargs);result=p.run(states,cond)
                else:
                    cls=DenseERPLocalCurrentStatePipeline if family=='dense' else BridgeFactorialPipeline
                    if family=='bridge':kwargs['backend_name']='flux'
                    p=cls(**kwargs);result=p.run_dense(states,[cond]*89)
                self.assertEqual(b.guided_prediction_count,5*89);self.assertEqual(len(p.refinement_tail.records),independent_count(5,fraction))
                if fraction==1.:
                    self.assertEqual(b.encode_count,0);self.assertEqual(b.decode_count,89)
                self.assertTrue(bool(torch.isfinite(result.erp_rgb).all()))
    def test_normal_dispatch_routes_all_families(self):
        from scripts.generate import _generate_with_selected_global_pipeline
        combinations=[('erp','erp_rgb_state','generate_erp_rgb'),('erp','erp_x0_consensus','generate_erp_x0_consensus'),
            ('planar','erp_rgb_state','generate_planar_rgb'),('planar','erp_x0_consensus','generate_planar_x0_consensus'),
            ('planar','native_multidiffusion','generate_planar_native_multidiffusion'),
            ('planar','implied_endpoint_consensus','generate_planar_implied_endpoint_consensus'),
            ('erp','erp_local_current_consensus','generate_erp_local_current_state')]
        for canvas,mode,function in combinations:
            c=ExperimentConfig();c.canvas.mode=canvas;c.global_pipeline.mode=mode;c.global_pipeline.refinement=RefinementConfig(.1)
            target=('diffpano.erp_local_consensus.' if mode=='erp_local_current_consensus' else 'scripts.generate.')+function
            with patch(target,return_value='sentinel') as call:
                self.assertEqual(_generate_with_selected_global_pipeline(c,object(),None),'sentinel');self.assertIs(call.call_args.args[0],c)
        for mode in ('erp_local_dense_consensus','erp_bridge_factorial'):
            c=ExperimentConfig();c.global_pipeline.mode=mode;c.global_pipeline.refinement=RefinementConfig(.1)
            with patch('diffpano.dense_generation.generate_dense',return_value='sentinel') as call:
                self.assertEqual(_generate_with_selected_global_pipeline(c,object(),None),'sentinel');self.assertIs(call.call_args.args[0],c)
