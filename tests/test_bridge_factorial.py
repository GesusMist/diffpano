"""Factorial causal controls, bridge/transition and exact saved camera tests."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import torch
from diffpano.bridge_factorial import *
from diffpano.camera import CubeFixedCameraSampler
from diffpano.config import ViewConfig, FusionConfig, WarpConfig, LPWConfig
from diffpano.fusion import RGBFusionAccumulator
from diffpano.projection import ERPContribution, perspective_world_rays
from diffpano.erp_noise_initialization import ERPNoiseConfig, initialize_erp_noise, native_cameras, primary_noise_size
from diffpano.pipelines.endpoints import EndpointPrediction
from diffpano.current_state_transition import interpolate_from_current_state
from scripts.bridge_factorial_common import verify_prompt, reference_specs
from test_detail_preserving_consensus import ContextMock

class BridgeMock(ContextMock):
    def __init__(self,pixel=False,ddim=False):super().__init__();self.pixel=pixel;self.ddim=ddim;self.poison=False
    def encode_clean(self,rgb):
        if self.pixel:raise AssertionError('PixelDiT VAE encode')
        self.encodes+=1
        return rgb*.83+.17
    def predict_clean_and_endpoint(self,state,timestep,conditioning):
        p=super().predict_clean_and_endpoint(state,timestep,conditioning)
        if self.ddim:p=replace(p,alpha=(1-p.sigma.square()).sqrt(),next_alpha=(1-p.next_sigma.square()).sqrt())
        if self.poison:p=replace(p,endpoint=torch.full_like(p.endpoint,float('nan')))
        return p

class FactorialTests(unittest.TestCase):
    def test_prompt_bytes_physical_lines_and_directional_routing(self):
        from diffpano.conditioning import expand_directional_prompts,camera_prompt_indices
        p=verify_prompt();self.assertEqual(p['physical_lines'],5);self.assertEqual(p['sha256'],PROMPT_SHA)
        c=make_config('flux','A0B0C0D0');cams=saved_cameras(c.view,(c.erp.height,c.erp.width))
        bank=expand_directional_prompts(p['lines']);slots=camera_prompt_indices(cams,bank.directions)
        rays=torch.stack([v.forward() for v in cams]);scores=rays@bank.directions.T
        self.assertTrue(torch.equal(slots,scores.argmax(1)))
        self.assertEqual(len(bank.prompts),20)
        self.assertEqual(set((slots//4).tolist()),set(range(5)))

    def test_exact_80_unique_cells_common_angular_geometry_and_settings(self):
        from collections import Counter
        rows=[];hashes=set()
        for name in BACKENDS:
            base=None
            for cell in factor_cells():
                c=make_config(name,cell);cams=saved_cameras(c.view,(c.erp.height,c.erp.width));rows.append((name,cell))
                hashes.add(digest(angular_geometry(cams)))
                self.assertEqual(len(cams),89);self.assertTrue(all(v.fov_x==v.fov_y==80 for v in cams))
                if base is None:base=common_settings(c)
                self.assertEqual(common_settings(c),base)
                self.assertEqual(c.consensus_transition.mode,'preserve_current_state')
            counts=Counter(round(__import__('math').degrees(v.pitch),1) for v in cams)
            self.assertEqual(dict(counts),{-90.:4,-67.5:8,-45.:11,-22.5:14,0.:15,22.5:14,45.:11,67.5:8,90.:4})
        self.assertEqual(len(set(rows)),80);self.assertEqual(len(hashes),1)

    def test_factorial_and_historical_guards(self):
        c=make_config('flux','A1B1C1D1')
        bad=[replace(c,consensus_transition=replace(c.consensus_transition,mode='preserve_prefusion_endpoint')),
             replace(c,consensus_transition=replace(c.consensus_transition,vae_residual_correction=False)),
             replace(c,fusion=replace(c.fusion,mode='average')),replace(c,view=replace(c.view,fov_x=90)),
             replace(c,prompt=replace(c.prompt,path='prompts/native_control.txt'))]
        for value in bad:
            with self.assertRaises(ValueError):value.validate()
        old=load_experiment_config('configs/experiments/erp_later/flux-l.yaml')
        with self.assertRaises(ValueError):replace(old,consensus_transition=replace(old.consensus_transition,vae_residual_correction=True)).validate()

    def test_cd_formulas_do_not_erase_spatial_weights(self):
        vals=[.2,.8];weights=[.7,.1]
        for cbit in (0,1):
            for dbit in (0,1):
                conf=FusionConfig(mode='detail_preserving_average' if cbit else 'weighted_average',weight_mode='spherediff_center' if dbit else 'uniform')
                acc=RGBFusionAccumulator(torch.zeros(1,3,2,4),conf);ws=weights if dbit else [1.,1.]
                for value,w in zip(vals,ws):acc.accumulate(ERPContribution(torch.full((1,3,2,4),value),torch.ones(1,1,2,4),torch.full((1,1,2,4),w)))
                expected=sum(v*w for v,w in zip(vals,ws))/sum(ws)
                if cbit:expected=sum(v*w*(abs(v)+conf.epsilon) for v,w in zip(vals,ws))/sum(w*(abs(v)+conf.epsilon) for v,w in zip(vals,ws))
                torch.testing.assert_close(acc.finalize().erp_rgb,torch.full((1,3,2,4),expected),atol=1e-6,rtol=1e-6)

    def tiny(self,cell,name='flux',reverse=False):
        c=make_config(name,cell);cams=CubeFixedCameraSampler(ViewConfig(height=16,width=16,fov_x=100,fov_y=100)).sample(0,5)
        b=BridgeMock(pixel=name=='pixeldit',ddim=name=='sd2');b.timesteps=b.timesteps[:2]
        op=make_operator(c)
        p=BridgeFactorialPipeline(backend=b,cameras=cams,erp_size=(16,32),warp_operator=op,backend_name=name,view_order=list(reversed(range(6))) if reverse else None)
        return b,p

    def test_all_BCD_paths_use_same_cameras_and_level_weights(self):
        for cell in factor_cells()[:8]:
            b,p=self.tiny(cell);f=factors(cell)
            self.assertIs(type(p.operator),LaplacianPyramidWarpOperator if f['B'] else StandardWarpOperator)
            if f['B']:
                acc=p._accumulator(1);acc.accumulate(torch.ones(1,3,16,16),p.cameras[0])
                self.assertEqual(len(acc.level_accumulators),5)
                for a in acc.level_accumulators:
                    self.assertEqual(a.config.mode,p.operator.fusion_config.mode);self.assertEqual(a.config.weight_mode,p.operator.fusion_config.weight_mode)
                    self.assertEqual(a.detail_num is not None,bool(f['C']))
            self.assertEqual(p.operator.fusion_config.weight_mode,'spherediff_center' if f['D'] else 'uniform')
            from diffpano.projection import erp_to_perspective
            with patch('diffpano.warp.erp_to_perspective',wraps=erp_to_perspective) as project:
                p.operator.erp_to_perspective(torch.zeros(1,3,16,32),p.cameras[0])
            self.assertEqual(project.call_count,5 if f['B'] else 1)
            for call in project.call_args_list:
                actual=call.args[1]
                self.assertEqual((actual.yaw,actual.pitch,actual.roll,actual.fov_x,actual.fov_y),
                                 (p.cameras[0].yaw,p.cameras[0].pitch,p.cameras[0].roll,p.cameras[0].fov_x,p.cameras[0].fov_y))
            self.assertEqual(project.call_args_list[0].args[1],p.cameras[0])

    def test_bridge_identity_and_locality(self):
        b,p=self.tiny('A0B0C0D0');z=torch.randn(1,3,16,16);rgb=b.decode_clean(z);r=p._local_clean_residual(z,rgb,{})
        recovered=p._consensus_native_clean(rgb,r,{})
        torch.testing.assert_close(recovered,z,atol=1e-6,rtol=1e-6)
        other=p._consensus_native_clean(rgb+1,r,{})
        torch.testing.assert_close(other,z+.83,atol=1e-6,rtol=1e-6)
        b,p=self.tiny('A0B0C0D0','pixeldit');self.assertIsNone(p._local_clean_residual(z,rgb,{}));self.assertTrue(torch.equal(p._consensus_native_clean(rgb,None,{}),rgb))
        self.assertEqual(p.bridge_mode,'not_applicable_identity')

    def test_matched_A_sampling_and_one_time_source(self):
        b,p=self.tiny('A0B0C0D0');size=primary_noise_size(native_cameras(b,p.cameras));initial=[];records=[]
        for a in (0,1):
            s,r=initialize_erp_noise(b,p.cameras,ERPNoiseConfig('V-shared-erp' if a else 'V-independent-erp',*size,0));initial.append(s);records.append(r)
        for key in ['map_sha256','source_shape','native_local_shapes','scaling','first_camera_sha256']:self.assertEqual(records[0][key],records[1][key])
        self.assertEqual([r['source_draws'] for r in records],[6,1]);self.assertTrue(all(r['source_released'] for r in records))
        self.assertFalse(torch.equal(initial[0][1],initial[1][1]))
        with patch('diffpano.erp_noise_initialization.sample_source',side_effect=AssertionError('source used after initialization')):
            p.run_dense(initial[0],[{'offset':torch.tensor(.3)}]*6)

    def test_every_factor_transition_uses_bridged_clean_poison_ignored_and_jacobi(self):
        for name in ('flux','sd2','pixeldit'):
            for cell in factor_cells():
                outputs=[]
                for poison in (False,True):
                    b,p=self.tiny(cell,name,reverse=poison);b.poison=poison;states=p.initialize_local_states(0)
                    corrected=[];original=p._consensus_native_clean
                    def track(*args):
                        value=original(*args);corrected.append(value.detach().clone());return value
                    p._consensus_native_clean=track;seen=[]
                    def transition(x,clean,*args,**kwargs):
                        torch.testing.assert_close(clean,corrected[-1],atol=0,rtol=0);seen.append(clean)
                        return interpolate_from_current_state(x,clean,*args,**kwargs)
                    with patch('diffpano.dense_consensus.interpolate_from_current_state',side_effect=transition),patch.object(EndpointPrediction,'reconstruct_next',side_effect=AssertionError('old endpoint transition')):
                        result=p.run_dense(states,[{'offset':torch.tensor(.3)}]*6)
                    self.assertEqual(b.guided_prediction_count,12);self.assertEqual(len(seen),12)
                    self.assertEqual(b.encodes,0 if name=='pixeldit' else 24)
                    self.assertEqual(b.decode_count,18)
                    for value,i in zip(b.forward_inputs[:6],p.view_order):self.assertTrue(torch.equal(value,states[i]))
                    self.assertTrue(torch.isfinite(result.erp_rgb).all());self.assertLess(result.metrics['current_state_error_max']['max'],2e-6)
                    outputs.append(result.erp_rgb)
                torch.testing.assert_close(outputs[0],outputs[1],atol=8e-6,rtol=8e-6)

    def test_boundary_metric_constant_and_finite(self):
        from diffpano.factorial_diagnostics import camera_boundary_metric
        _,p=self.tiny('A0B0C0D0')
        record=camera_boundary_metric(torch.ones(1,3,16,32),p.cameras)
        self.assertEqual(record['boundary_gradient'],0.)
        self.assertEqual(record['interior_gradient'],0.)
        self.assertGreater(record['boundary_edges'],0)

    def test_factorial_contrast_signs_and_interactions(self):
        from diffpano.factorial_diagnostics import factorial_contrasts
        rows=[]
        for cell in factor_cells():
            f=factors(cell);z={k:2*v-1 for k,v in f.items()}
            rows.append(dict(cell=cell,**f,score=10+2*z['A']-3*z['B']+5*z['C']*z['D']))
        e=factorial_contrasts(rows,['score'])
        self.assertAlmostEqual(e['A']['score']['effect'],4)
        self.assertAlmostEqual(e['B']['score']['effect'],-6)
        self.assertAlmostEqual(e['CxD']['score']['effect'],10)
        self.assertAlmostEqual(e['AxD']['score']['effect'],0)
        with self.assertRaises(ValueError):factorial_contrasts(rows[:-1],['score'])

    def test_original_reference_settings_are_external_and_fixed(self):
        r=reference_specs();self.assertEqual(r['flux']['call']['n_spherical_points'],26500);self.assertEqual(r['sana']['call']['n_spherical_points'],2600)
        self.assertEqual(r['flux']['call']['num_inference_steps'],28)
        self.assertTrue(all(v['call']['erp_height']==2048 and v['call']['erp_width']==4096 and not v['enable_model_cpu_offload'] for v in r.values()))

if __name__=='__main__':unittest.main()
