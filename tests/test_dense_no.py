"""L/N/O ablation and snapshot/trajectory invariants."""
import hashlib
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import torch
from diffpano.camera import CubeFixedCameraSampler, spherediff_camera_cover, camera_for_direction
from diffpano.conditioning import camera_prompt_indices, expand_directional_prompts
from diffpano.config import ViewConfig, FusionConfig, WarpConfig, LPWConfig, load_experiment_config
from diffpano.dense_consensus import DenseERPLocalCurrentStatePipeline, snapshot_schedule
from diffpano.erp_local_consensus import camera_digest
from diffpano.lpw import build_laplacian_pyramid, reconstruct_laplacian_pyramid, pyramid_upsample
from diffpano.projection import ProjectionCache
from diffpano.warp import StandardWarpOperator, LaplacianPyramidWarpOperator
from test_erp_local_consensus import LocalMock

class DenseNOTests(unittest.TestCase):
    def make(self,label,order=None):
        b=LocalMock()
        cameras=CubeFixedCameraSampler(ViewConfig(height=32,width=32,fov_x=100,fov_y=100)).sample(0,5)
        warp=WarpConfig(mode='lpw' if label=='O' else 'standard',lpw=LPWConfig(5,'none'))
        fusion=FusionConfig(mode='average' if label=='L' else 'detail_preserving_average',weight_mode='uniform')
        cache=ProjectionCache(max_entries=2,cpu_fallback=True)
        op=(LaplacianPyramidWarpOperator(warp,fusion,cache,periodic_reconstruction=True) if label=='O'
            else StandardWarpOperator(warp,fusion,cache))
        return b,DenseERPLocalCurrentStatePipeline(backend=b,cameras=cameras,erp_size=(32,64),warp_operator=op,view_order=order)

    def test_all_five_resolved_ablation_configs_and_prompt_slots(self):
        from scripts.dense_erp_experiment import check_l_pairing,config_differences
        bank=expand_directional_prompts(['top','upper','equator','lower','bottom'])
        for b in ('sd35','flux','sana','sd2','pixeldit'):
            configs={e:load_experiment_config('configs/experiments/erp_later/'+b+'-'+e.lower()+'.yaml') for e in ('L','N','O')}
            ref=configs['L'];cameras=spherediff_camera_cover(ref.view)
            fake={'config':ref.to_dict(),'prompt_sha256':hashlib.sha256(Path(ref.prompt.path).read_bytes()).hexdigest()}
            for e in ('N','O'):
                with patch('scripts.dense_erp_experiment.read',return_value=fake):check_l_pairing(configs[e])
                actual=spherediff_camera_cover(configs[e].view)
                self.assertEqual(camera_digest(cameras),camera_digest(actual))
                self.assertTrue(torch.equal(camera_prompt_indices(cameras,bank.directions),camera_prompt_indices(actual,bank.directions)))
            difference=config_differences(configs['N'].to_dict(),configs['O'].to_dict())
            self.assertEqual(set(difference),{'dense_consensus.experiment','warp.mode','warp.lpw.levels','warp.lpw.lod_mode'})

    def test_validation_preserves_historical_and_new_guards(self):
        for e in ('l','m','n','o'):
            c=load_experiment_config('configs/experiments/erp_later/sd35-'+e+'.yaml')
            for bad in (replace(c,consensus_transition=replace(c.consensus_transition,vae_residual_correction=True)),
                        replace(c,sampling=replace(c.sampling,rotation=replace(c.sampling.rotation,enabled=True))),
                        replace(c,fusion=replace(c.fusion,mode='weighted_average'))):
                with self.assertRaises(ValueError):bad.validate()
        c=load_experiment_config('configs/experiments/erp_later/sd35-o.yaml')
        for bad in (replace(c,warp=replace(c.warp,mode='standard')),
                    replace(c,warp=replace(c.warp,lpw=replace(c.warp.lpw,lod_mode='jacobian'))),
                    replace(c,dense_consensus=replace(c.dense_consensus,geometry_file='experiment_m.json'))):
            with self.assertRaises(ValueError):bad.validate()

    def test_snapshot_mapping_integer_ceil_and_duplicates(self):
        for steps in (20,30,40,50):
            mapping=snapshot_schedule(steps);self.assertEqual(len(mapping),9)
            for step,ps in mapping.items():self.assertEqual(step,ps[0]*steps//100)
            self.assertNotIn(steps,mapping)
        self.assertEqual(snapshot_schedule(3),{1:[10,20,30],2:[40,50,60],3:[70,80,90]})
        with self.assertRaises(ValueError):snapshot_schedule(0)

    def test_same_initial_stream_and_L_digest_guard(self):
        results=[]
        for e in ('L','N','O'):
            _,p=self.make(e);results.append(p.initialize_local_states(12))
        for streams in zip(*results):
            self.assertTrue(torch.equal(streams[0],streams[1]));self.assertTrue(torch.equal(streams[0],streams[2]))
        b,p=self.make('N')
        with self.assertRaisesRegex(AssertionError,'Initial local states differ'):
            p.run_dense(results[1],[None]*6,expected_initial_sha256='wrong')
        self.assertEqual(getattr(b,'guided_prediction_count',0),0)

    def test_NO_Jacobi_counts_and_snapshot_invariance(self):
        for label in ('N','O'):
            outputs=[];hashes=[]
            for reverse,snapshots in ((False,False),(True,True),(False,True)):
                order=list(reversed(range(6))) if reverse else list(range(6))
                b,p=self.make(label,order);states=p.initialize_local_states(7);saved=[]
                predict=b.predict_clean_and_endpoint
                def poison(*args,**kwargs):
                    pair=predict(*args,**kwargs)
                    return replace(pair,endpoint=torch.full_like(pair.endpoint,float('nan')))
                def callback(record,rgb):
                    saved.append(record);rgb.zero_()
                with patch.object(b,'predict_clean_and_endpoint',side_effect=poison), \
                     patch('diffpano.vae_residual.vae_residual',side_effect=AssertionError('residual')), \
                     patch('diffpano.noise.FixedPatchNoiseBank',side_effect=AssertionError('noise')):
                    result=p.run_dense(states,[{'offset':torch.tensor(.3)}]*6,snapshot_callback=callback if snapshots else None)
                self.assertEqual(b.guided_prediction_count,30);self.assertEqual(b.encodes,30);self.assertEqual(b.decode_count,36)
                self.assertEqual(len(saved),5 if snapshots else 0)
                for seen,i in zip(b.forward_inputs[:6],order):self.assertTrue(torch.equal(seen,states[i]))
                self.assertLess(result.metrics['current_state_error_max']['max'],2e-6)
                self.assertEqual(result.audit['fusion'],'detail_preserving_average')
                self.assertEqual(result.audit['warp'],'lpw' if label=='O' else 'standard')
                self.assertEqual(type(p.operator),LaplacianPyramidWarpOperator if label=='O' else StandardWarpOperator)
                if label=='O':self.assertTrue({'pyramid_construction','pyramid_level_fusion','erp_reconstruction'}<=set(result.stage_seconds))
                outputs.append(result.erp_rgb);hashes.append(result.audit['initial_local_sha256'])
            torch.testing.assert_close(outputs[0],outputs[1],atol=3e-6,rtol=3e-6)
            self.assertTrue(torch.equal(outputs[0],outputs[2]));self.assertEqual(len(set(hashes)),1)

    def test_O_coefficient_projection_and_five_DPA_accumulators(self):
        _,p=self.make('O');acc=p._accumulator(1);camera=p.cameras[0]
        from diffpano import warp
        with patch('diffpano.warp.perspective_to_erp',wraps=warp.perspective_to_erp) as project:
            acc.accumulate(torch.randn(1,3,32,32),camera)
        self.assertEqual([call.args[0].shape[-1] for call in project.call_args_list[:5]],[32,16,8,4,2])
        self.assertEqual(len(acc.level_accumulators),5)
        self.assertTrue(all(a.detail_num is not None and a.detail_den is not None for a in acc.level_accumulators))
        from diffpano.projection import erp_to_perspective
        with patch('diffpano.warp.erp_to_perspective',wraps=erp_to_perspective) as project:
            p.operator.erp_to_perspective(torch.randn(1,3,32,64),camera)
        self.assertEqual([call.args[0].shape[-1] for call in project.call_args_list],[64,32,16,8,4])

    def test_O_periodic_reconstruction_and_wrap_each_level(self):
        image=torch.randn(1,3,32,64)
        pyramid=build_laplacian_pyramid(image,5,spherical_erp=True,periodic_upsampling=True)
        torch.testing.assert_close(reconstruct_laplacian_pyramid(pyramid,spherical_erp=True),image,atol=2e-6,rtol=2e-6)
        for level in pyramid[1:]:
            size=tuple(v*2 for v in level.shape[-2:])
            lhs=pyramid_upsample(torch.roll(level,1,-1),size,spherical_erp=True)
            rhs=torch.roll(pyramid_upsample(level,size,spherical_erp=True),2,-1)
            torch.testing.assert_close(lhs,rhs,atol=2e-6,rtol=2e-6)
        _,p=self.make('O')
        for yaw in (-180,180):
            c=camera_for_direction(yaw,0,height=32,width=32)
            view=p.operator.erp_to_perspective(image,c)
            if yaw==-180:first=view
            else:torch.testing.assert_close(first,view,atol=2e-5,rtol=2e-5)

    def test_O_DPA_constant_dense_cover(self):
        _,p=self.make('O');acc=p._accumulator(1)
        for c in p.cameras:acc.accumulate(torch.full((1,3,32,32),.625),c)
        result=acc.finalize();self.assertTrue(result.coverage_mask.all())
        torch.testing.assert_close(result.erp_rgb,torch.full_like(result.erp_rgb,.625),atol=2e-6,rtol=2e-6)

if __name__=='__main__':unittest.main()
