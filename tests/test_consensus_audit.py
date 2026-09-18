import unittest
from dataclasses import replace
import torch
from diffpano.config import FusionConfig,WarpConfig,ViewConfig,load_experiment_config
from diffpano.fusion import RGBFusionAccumulator,create_view_weight_map
from diffpano.projection import ERPContribution,ProjectionCache
from diffpano.consensus_audit import StageAudit,milestones,camera_slots,image_metrics
from diffpano.camera import CubeFixedCameraSampler,spherediff_camera_cover
from diffpano.dense_consensus import DenseERPLocalCurrentStatePipeline
from diffpano.warp import StandardWarpOperator
from test_erp_local_consensus import LocalMock

class AuditTests(unittest.TestCase):
    def test_milestones_and_physical_slots(self):
        self.assertEqual(milestones(3),[1,2,3]);self.assertEqual(milestones(20),[2,10,18,20])
        c=spherediff_camera_cover(ViewConfig());ids=camera_slots(c)
        self.assertEqual(len(set(ids)),4)
        self.assertAlmostEqual(c[ids[0]].pitch,0);self.assertAlmostEqual(c[ids[1]].pitch,0)
        self.assertGreater(float(c[ids[0]].forward()@c[ids[1]].forward()),.8)
        self.assertGreater(c[ids[2]].pitch,0);self.assertAlmostEqual(c[ids[3]].pitch,torch.pi/2)

    def test_weight_normalization_and_influence(self):
        def fuse(order=(0,1),scale=1.,repeat=1,constant=False):
            a=RGBFusionAccumulator(torch.zeros(1,3,8,8),FusionConfig(mode='weighted_average',weight_mode='spherediff_center'))
            center=create_view_weight_map(8,8,'spherediff_center')*scale
            for _ in range(repeat):
                for i in order:
                    mask=torch.ones_like(center);mask[...,0,0]=0
                    value=torch.full((1,3,8,8),.4 if constant else float(i))
                    a.accumulate(ERPContribution(value,mask,center if i==0 else center.flip(-1)*.1))
            return a.finalize().erp_rgb
        base=fuse();self.assertAlmostEqual(float(base[0,0,4,4]),1/11,places=6)
        for candidate in (fuse(order=(1,0)),fuse(repeat=2),fuse(scale=1e-12)):
            torch.testing.assert_close(base,candidate,atol=1e-7,rtol=1e-6)
        constant=fuse(constant=True,scale=1e-12)
        self.assertEqual(float(constant[...,0,0].abs().max()),0)
        torch.testing.assert_close(constant[...,1:,1:],torch.full_like(constant[...,1:,1:],.4))

    def test_stage_hooks_leave_trajectory_rng_inputs_and_counts_unchanged(self):
        cameras=CubeFixedCameraSampler(ViewConfig(height=12,width=12,fov_x=100,fov_y=100)).sample(0,5)
        outputs=[];inputs=[]
        for enabled in (False,True):
            backend=LocalMock();operator=StandardWarpOperator(WarpConfig(mode='standard'),FusionConfig(mode='average',weight_mode='uniform'),ProjectionCache(max_entries=2))
            pipe=DenseERPLocalCurrentStatePipeline(backend=backend,cameras=cameras,erp_size=(16,32),warp_operator=operator)
            states=pipe.initialize_local_states(7);copies=[x.clone() for x in states]
            rng=torch.random.get_rng_state().clone();cond={'offset':torch.tensor(.3)}
            audit=StageAudit(cameras,5,'flux') if enabled else None
            result=pipe.run_dense(states,[cond]*6,stage_audit=audit)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            for x,y in zip(states,copies):self.assertTrue(torch.equal(x,y))
            self.assertEqual(backend.guided_prediction_count,30);self.assertEqual(backend.encodes,30)
            self.assertEqual(backend.decode_count,36+(12 if enabled else 0))
            outputs.append(result.erp_rgb);inputs.append(backend.forward_inputs)
        self.assertTrue(torch.equal(*outputs))
        for x,y in zip(*inputs):self.assertTrue(torch.equal(x,y))

    def test_planar_first_order_oracle_and_independent_initialization(self):
        from diffpano.config import NativeMultiDiffusionConfig
        from diffpano.planar_initialization_control import paired_run,independent_run
        from test_detail_preserving_consensus import ContextMock
        g=NativeMultiDiffusionConfig(canvas_height=4,canvas_width=8,patch_size=4,stride=2)
        b=ContextMock();initial=torch.randn(1,3,4,8,generator=torch.Generator().manual_seed(4))
        native,shared,records,calls=paired_run(b,g,initial,{'offset':torch.tensor(.3)})
        torch.testing.assert_close(native,shared,atol=2e-6,rtol=2e-6)
        self.assertEqual(calls,30)
        self.assertLess(max(r['shared_overlap_max_abs'] for r in records),2e-6)
        generator=torch.Generator().manual_seed(4)
        independent=[torch.randn(1,3,4,4,generator=generator) for _ in range(3)]
        result,_,calls=independent_run(b,g,independent,{'offset':torch.tensor(.3)})
        self.assertEqual(calls,15);self.assertFalse(torch.allclose(result,shared))

    def test_dense_interventions_and_historical_guards(self):
        from scripts.consensus_dense_controls import make_config,pairing
        for name in ('flux','pixeldit'):
            for label in ('P','S','T'):
                c=make_config(label,name);pairing(c)
                if label=='S':
                    c.fusion.mode='average'
                    with self.assertRaises(ValueError):c.validate()
        c=load_experiment_config('configs/experiments/erp_later/flux-l.yaml')
        c.fusion.mode='weighted_average';c.fusion.weight_mode='spherediff_center'
        with self.assertRaises(ValueError):c.validate()

    def test_q_coherent_operator_duplication_and_smooth_direction_signal(self):
        from scripts.consensus_spatial_controls import assemble,signal,frequency_retention
        from diffpano.geometry import erp_world_directions
        from diffpano.projection import perspective_world_rays
        from diffpano.warp import LaplacianPyramidWarpOperator
        cameras=spherediff_camera_cover(ViewConfig(height=16,width=16))
        def source(c):
            return perspective_world_rays(c,device=torch.device('cpu')).permute(2,0,1)[None]*.3
        for label in ('l','n','o'):
            c=load_experiment_config('configs/experiments/erp_later/flux-'+label+'.yaml')
            op=LaplacianPyramidWarpOperator(c.warp,c.fusion,ProjectionCache(max_entries=2),periodic_reconstruction=True) if label=='o' else StandardWarpOperator(c.warp,c.fusion,ProjectionCache(max_entries=2))
            a=assemble(op,cameras,(16,32),source)
            b=assemble(op,cameras,(16,32),source,repeat=2)
            self.assertTrue(bool(a.coverage_mask.all()))
            torch.testing.assert_close(a.erp_rgb,b.erp_rgb,atol=2e-6,rtol=2e-6)
        reference=signal(erp_world_directions(32,64,device=torch.device('cpu')))
        self.assertTrue(bool(torch.isfinite(reference).all()))
        for band in frequency_retention(reference).values():self.assertAlmostEqual(band['retention'],1.,places=4)

    def test_metrics_preserve_raw_range(self):
        x=torch.full((1,3,8,8),2.);before=x.clone();m=image_metrics(x)
        self.assertEqual(m['mean'],2);self.assertEqual(m['out_of_range_fraction'],1)
        self.assertTrue(torch.equal(x,before))

if __name__=='__main__':unittest.main()
