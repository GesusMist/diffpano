import math
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import torch
from diffpano.camera import spherediff_camera_cover, CubeFixedCameraSampler, SphereDiffFixedCameraSampler, camera_for_direction
from diffpano.conditioning import expand_directional_prompts, camera_prompt_indices
from diffpano.config import ViewConfig, FusionConfig, WarpConfig, load_experiment_config
from diffpano.dense_consensus import DenseERPLocalCurrentStatePipeline, prepare_camera_conditioning
from diffpano.dense_geometry import measure_cover, search_dense_cover
from diffpano.erp_local_consensus import ERPLocalCurrentStatePipeline, camera_digest
from diffpano.projection import ProjectionCache
from diffpano.warp import StandardWarpOperator
from test_erp_local_consensus import LocalMock


class DenseTests(unittest.TestCase):
    def test_official_cover_count_construction_and_fixed_slots(self):
        # Independent transcription of SphereDiff 2c8c68b spherical_functions.py
        # horizontal_and_vertical_view_dirs_v3_fov_xy_dense_equator, lines 297-322.
        angles = torch.linspace(0,90,math.ceil(130/32)).tolist()
        angles += [-x for x in angles[1:]];angles.sort(key=lambda x:abs(x+.01))
        reference=[]
        for phi in angles:
            n=math.ceil(math.cos(math.radians(phi))*360/32)+3
            yaw=torch.linspace(-math.pi,math.pi,n+1)[:-1]
            p=torch.ones_like(yaw)*math.radians(phi)
            reference.append(torch.stack([p.cos()*yaw.sin(),p.sin(),p.cos()*yaw.cos()],dim=-1))
        cameras=spherediff_camera_cover(ViewConfig(height=8,width=8,fov_x=80,fov_y=80))
        self.assertEqual(len(cameras),89)
        torch.testing.assert_close(torch.stack([c.forward() for c in cameras]),torch.cat(reference),atol=2e-7,rtol=2e-7)
        sampler=SphereDiffFixedCameraSampler(ViewConfig())
        self.assertEqual(camera_digest(sampler.sample(0,20)),camera_digest(sampler.sample(19,20)))
        self.assertTrue(all(c.fov_x==80 and c.fov_y==80 for c in cameras))

    def test_official_prompt_semantics_and_yaw_anchors(self):
        labels=['top','upper','equator','lower','bottom'];bank=expand_directional_prompts(labels)
        for band,(our_pitch,official_phi) in enumerate(zip((90,10,0,-10,-90),(-90,-10,0,10,90))):
            # Official extraction rotates the central negative-z ray with R^T;
            # its physical vertical component is -sin(phi), hence pitch=-phi.
            self.assertAlmostEqual(math.sin(math.radians(our_pitch)),-math.sin(math.radians(official_phi)))
            for yaw_slot,yaw in enumerate((0,90,180,270)):
                c=camera_for_direction(yaw,our_pitch,height=8,width=8)
                slot=int(camera_prompt_indices([c],bank.directions)[0])
                self.assertEqual(slot//4,band);self.assertEqual(bank.prompts[slot],labels[band])
                if abs(our_pitch)!=90:self.assertEqual(slot%4,yaw_slot)
        # All bands use the same maximum-cosine rule as official get_prompt_indices.
        cameras=spherediff_camera_cover(ViewConfig())
        forwards=torch.stack([c.forward() for c in cameras])
        official_dirs=bank.directions.clone();official_dirs[:,1]*=-1
        parameters=forwards.clone();parameters[:,1]*=-1
        scores=parameters@official_dirs.T
        slots=camera_prompt_indices(cameras,bank.directions)
        torch.testing.assert_close(scores[torch.arange(89),slots],scores.max(1).values)

    def test_all_five_backend_conditioning_rows_and_values(self):
        from diffpano.pipelines.sd2 import SD2ViewDenoiser,SD2PromptBank
        from diffpano.pipelines.sana import SanaViewDenoiser,SanaPromptBank
        from diffpano.pipelines.flux import FluxViewDenoiser,FluxPromptBank
        from diffpano.pipelines.sd35 import SD35ViewDenoiser,SD35PromptBank
        from test_pixeldit import make_adapter
        from diffpano.pipelines.pixeldit import PixelDiTPromptBank
        directions=expand_directional_prompts(['top','upper','equator','lower','bottom']).directions
        vals=torch.arange(20).float()[:,None,None];mask=torch.ones(20,1)
        configs=[(SD2ViewDenoiser,SD2PromptBank(directions,vals,-vals)),
            (SanaViewDenoiser,SanaPromptBank(directions,vals,mask,-vals,mask)),
            (FluxViewDenoiser,FluxPromptBank(directions,vals,vals[:,0],torch.zeros(1,3),None,None,None)),
            (SD35ViewDenoiser,SD35PromptBank(directions,vals,vals[:,0],-vals,-vals[:,0]))]
        cameras=[camera_for_direction(0,p,height=8,width=8) for p in (90,10,0,-10,-90)]
        for cls,bank in configs:
            adapter=object.__new__(cls)
            adapter.pipeline=SimpleNamespace(_execution_device=torch.device('cpu'),device=torch.device('cpu'),transformer=SimpleNamespace(device=torch.device('cpu')))
            adapter.guidance_scale=4.5
            conds,slots=prepare_camera_conditioning(adapter,bank,cameras,'spherediff_directional',batch_size=2)
            self.assertEqual(slots,[0,4,8,12,16])
            for slot,c in zip(slots,conds):
                value=c if isinstance(c,torch.Tensor) else c['embeds']
                expected=2 if cls is FluxViewDenoiser else 4
                self.assertEqual(value.shape[0],expected)
                self.assertTrue((value[-2:]==slot).all())
        adapter=make_adapter();bank=PixelDiTPromptBank(directions,vals[:,None],mask,torch.zeros(1,1,1,1),torch.ones(1,1))
        conds,slots=prepare_camera_conditioning(adapter,bank,cameras,'spherediff_directional',batch_size=2)
        self.assertTrue(all(c.positive.shape[0]==2 and c.negative.shape[0]==2 for c in conds))
        for slot,c in zip(slots,conds):self.assertTrue((c.positive==slot).all())
        conds,slots=prepare_camera_conditioning(adapter,bank,cameras,'original_k_global_slot_8')
        self.assertEqual(slots,[8]*5);self.assertTrue(all(c is conds[0] for c in conds))

    def test_streaming_matches_original_k_and_order_invariance_without_endpoint(self):
        cameras=CubeFixedCameraSampler(ViewConfig(height=12,width=12,fov_x=100,fov_y=100)).sample(0,5)
        outputs=[]
        for order in (list(range(6)),list(reversed(range(6)))):
            backend=LocalMock();op=StandardWarpOperator(WarpConfig(mode="standard"),FusionConfig(mode='average',weight_mode='uniform'),ProjectionCache(max_entries=2,cpu_fallback=True))
            pipe=DenseERPLocalCurrentStatePipeline(backend=backend,cameras=cameras,erp_size=(16,32),warp_operator=op,view_order=order)
            states=pipe.initialize_local_states(7);conditioning={'offset':torch.tensor(.3)}
            original=backend.predict_clean_and_endpoint
            def poisoned(*args,**kwargs):
                pair=original(*args,**kwargs);return replace(pair,endpoint=torch.full_like(pair.endpoint,float('nan')))
            with patch.object(backend,'predict_clean_and_endpoint',side_effect=poisoned), \
                 patch('diffpano.vae_residual.vae_residual',side_effect=AssertionError('residual')), \
                 patch('diffpano.noise.FixedPatchNoiseBank',side_effect=AssertionError('fixed noise')):
                result=pipe.run_dense(states,[conditioning]*6)
            self.assertEqual(backend.guided_prediction_count,30);self.assertEqual(backend.encodes,30)
            self.assertEqual(backend.decode_count,36)
            self.assertLess(result.metrics['current_state_error_max']['max'],2e-6)
            self.assertFalse(result.audit['erp_latent']);self.assertFalse(result.audit['spherical_latent'])
            for seen,i in zip(backend.forward_inputs[:6],order):self.assertTrue(torch.equal(seen,states[i]))
            self.assertFalse(hasattr(result,'transition_records'));self.assertFalse(hasattr(result,'final_views'))
            outputs.append(result.erp_rgb)
        torch.testing.assert_close(*outputs,atol=2e-6,rtol=2e-6)
        backend=LocalMock();old=ERPLocalCurrentStatePipeline(backend=backend,cameras=cameras,erp_size=(16,32),warp_operator=StandardWarpOperator(WarpConfig(mode="standard"),FusionConfig(mode='average',weight_mode='uniform')))
        expected=old.run(old.initialize_local_states(7),{'offset':torch.tensor(.3)}).erp_rgb
        torch.testing.assert_close(outputs[0],expected,atol=2e-6,rtol=2e-6)

    def test_search_determinism_and_strict_full_resolution_gate(self):
        # Full production grids are independently required by the geometry job.
        args=dict(overlaps=(.5,.55,.6),resolutions=((64,128,32),))
        first=search_dense_cover(**args);second=search_dense_cover(**args)
        self.assertEqual(first,second);self.assertEqual(first[0],.6)
        self.assertGreaterEqual(first[2][0]['minimum'],5)
        self.assertEqual(first[2][0]['coverage_percent'],100)
        with self.assertRaises(RuntimeError):search_dense_cover(overlaps=(.0,),cap=16,resolutions=((16,32,8),))

    def test_all_configs_pairing_and_forbidden_modes(self):
        from scripts.dense_erp_experiment import check_pairing
        for label in ('l','m'):
            for backend in ('sd35','flux','sana','sd2','pixeldit'):
                config=load_experiment_config('configs/experiments/erp_later/'+backend+'-'+label+'.yaml')
                check_pairing(config)
                with self.assertRaises(ValueError):replace(config,warp=WarpConfig(mode='lpw')).validate()
                with self.assertRaises(ValueError):replace(config,sampling=replace(config.sampling,rotation=replace(config.sampling.rotation,enabled=True))).validate()

if __name__=='__main__':unittest.main()
