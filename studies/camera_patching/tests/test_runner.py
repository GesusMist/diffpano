import unittest
from unittest.mock import patch
import torch
from diffpano.config import ViewConfig
from studies.tt_cea.tests.test_pipeline import IndependentMock,config
from studies.tt_cea.pipeline import ExperimentalPipeline
from studies.tt_cea.schedule import prepare_interval_table
from studies.all_prompts.runtime import ordinary_run
from studies.camera_patching.cameras import cover
from studies.camera_patching.common import rows,expected_counts

class RunnerTests(unittest.TestCase):
    def test_matrix(self):
        r=rows();self.assertEqual(len(r),48);self.assertEqual(len({x['output'] for x in r}),48)
        for strategy in ('fibonacci','random'):
            for projection in ('erp','cea'):
                self.assertEqual(sum(x['strategy']==strategy and x['projection']==projection for x in r),12)
    def test_generic_trajectory_bridge_counts_and_no_replay(self):
        for strategy in ('fibonacci','random'):
            cams=cover(strategy,ViewConfig(height=4,width=4),89)
            for pixel in (False,True):
                for projection in ('erp','cea'):
                    b=IndependentMock(pixel);p=ExperimentalPipeline(b,cams,config(pixel),projection=projection,size=(8,16))
                    states=[torch.ones(1,3,4,4)*.2 for _ in cams];cond=[{'offset':torch.tensor(.1)}]*len(cams)
                    with patch('studies.tt_cea.pipeline.OriginalNoiseBank',side_effect=AssertionError('No original noise bank')),patch('studies.tt_cea.pipeline.backward_original_noise',side_effect=AssertionError('No time travel')):
                        native,image,detail=ordinary_run(p,states,cond,prepare_interval_table(b))
                    self.assertEqual(detail['guided_predictions'],len(cams)*3)
                    self.assertEqual(detail['replay_count'],0);self.assertEqual(detail['backward_calls'],0)
                    self.assertEqual(len(b.bridge_inputs),0 if pixel else 2*len(cams)*3)
                    self.assertTrue(torch.isfinite(image).all())
    def test_call_counts_current_and_arbitrary(self):
        for m in (1,64,89,128):
            for steps in (20,40,50):
                self.assertEqual(expected_counts(m,steps)['decode'],m*(steps+1))
                self.assertEqual(expected_counts(m,steps,True)['encode'],0)

    def test_runtime_config_publication_hash_and_overwrite_guard(self):
        import os
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from PIL import Image
        from studies.camera_patching.common import publish,complete,read,sha,validate_config
        from studies.camera_patching.runtime import compact_config
        from studies.camera_patching.cameras import routing
        from studies.all_prompts.audit import configuration
        from studies.all_prompts.common import prompt_record,REPO
        for backend in ('sana','flux','sd35','pixeldit'):
            row=next(r for r in rows() if r['strategy']=='fibonacci' and r['backend']==backend and r['projection']=='cea' and r['prompt']=='ruins')
            c,stub,_,_,old=configuration(backend)
            cams=cover('fibonacci',c.view,89)
            b=SimpleNamespace(dtype=torch.bfloat16,native_channels=stub.native_channels,
                native_spatial_shape_for_rgb=stub.native_spatial_shape_for_rgb,timesteps=torch.arange(row['steps']),
                guidance_scale=c.generation.guidance_scale,true_cfg_scale=c.generation.true_cfg_scale,
                cfg_scale=c.pixeldit.cfg_scale,negative_prompt=c.pixeldit.negative_prompt,interval_guidance=c.pixeldit.interval_guidance,
                solver=SimpleNamespace(flow_shift=4.),pipeline=SimpleNamespace(vae=SimpleNamespace(use_tiling=c.model.vae_tiling)))
            pipeline=SimpleNamespace(cameras=cams,canvas=SimpleNamespace(spec=SimpleNamespace(projection='cea',height=2048,width=4096)),bridge_mode='local_identity_preserving')
            route=routing(cams,prompt_record(REPO/'prompts/ruins.txt')[0]['effective_lines'])
            detail=dict(time_travel_enabled=False,replay_count=0,backward_calls=0,original_noise_reinjections=0)
            with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'SLURM_JOB_ID':'unit-test'}),patch('torch.cuda.max_memory_allocated',return_value=0),patch('studies.camera_patching.common.job_success',return_value=True):
                row=dict(row,output=d)
                record=compact_config(row,c,b,pipeline,old,route,old['initialization'],detail,
                    expected_counts(len(cams),row['steps'],backend=='pixeldit'),dict(git_commit='fixture',git_dirty=True,fingerprint='fixture'),0.)
                publish(Image.new('RGB',(4096,2048),'blue'),record,row)
                self.assertTrue(complete(row));saved=read(Path(d)/'config.json');validate_config(saved,row)
                self.assertEqual(saved['output']['sha256'],sha(Path(d)/'final.png'))
                self.assertEqual({p.name for p in Path(d).iterdir()},{'final.png','config.json'})
                before=sha(Path(d)/'final.png')
                with self.assertRaises(FileExistsError):publish(Image.new('RGB',(4096,2048),'red'),record,row)
                self.assertEqual(before,sha(Path(d)/'final.png'))
