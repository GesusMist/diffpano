import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from studies.camera_count_v2 import common,random_search
from studies.camera_count_v2.common import rows,expected_counts,geometry_identity
from studies.camera_count_v2.run import resolve

class ProtocolTests(unittest.TestCase):
    def test_forbidden_sphere_strategies_rejected(self):
        for strategy in ('fibonacci','random'):
            with self.assertRaises(ValueError):
                resolve(argparse.Namespace(index=None,family='spherediff_camera_override',strategy=strategy,
                    camera_count=70,backend='flux',prompt='ruins'))
    def test_first_passing_seed_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);calls=[];original=random_search.build
            def build(strategy,n,seed):
                calls.append(seed);return original(strategy,n,seed)
            with patch.object(common,'ROOT',root),patch.object(random_search,'ROOT',root),patch.dict('os.environ',{'SLURM_JOB_ID':'test'}),\
                 patch.object(random_search,'build',side_effect=build),\
                 patch.object(random_search,'quick_pass',side_effect=[False,False,True,True]),\
                 patch.object(random_search,'scan',return_value={'passed':True}),\
                 patch.object(random_search,'diffpano_support',return_value={'passed':True}):
                before=torch.random.get_rng_state()
                self.assertTrue(random_search.search(30,seconds=60,device='cpu'))
                self.assertEqual(calls,[0,1,2])
                self.assertEqual(common.read(common.layout_path('random',30))['accepted_seed'],2)
                self.assertEqual(common.read(root/'layouts/random_search_n30.json')['next_seed'],3)
                torch.testing.assert_close(before,torch.random.get_rng_state(),rtol=0,atol=0)
                self.assertTrue(random_search.search(30,seconds=60,device='cpu'))
                self.assertEqual(calls,[0,1,2])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);calls=[]
            progress=dict(n=50,base_seed=0,next_seed=7,last_attempted_seed=6,rejected_by_gate={'A':7},
                elapsed_seconds=1.,geometry_identity=geometry_identity(),rng='test',status='searching')
            common.atomic(root/'layouts/random_search_n50.json',progress)
            with patch.object(common,'ROOT',root),patch.object(random_search,'ROOT',root),patch.dict('os.environ',{'SLURM_JOB_ID':'test'}),\
                 patch.object(random_search,'build',side_effect=build),patch.object(random_search,'quick_pass',return_value=True),\
                 patch.object(random_search,'scan',return_value={'passed':True}),patch.object(random_search,'diffpano_support',return_value={'passed':True}):
                self.assertTrue(random_search.search(50,seconds=60,device='cpu'))
                self.assertEqual(calls,[7])
    def test_ambiguous_submission_is_reconciled_without_resubmission(self):
        from studies.camera_count_v2 import launch
        intent=dict(event='SUBMISSION_INTENT',token='ccv2:test',task='run/0',kind='run',source_fingerprint='frozen')
        with patch.object(launch,'events',return_value=[intent]),patch.object(launch,'shell_command',side_effect=['12345|ccv2:test','']),patch.object(launch,'ledger') as ledger:
            launch.reconcile_intents()
            self.assertEqual(ledger.call_args.kwargs['job'],'12345')
        with patch.object(launch,'events',return_value=[intent]),patch.object(launch,'shell_command',side_effect=['','']),patch.object(launch,'ledger') as ledger:
            with self.assertRaises(RuntimeError):launch.reconcile_intents()
            ledger.assert_not_called()
    def test_partial_pair_never_complete(self):
        with tempfile.TemporaryDirectory() as d,patch.object(common,'ROOT',Path(d)):
            r=rows()[0];p=common.folder(r);p.mkdir(parents=True)
            (p/'final.png').write_bytes(b'incomplete')
            self.assertFalse(common.complete(r))
            (p/'config.json').write_text('{}')
            self.assertFalse(common.complete(r))
    def test_call_counts_for_both_families(self):
        for r in rows():
            c=expected_counts(r);n=r['num_cameras'];s=r['steps']
            if r['family']=='spherediff_camera_override':
                self.assertEqual(c['decode'],n);self.assertEqual(c['encode'],0)
            elif r['backend']=='pixeldit':
                self.assertEqual(c['decode'],0);self.assertEqual(c['encode'],0)
            else:
                self.assertEqual(c['decode'],n*s+n);self.assertEqual(c['encode'],2*n*s)
if __name__=='__main__':unittest.main()
