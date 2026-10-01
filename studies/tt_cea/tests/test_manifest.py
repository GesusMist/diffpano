import tempfile
import unittest
from pathlib import Path
from studies.tt_cea.common import *
from studies.tt_cea.schedule import eligible_indices

class ManifestTests(unittest.TestCase):
    def test_exact_matrix_and_budget(self):
        counts=[];keys=set()
        for n in BACKENDS:
            for prompt in PROMPTS:
                for case in CASES:
                    c,_,_,m,r=resolved(n,prompt,case)
                    keys.add((n,prompt,case));counts.append(case)
                    self.assertEqual(r['camera_count'],89)
                    self.assertEqual(c.generation.num_inference_steps,STEPS[n]+len(eligible_indices(STEPS[n])) if case=='TB' else STEPS[n])
                    self.assertEqual(r['expected_guided_predictions'],89*r['downhill_passes'])
                    self.assertEqual(c.view.fov_x,80)
                    self.assertEqual(c.view.height,m['local_RGB_resolution'][0])
                    self.assertFalse(r['projection']=='cea' and r['time_travel']!='off')
        self.assertEqual(len(keys),60);self.assertEqual(counts.count('R0'),10)
    def test_immutable_refuses_mixing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json';immutable(p,{'seed':0});immutable(p,{'seed':0})
            with self.assertRaises(RuntimeError):immutable(p,{'seed':1})
    def test_fixed_settings(self):
        v=load_settings();self.assertEqual(v['canvas']['width'],4096)
        self.assertFalse(v['execution']['run_combined_cea_time_travel'])
    def test_baseline_report_reads_all_ten_records(self):
        from studies.tt_cea.report import audit_row
        for row in read(ROOT/'manifest.json')['rows']:
            if row['case']=='R0':
                result=audit_row(row)
                self.assertEqual(result['status'],'reused')
                self.assertEqual(result['actual_guided_predictions'],row['expected_guided_predictions'])
    def test_scheduler_sentinel_does_not_allow_nonfinite_diagnostics(self):
        value=historical_scheduler_json({'config':{'lambda_min_clipped':float('-inf')}})
        self.assertEqual(value['config']['lambda_min_clipped'],'-Infinity')
        with self.assertRaises(ValueError):historical_scheduler_json({'update':float('nan')})
    def test_batch_activation_uses_system_helper_after_module_purge(self):
        import os
        from unittest.mock import patch
        from studies.tt_cea.launch import submission_environment
        with patch.dict(os.environ,{'VIRTUAL_ENV':'/some/env','PATH':'/some/env/bin:/usr/bin:/bin','VIRTUAL_ENV_PROMPT':'env'}):
            env=submission_environment()
            self.assertEqual(env['PATH'],'/usr/bin:/bin')
            self.assertNotIn('VIRTUAL_ENV',env)
            self.assertEqual(os.environ['VIRTUAL_ENV'],'/some/env')
