"""Integration guards around frozen workers; no API or scheduler calls."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import prepare_production,snapshot


class AdapterTests(unittest.TestCase):
    def test_manifest_cli_does_not_write_while_api_worker_owns_lock(self):
        with patch('sys.argv',['prepare_production','--run','/unused']), \
             patch.object(prepare_production.vlm,'worker_lock',side_effect=BlockingIOError), \
             patch.object(prepare_production,'prepare') as prepare:
            with self.assertRaises(BlockingIOError):prepare_production.main()
            prepare.assert_not_called()

    def test_manifest_cli_holds_worker_lock_for_entire_mutation(self):
        held=[]
        @contextmanager
        def guard(run):
            held.append(run)
            try:yield
            finally:held.clear()
        def prepare(run,views):
            self.assertEqual(held,[run])
            return {'state':'fixture'}
        with patch('sys.argv',['prepare_production','--run','/fixture']), \
             patch.object(prepare_production.vlm,'worker_lock',guard), \
             patch.object(prepare_production,'prepare',prepare),patch('builtins.print'):
            prepare_production.main()
        self.assertFalse(held)

    def test_report_launcher_reuses_active_report_allocation(self):
        ledger={'jobs':{'report-snapshot-old':{'job_id':'5','role':'cpu','scheduler_state':['RUNNING']}},
                'queue_snapshot':[{'job_id':'5'}]}
        with tempfile.TemporaryDirectory() as folder, \
             patch('studies.paper_eval_v2.resources.reconcile',return_value=ledger), \
             patch('studies.paper_eval_v2.resources.submit') as submit:
            (Path(folder)/'report').mkdir()
            result=snapshot.launch(Path(folder))
            self.assertEqual(result['state'],'already_pending_or_active')
            submit.assert_not_called()

    def test_report_launcher_uses_finite_script_and_shared_journal(self):
        ledger={'jobs':{},'queue_snapshot':[]}
        with tempfile.TemporaryDirectory() as folder, \
             patch('studies.paper_eval_v2.resources.reconcile',return_value=ledger), \
             patch('studies.paper_eval_v2.resources.submit',return_value={'job_id':'6'}) as submit:
            (Path(folder)/'report').mkdir()
            result=snapshot.launch(Path(folder))
            self.assertEqual(result['job_id'],'6')
            args,kwargs=submit.call_args
            self.assertEqual(args[2].name,'snapshot.slurm')
            self.assertEqual(kwargs['role'],'cpu')
            self.assertEqual(kwargs['script_args'],('--run',folder))
            self.assertGreater(kwargs['reserve_su'],0)

if __name__=='__main__':unittest.main()
