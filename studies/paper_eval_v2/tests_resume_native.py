"""Native resume scheduler tests without GPU work or scheduler submissions."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .common import write
from . import resume_native as rn


class NativeResumeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.run = Path(temp.name)
        self.manifest = dict(expected=240, rows=[dict(index=i) for i in range(240)],
            generation_groups=[list(range(i, i+30)) for i in range(0, 240, 30)])
        write(self.run / "native/manifest.json", self.manifest)
        self.scheduler = dict(jobs={"native-v1": dict(job_id="old", role="generation", scheduler_state="FAILED")}, attempts=[], queue_snapshot=[])

    def call(self, pending, execute=False):
        with patch("studies.paper_eval_v2.native.frozen", return_value=self.manifest) as frozen, \
             patch("studies.paper_eval_v2.native.complete", side_effect=lambda run, row: row["index"] not in pending), \
             patch("studies.paper_eval_v2.resources.reconcile", return_value=self.scheduler), \
             patch("studies.paper_eval_v2.resources.balance", return_value=10000), \
             patch("studies.paper_eval_v2.resources.submit", return_value=dict(job_id="new")) as submit:
            result = rn.resume(self.run, execute)
            frozen.assert_called_once_with(self.run)
            return result, submit

    def test_documented_cli_submits_and_dry_run_is_explicit(self):
        with patch("sys.argv", ["resume_native", "--run", str(self.run)]), patch.object(rn, "resume", return_value={}) as resume:
            rn.main()
            resume.assert_called_once_with(self.run, execute=True)
        with patch("sys.argv", ["resume_native", "--run", str(self.run), "--dry-run"]), patch.object(rn, "resume", return_value={}) as resume:
            rn.main()
            resume.assert_called_once_with(self.run, execute=False)

    def test_only_unfinished_groups_are_planned_by_default(self):
        result, submit = self.call({31, 239})
        self.assertEqual(result["groups"], [1, 7])
        self.assertEqual(result["unfinished_indices"], [31, 239])
        self.assertEqual(result["array"], "1,7%2")
        self.assertEqual(result["reservation_su"], 292)
        self.assertTrue(result["dry_run"])
        submit.assert_not_called()

    def test_new_resume_key_does_not_reuse_the_failed_native_v1_job(self):
        self.scheduler["jobs"]["native-resume-v1"] = dict(job_id="previous", role="generation", scheduler_state="FAILED")
        result, submit = self.call({31}, execute=True)
        self.assertEqual(result["key"], "native-resume-v2")
        self.assertTrue(result["submitted"])
        self.assertEqual(submit.call_args.args[1], "native-resume-v2")
        self.assertEqual(submit.call_args.kwargs["options"], ("--array=1%2", "--exclude=fc026"))
        self.assertEqual(submit.call_args.kwargs["reserve_su"], 146)

    def test_finished_bank_does_not_submit_anything(self):
        result, submit = self.call(set(), execute=True)
        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["verified_images"], 240)
        submit.assert_not_called()

    def test_pending_generation_refuses_resume_but_evaluation_is_independent(self):
        self.scheduler["jobs"]["core"] = dict(job_id="core", role="generation", scheduler_state="PENDING")
        with self.assertRaisesRegex(RuntimeError, "Conflicting generation"):
            self.call({0}, execute=True)
        self.scheduler["jobs"]["core"]["role"] = "evaluation"
        result, _ = self.call({0}, execute=True)
        self.assertTrue(result["submitted"])

    def test_ambiguous_prior_submission_cannot_be_bypassed_with_a_new_key(self):
        self.scheduler["attempts"] = [dict(key="native-resume-v3", role="generation", state="ambiguous_submission")]
        with self.assertRaisesRegex(RuntimeError, "ambiguous submission"):
            self.call({0}, execute=True)
        self.assertEqual(rn.next_version(self.scheduler), 4)

    def test_untracked_gpu_is_a_conflict_and_manifest_partition_is_validated(self):
        self.scheduler["queue_snapshot"] = [dict(job_id="external", resources="gres/gpu:a100:1", name="other")]
        result, _ = self.call({0})
        self.assertEqual(result["state"], "blocked")
        self.manifest["generation_groups"][0].append(30)
        with self.assertRaisesRegex(ValueError, "partition"):
            self.call({0})


if __name__ == "__main__":
    unittest.main()
