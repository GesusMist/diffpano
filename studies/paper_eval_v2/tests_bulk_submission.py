"""Offline checks of finite scheduler reservations and preserved attempt identities."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .common import read, write
from . import bulk_submission as b


class BulkSubmissionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.run = Path(temp.name)
        self.plan = dict(profile_indices=[0, 1, 2], rows=[dict(index=i, case_id="case%d" % i,
            selected=True, method_id=("spherope", "sphere20", "select")[i % 3]) for i in range(72)])
        write(self.run / "generation/manifest.json", self.plan)

    def audit(self, scheduler, completed=None):
        return b.audit(self.run, self.plan, scheduler, set(completed or ("case0", "case1", "case2")),
                       {"spherope", "sphere20", "select"})

    def test_explicitly_rejected_submission_does_not_count_as_a_scientific_attempt(self):
        result = self.audit(dict(jobs={}, attempts=[dict(key="confirmation-bulk", state="rejected")]))
        self.assertEqual(result["attempted_indices"], [])
        self.assertEqual(len(result["pending_indices"]), 69)
        self.assertEqual(result["submission_blockers"], [])

    def test_ambiguous_submission_blocks_additional_generation(self):
        result = self.audit(dict(jobs={}, attempts=[dict(key="confirmation-bulk", state="ambiguous_submission")]))
        self.assertEqual(len(result["submission_blockers"]), 1)

    def test_submitted_but_failed_or_missing_images_are_never_in_new_pending_indices(self):
        jobs = {"confirmation-bulk-a": dict(command=["sbatch", "--array=3-8%2"], scheduler_state="FAILED")}
        result = self.audit(dict(jobs=jobs), completed=["case%d" % i for i in range(6)])
        self.assertEqual(result["failed_case_ids"], ["case6", "case7", "case8"])
        self.assertEqual(result["pending_indices"][0], 9)

    def test_unjournaled_scientific_attempt_is_not_retried(self):
        write(self.run / "generation/cases/case3/status.json", dict(state="running"))
        result = self.audit(dict(jobs={}))
        self.assertEqual(result["failed_case_ids"], ["case3"])
        self.assertNotIn(3, result["pending_indices"])

    def test_batch_size_tracks_full_scheduler_walltime_plus_next_stage_reserve(self):
        with patch("studies.paper_eval_v2.resources.submit", return_value=dict(job_id="123")) as submit:
            result = b.launch(self.run, self.plan, list(range(3, 72)), 1800)
        self.assertEqual(result["indices"], [3, 4])
        self.assertEqual(submit.call_args.kwargs["reserve_su"], 2 * 584 + 320)
        self.assertEqual(submit.call_args.kwargs["options"][0], "--array=3,4%2")
        journal = read(next((self.run / "controller/generation_batches").glob("*.json")))
        self.assertEqual(journal["job_id"], "123")
        self.assertEqual(journal["walltime_hours"], 4)

    def test_large_balance_still_caps_one_batch_at_six_images(self):
        with patch("studies.paper_eval_v2.resources.submit", return_value=dict(job_id="123")):
            result = b.launch(self.run, self.plan, list(range(3, 72)), 100000)
        self.assertEqual(result["indices"], list(range(3, 9)))

    def test_insufficient_full_task_reservation_does_not_submit(self):
        with patch("studies.paper_eval_v2.resources.submit") as submit:
            result = b.launch(self.run, self.plan, [3], 903)
        self.assertEqual(result["state"], "insufficient_scheduler_reservation")
        submit.assert_not_called()

    def test_journal_mismatch_blocks_further_generation(self):
        with patch("studies.paper_eval_v2.resources.submit", return_value=dict(job_id="123")):
            result = b.launch(self.run, self.plan, list(range(3, 72)), 5000)
        key = "confirmation-bulk-" + result["batch_id"]
        jobs = {key: dict(command=["sbatch", "--array=9-14%2"], scheduler_state="COMPLETED")}
        audited = self.audit(dict(jobs=jobs))
        self.assertIn("differs", audited["submission_blockers"][0])


if __name__ == "__main__":
    unittest.main()
