"""Submission guard tests use a temporary directory and never call Slurm."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from . import launch


class LaunchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.out = self.root / "recovery-v1"
        self.out.mkdir()
        self.plan = self.out / "manifest.json"
        self.plan.write_text("{}")
        for name, value in (("OUT", self.out), ("ROOT", self.root), ("PLAN", self.plan)):
            patcher = patch.object(launch, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(launch, "gate", return_value={"increased_walltime_minutes": 240})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.calls = []

    def query(self, command, **kwargs):
        if command[0] == "squeue":
            return "other-study\n"
        return "|142684268708|2027|Y|20000.00|-11592.69|8407.31|PI|\n"

    def submit(self, command, **kwargs):
        journal = json.loads((self.out / "launch.json").read_text())
        intent = journal["attempts"][-1]
        self.assertEqual(intent["state"], "submission_intent")
        self.assertEqual(intent["command"], command)
        self.calls.append(command)
        return subprocess.CompletedProcess(command, 0, str(200 + len(self.calls)) + "\n", "")

    def test_dry_run_is_offline_and_read_only(self):
        result = launch.launch(query=lambda *_a, **_k: self.fail("Unexpected query"),
                               runner=lambda *_a, **_k: self.fail("Unexpected submission"))
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["generation_cases"], 10)
        self.assertEqual(result["generation_max_su"], 5840)
        self.assertAlmostEqual(result["total_max_su"], 5845.333333333333)
        self.assertEqual({path.name for path in self.out.iterdir()}, {"manifest.json"})

    def test_exact_retry_array_report_dependency_and_duplicate_guard(self):
        result = launch.launch(execute=True, runner=self.submit, query=self.query)
        self.assertEqual(len(self.calls), 2)
        self.assertIn("--array=24-27,42-47%2", self.calls[0])
        self.assertIn("--exclude=fc026", self.calls[0])
        self.assertIn("--job-name=refinement28-resume", self.calls[0])
        self.assertIn("--dependency=afterany:201", self.calls[1])
        self.assertIn("--kill-on-invalid-dep=yes", self.calls[1])
        self.assertEqual(result["state"], "submitted")
        with self.assertRaisesRegex(RuntimeError, "journal already exists"):
            launch.launch(execute=True, runner=self.submit, query=self.query)
        self.assertEqual(len(self.calls), 2)

    def test_ambiguous_submit_is_journaled_and_never_repeated(self):
        def ambiguous(command, **kwargs):
            self.calls.append(command)
            raise TimeoutError("Response unavailable")
        with self.assertRaises(TimeoutError):
            launch.launch(execute=True, runner=ambiguous, query=self.query)
        journal = json.loads((self.out / "launch.json").read_text())
        self.assertEqual(journal["state"], "ambiguous_submission")
        with self.assertRaisesRegex(RuntimeError, "journal already exists"):
            launch.launch(execute=True, runner=self.submit, query=self.query)
        self.assertEqual(len(self.calls), 1)

    def test_rejected_submit_stops_dependent_submission(self):
        def rejected(command, **kwargs):
            self.calls.append(command)
            return subprocess.CompletedProcess(command, 1, "", "Rejected")
        with self.assertRaisesRegex(RuntimeError, "Submission requires inspection"):
            launch.launch(execute=True, runner=rejected, query=self.query)
        self.assertEqual(len(self.calls), 1)
        journal = json.loads((self.out / "launch.json").read_text())
        self.assertEqual(journal["state"], "ambiguous_submission")

    def test_balance_and_budget_guard_before_submission(self):
        with self.assertRaises(ValueError):
            launch.launch(execute=True, max_su=5840, runner=self.submit, query=self.query)
        def low_balance(command, **kwargs):
            return "" if command[0] == "squeue" else "|142684268708|2027|Y|20000|-15000|5000|PI|"
        with self.assertRaisesRegex(RuntimeError, "balance"):
            launch.launch(execute=True, runner=self.submit, query=low_balance)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.out / "launch.json").exists())

    def test_existing_original_or_retry_jobs_prevent_submission(self):
        for name in ("refinement28-bulk", "refinement28-resume", "refinement28-report"):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "Existing refinement28 jobs"):
                launch.launch(execute=True, runner=self.submit, query=lambda *_a, **_k: name + "\n")
        self.assertEqual(self.calls, [])

    def test_source_gate_or_scope_failure_prevents_submission(self):
        with patch.object(launch, "gate", side_effect=ValueError("Changed source")):
            with self.assertRaisesRegex(ValueError, "Changed source"):
                launch.launch(execute=True, runner=self.submit, query=self.query)
        with patch.object(launch, "INDICES", (0, 1)):
            with self.assertRaisesRegex(ValueError, "ten inspected"):
                launch.launch(execute=True, runner=self.submit, query=self.query)
        with patch.object(launch, "gate", return_value={"increased_walltime_minutes": 180}):
            with self.assertRaisesRegex(ValueError, "four-hour"):
                launch.launch(execute=True, runner=self.submit, query=self.query)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
