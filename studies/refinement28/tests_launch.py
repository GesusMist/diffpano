"""Mock-only submission safety tests; no scheduler jobs are created."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from . import launch
from . import report


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.out = self.root / "output"
        self.out.mkdir()
        (self.out / "manifest.json").write_text("{}")
        self.manifest = dict(source_sha256="source-hash", rows=[dict(index=i, prompt="prompt-%02d" % (i // 2),
                             refinement_steps=5 if i % 2 == 0 else 8) for i in range(48)])
        for name, value in (("OUT", self.out), ("ROOT", self.root)):
            p = patch.object(launch, name, value)
            p.start()
            self.addCleanup(p.stop)
        for name, value in (("freeze_check", self.manifest), ("validation_gate", True)):
            p = patch.object(launch, name, return_value=value)
            p.start()
            self.addCleanup(p.stop)
        self.calls = []

    def query(self, args, **kwargs):
        if args[0] == "squeue":
            return "other-study\n"
        return "|142684268708|2027|Y|20000.00|-5880.47|14119.53|PI|\n"

    def submit(self, command, **kwargs):
        journal = json.loads((self.out / "launch.json").read_text())
        self.assertEqual(journal["attempts"][-1]["state"], "submission_intent")
        self.assertEqual(journal["attempts"][-1]["command"], command)
        self.calls.append(command)
        return subprocess.CompletedProcess(command, 0, str(100 + len(self.calls)) + "\n", "")

    def test_dry_run_is_offline_and_does_not_create_journal(self):
        result = launch.launch(query=lambda *_a, **_k: self.fail("Scheduler query during dry-run"),
                               runner=lambda *_a, **_k: self.fail("Submission during dry-run"))
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["generation_cases"], 48)
        self.assertEqual(result["generation_max_su"], 10512)
        self.assertEqual({p.name for p in self.out.iterdir()}, {"manifest.json"})

    def test_dependencies_and_two_job_concurrency(self):
        result = launch.launch(execute=True, runner=self.submit, query=self.query)
        self.assertEqual(len(self.calls), 3)
        self.assertIn("--array=0,1%2", self.calls[0])
        self.assertIn("--array=2-47%2", self.calls[1])
        self.assertIn("--dependency=afterok:101", self.calls[1])
        self.assertIn("--dependency=afterany:101:102", self.calls[2])
        self.assertTrue(all("--kill-on-invalid-dep=yes" in command for command in self.calls[1:]))
        self.assertEqual(result["state"], "submitted")
        with self.assertRaisesRegex(RuntimeError, "journal already exists"):
            launch.launch(execute=True, runner=self.submit, query=self.query)
        self.assertEqual(len(self.calls), 3)

    def test_ambiguous_submission_is_persisted_and_never_repeated(self):
        def ambiguous(*args, **kwargs):
            self.calls.append(args)
            raise TimeoutError("Mock lost sbatch response")
        with self.assertRaises(TimeoutError):
            launch.launch(execute=True, runner=ambiguous, query=self.query)
        journal = json.loads((self.out / "launch.json").read_text())
        self.assertEqual(journal["state"], "ambiguous_submission")
        with self.assertRaises(RuntimeError):
            launch.launch(execute=True, runner=self.submit, query=self.query)
        self.assertEqual(len(self.calls), 1)

    def test_nonzero_sbatch_result_stops_later_submissions(self):
        def failure(command, **kwargs):
            self.calls.append(command)
            return subprocess.CompletedProcess(command, 1, "", "Mock rejection")
        with self.assertRaises(RuntimeError):
            launch.launch(execute=True, runner=failure, query=self.query)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(json.loads((self.out / "launch.json").read_text())["state"], "ambiguous_submission")

    def test_budget_and_available_balance_fail_before_submission(self):
        with self.assertRaises(ValueError):
            launch.launch(execute=True, max_su=10512, runner=self.submit, query=self.query)
        def low_balance(args, **kwargs):
            return "" if args[0] == "squeue" else "|142684268708|2027|Y|20000|-15000|5000|PI|"
        with self.assertRaises(RuntimeError):
            launch.launch(execute=True, runner=self.submit, query=low_balance)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.out / "launch.json").exists())

    def test_live_study_jobs_prevent_duplicate_launch(self):
        with self.assertRaisesRegex(RuntimeError, "Existing refinement28 jobs"):
            launch.launch(execute=True, runner=self.submit, query=lambda *_a, **_k: "refinement28-bulk\n")
        self.assertEqual(self.calls, [])

    def test_manifest_scope_and_validation_fail_closed(self):
        self.manifest["rows"][1]["refinement_steps"] = 7
        with self.assertRaises(ValueError):
            launch.launch(execute=True, runner=self.submit, query=self.query)
        self.manifest["rows"][1]["refinement_steps"] = 8
        with patch.object(launch, "validation_gate", side_effect=ValueError("Validation failed")):
            with self.assertRaises(ValueError):
                launch.launch(execute=True, runner=self.submit, query=self.query)
        self.assertEqual(self.calls, [])


class ReportTests(unittest.TestCase):
    def test_only_complete_pairs_get_comparison_panels(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = [dict(index=i, case_id="case-%02d" % i, prompt="prompt-%02d" % (i // 2),
                         refinement_steps=5 if i % 2 == 0 else 8) for i in range(48)]
            manifest = dict(rows=rows, source_sha256="source")
            for index in (0, 1, 2):
                folder = output / ("case-%02d" % index)
                folder.mkdir()
                (folder / "status.json").write_text('{"state":"complete"}')
                (folder / "metadata.json").write_text('{"generation_seconds":100,"peak_allocated_gib":30}')
            with patch.object(report, "OUT", output), patch.object(report, "freeze_check", return_value=manifest), \
                    patch.object(report, "folder", side_effect=lambda row: output / row["case_id"]), \
                    patch.object(report, "complete", side_effect=lambda row, **kwargs: row["index"] in (0, 1, 2)), \
                    patch.object(report, "sha", return_value="checksum"), \
                    patch.object(report, "make_panel", return_value={"path": "one-panel"}) as panels:
                summary = report.generate_report()
            self.assertEqual(summary["expected_cases"], 48)
            self.assertEqual(summary["verified_complete_cases"], 3)
            self.assertEqual(summary["complete_pairs"], 1)
            self.assertEqual(summary["state"], "incomplete")
            self.assertEqual(panels.call_count, 1)
            self.assertEqual(len((output / "completion.csv").read_text().splitlines()), 49)

    def test_original_erp_pixels_and_dimensions_are_preserved_in_panel(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left_path, right_path = root / "left.png", root / "right.png"
            Image.new("RGB", (4096, 2048), (3, 7, 11)).save(left_path)
            Image.new("RGB", (4096, 2048), (17, 19, 23)).save(right_path)
            before = [path.read_bytes() for path in (left_path, right_path)]
            left = dict(image_path=str(left_path), image_sha256="left", prompt="synthetic")
            right = dict(image_path=str(right_path), image_sha256="right", prompt="synthetic")
            result = report.make_panel(left, right, root / "panel.png")
            self.assertEqual(result["image_dimensions"], [4096, 2048])
            self.assertEqual(result["interpolation"], "none")
            with Image.open(root / "panel.png") as panel:
                self.assertEqual(panel.size, (8264, 2184))
                self.assertEqual(panel.getpixel((24 + 4095, 112 + 2047)), (3, 7, 11))
                self.assertEqual(panel.getpixel((4096 + 48, 112)), (17, 19, 23))
            self.assertEqual(before, [path.read_bytes() for path in (left_path, right_path)])


if __name__ == "__main__":
    unittest.main()
