"""Offline orchestration tests; no scheduler, GPU, network, or paid requests."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .common import write
from . import controller as c


class ControllerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.run = Path(temp.name)
        self.plan = dict(tier="core", selected_indices=list(range(72)), profile_indices=[0, 1, 2], rows=[dict(
            index=i, case_id="case%d" % i, method_id=("spherope", "sphere20", "select")[i % 3], selected=True) for i in range(72)])
        write(self.run / "generation/manifest.json", self.plan)
        self.state = dict(state="running")

    def job(self, key, role, state="RUNNING"):
        result = dict(job_id=key, role=role, scheduler_state=state)
        if role == "generation" and key == "bulk":
            result["command"] = ["sbatch", "--array=3-71%2"]
        return result

    def poll(self, complete, jobs, extra=None):
        queue = [dict(job_id=job["job_id"]) for job in jobs.values() if not c.terminal(job)]
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("studies.paper_eval_v2.resources.reconcile", return_value=dict(jobs=jobs, queue_snapshot=queue)))
        stack.enter_context(patch("studies.paper_eval_v2.generation.gate", return_value=self.plan))
        stack.enter_context(patch.object(c, "generation_snapshot", return_value=set(complete)))
        submit = stack.enter_context(patch("studies.paper_eval_v2.resources.submit", return_value=dict(job_id="new-job")))
        stack.enter_context(patch("studies.paper_eval_v2.confirmation_index.index", side_effect=lambda run: write(run / "generation/inventory_records.json", [])))
        if extra:
            extra(stack)
        result = c.poll(self.run, self.state)
        return result, submit

    def test_bulk_requires_three_verified_profiles_and_measured_balance(self):
        def extra(stack):
            stack.enter_context(patch.object(c, "profile_budget", return_value=dict(required_su=500)))
            stack.enter_context(patch("studies.paper_eval_v2.resources.balance", return_value=499))
        result, submit = self.poll(["case0", "case1", "case2"], {"confirmation-profiles": self.job("profile", "generation", "COMPLETED")}, extra)
        self.assertIn("generation_budget", result["blockers"])
        self.assertFalse(any("generation.slurm" in str(call) for call in submit.call_args_list))

    def test_failed_profile_does_not_block_verified_independent_methods(self):
        def extra(stack):
            stack.enter_context(patch.object(c, "profile_budget", return_value=dict(required_su=500)))
            stack.enter_context(patch("studies.paper_eval_v2.resources.balance", return_value=5000))
        result, submit = self.poll(["case1", "case2"], {"confirmation-profiles": self.job("profile", "generation", "FAILED")}, extra)
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(submit.call_args.kwargs["role"], "generation")
        array = submit.call_args.kwargs["options"][0].split("=", 1)[1].split("%")[0]
        indices = [int(value) for value in array.split(",")]
        self.assertEqual(len(indices), 6)
        self.assertTrue(all(self.plan["rows"][i]["method_id"] in ("sphere20", "select") for i in indices))
        self.assertNotIn(0, indices)

    def test_completed_interim_batch_launches_only_next_unattempted_images(self):
        def extra(stack):
            stack.enter_context(patch.object(c, "profile_budget", return_value=dict(required_su=500)))
            stack.enter_context(patch("studies.paper_eval_v2.resources.balance", return_value=5000))
        jobs = {"confirmation-profiles": self.job("profile", "generation", "COMPLETED"),
                "confirmation-bulk-first": dict(self.job("bulk", "generation", "COMPLETED"), command=["sbatch", "--array=3-8%2"])}
        result, submit = self.poll(["case%d" % i for i in range(9)], jobs, extra)
        self.assertEqual(submit.call_args.kwargs["role"], "generation")
        self.assertEqual(submit.call_args.kwargs["options"][0], "--array=9,10,11,12,13,14%2")
        self.assertEqual(result["state"], "running")
        self.assertNotIn("generation", result["blockers"])

    def test_failed_batch_cases_are_preserved_while_untouched_images_continue(self):
        def extra(stack):
            stack.enter_context(patch.object(c, "profile_budget", return_value=dict(required_su=500)))
            stack.enter_context(patch("studies.paper_eval_v2.resources.balance", return_value=5000))
        jobs = {"confirmation-profiles": self.job("profile", "generation", "COMPLETED"),
                "confirmation-bulk-first": dict(self.job("bulk", "generation", "FAILED"), command=["sbatch", "--array=3-8%2"])}
        result, submit = self.poll(["case%d" % i for i in (0, 1, 2, 3, 4, 5, 7, 8)], jobs, extra)
        self.assertEqual(submit.call_args.kwargs["options"][0], "--array=9,10,11,12,13,14%2")
        self.assertIn("case6", result["blockers"]["generation_cases"])

    def test_pending_reference_reserves_the_evaluation_slot(self):
        jobs = {"confirmation-bulk": self.job("bulk", "generation"), "reference": self.job("ref", "evaluation", "PENDING")}
        _, submit = self.poll(["case%d" % i for i in range(6)], jobs)
        submit.assert_not_called()

    def test_six_new_images_launch_one_finite_evaluation(self):
        jobs = {"confirmation-bulk": self.job("bulk", "generation")}
        result, submit = self.poll(["case%d" % i for i in range(6)], jobs)
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(submit.call_args.kwargs["role"], "evaluation")
        self.assertEqual(submit.call_args.kwargs["reserve_su"], 292)
        self.assertEqual(len(next(iter(result["evaluation_batches"].values()))["case_ids"]), 6)

    def test_live_api_blocks_manifest_preparation(self):
        write(self.run / "vlm/views.json", dict(views=[]))
        batch = "batch"
        self.state["evaluation_batches"] = {batch: dict(key="eval", job_id="eval")}
        write(self.run / "controller/evaluations" / (batch + ".json"), dict(state="complete", completed_case_ids=["case%d" % i for i in range(6)]))
        with patch("studies.paper_eval_v2.prepare_production.prepare") as prepare:
            _, submit = self.poll(["case%d" % i for i in range(6)], {"confirmation-bulk": self.job("bulk", "generation"), "api-production-v2": self.job("api", "api")})
            prepare.assert_not_called()
            submit.assert_not_called()

    def test_existing_prompt_blocks_cannot_gain_new_observations(self):
        old = [dict(prompt_id="p", case_id="a", view_id="front", image_sha256="hash", seed=0)]
        c.validate_monotonic_views(old, old + [dict(prompt_id="q", case_id="b", view_id="front", image_sha256="other", seed=0)])
        with self.assertRaisesRegex(ValueError, "existing API prompt block"):
            c.validate_monotonic_views(old, old + [dict(old[0], view_id="back")])
        with self.assertRaisesRegex(ValueError, "existing API prompt block"):
            c.validate_monotonic_views(old, [dict(old[0], image_sha256="changed")])

    def test_failed_evaluation_is_not_resubmitted(self):
        self.state["evaluation_batches"] = {"batch": dict(key="eval", job_id="eval")}
        result, submit = self.poll(["case%d" % i for i in range(6)], {"confirmation-bulk": self.job("bulk", "generation"), "eval": self.job("eval", "evaluation", "FAILED")})
        self.assertIn("evaluation", result["blockers"])
        submit.assert_not_called()

    def test_finished_is_not_inferred_from_submission_or_unknown_accounting(self):
        jobs = {"submitted": self.job("123", "evaluation", "submitted"), "done": self.job("124", "evaluation", "COMPLETED")}
        self.assertEqual(set(c.active_jobs(dict(jobs=jobs, queue_snapshot=[]))), {"submitted"})
        self.assertFalse(c.terminal(dict(scheduler_state="unknown")))

    def test_late_reference_features_change_finalization_and_gate_distribution(self):
        cases = ["case%d" % i for i in range(72)]
        self.state["evaluation_batches"] = {"all": dict(key="eval", job_id="eval")}
        write(self.run / "controller/evaluations/all.json", dict(state="complete", completed_case_ids=cases))
        write(self.run / "vlm/production_result.json", dict(state="complete", valid_requests=1, planned_requests=1))
        jobs = {"confirmation-bulk": self.job("bulk", "generation", "COMPLETED"), "eval": self.job("eval", "evaluation", "COMPLETED")}
        _, first = self.poll(cases, jobs)
        self.assertNotIn("--distribution", first.call_args.kwargs["script_args"])
        first_key = first.call_args.args[1]
        write(self.run / "group2/reference_features.json", dict(complete=True, records=[]))
        _, second = self.poll(cases, jobs)
        self.assertIn("--distribution", second.call_args.kwargs["script_args"])
        self.assertNotEqual(second.call_args.args[1], first_key)

    def test_explicit_resume_preserves_batches_and_blockers_in_a_new_epoch(self):
        from .common import read
        prior = dict(state="deadline_reached", epoch=1, started_utc="2026-10-26T00:00:00+00:00", duration_seconds=100,
                     blockers={"api": "uncertain paid request"}, evaluation_batches={"a": {"key": "done"}}, api_batches={"b": {"key": "paid"}})
        write(self.run / "controller/state.json", prior)
        scheduler = dict(jobs={"continuation-controller-v1": self.job("controller", "cpu", "COMPLETED")}, queue_snapshot=[])
        with patch("studies.paper_eval_v2.resources.reconcile", return_value=scheduler), patch("studies.paper_eval_v2.resources.submit", return_value=dict(job_id="next")) as submit:
            c.launch_controller(self.run, resume=True)
        current = read(self.run / "controller/state.json")
        self.assertEqual(current["epoch"], 2)
        self.assertEqual(current["state"], "running")
        self.assertEqual(current["blockers"], prior["blockers"])
        self.assertEqual(current["evaluation_batches"], prior["evaluation_batches"])
        self.assertEqual(current["api_batches"], prior["api_batches"])
        self.assertEqual(read(self.run / "controller/epochs/epoch-0001-state.json"), prior)
        self.assertEqual(submit.call_args.args[1], "continuation-controller-v2")

    def test_explicit_resume_refuses_an_active_or_pending_controller(self):
        scheduler = dict(jobs={"continuation-controller-v1": self.job("controller", "cpu", "PENDING")}, queue_snapshot=[dict(job_id="controller")])
        with patch("studies.paper_eval_v2.resources.reconcile", return_value=scheduler), patch("studies.paper_eval_v2.resources.submit") as submit:
            with self.assertRaisesRegex(ValueError, "still active"):
                c.launch_controller(self.run, resume=True)
            submit.assert_not_called()

    def test_controller_duration_and_sleep_are_finite(self):
        with self.assertRaises(ValueError):
            c.run_controller(self.run, hours=25)
        with self.assertRaises(ValueError):
            c.run_controller(self.run, interval=61)


if __name__ == "__main__":
    unittest.main()
