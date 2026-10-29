"""Finite, journaled continuation of confirmation generation and evaluation."""
import argparse
from datetime import datetime, timezone
import fcntl
import math
from pathlib import Path
import time
import traceback

from .common import ROOT, RUN, digest, lock, now, read, sha, write

TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE")


def terminal(job):
    states = job.get("scheduler_state", "unknown")
    if isinstance(states, str):
        states = [states]
    return bool(states) and all(any(str(state).startswith(final) for final in TERMINAL) for state in states)


def active_jobs(scheduler):
    live = {row["job_id"].split("_")[0] for row in scheduler.get("queue_snapshot", [])}
    return {key: value for key, value in scheduler.get("jobs", {}).items() if value["job_id"] in live or not terminal(value)}


def writers(jobs):
    return {key: value for key, value in jobs.items() if value["role"] == "evaluation" or any(token in key for token in ("metrics", "distribution", "reference", "report", "finalize", "views"))}


def profile_budget(run, plan, completed, pending_indices=None):
    """Use measured full-process profile time, with a50% margin and one next-stage reserve."""
    from .generation import case_folder
    durations = {}
    details = []
    for index in plan["profile_indices"]:
        row = plan["rows"][index]
        if row["case_id"] not in completed:
            continue
        metadata = read(case_folder(run, row) / "metadata.json")
        seconds = float(metadata.get("process_generation_seconds", metadata["generation_seconds"] + metadata["setup_seconds"]))
        if not 0 < seconds <= 4 * 3600:
            raise ValueError("Profile runtime is missing or outside the four-hour job limit")
        durations[row["method_id"]] = seconds
        details.append(dict(method_id=row["method_id"], seconds=seconds, gpu=metadata.get("gpu"), metadata_sha256=sha(case_folder(run, row) / "metadata.json")))
    if not durations:
        raise ValueError("At least one verified profile is required")
    pending = [row for row in plan["rows"] if row["selected"] and row["case_id"] not in completed and row["method_id"] in durations
               and (pending_indices is None or row["index"] in pending_indices)]
    required = sum(durations[row["method_id"]] * 1.5 * 146 / 3600 for row in pending) + 292 + 24 + 4
    return dict(required_su=required, remaining_cases=len(pending), profiles=details, eligible_methods=sorted(durations),
                policy="1.5x observed process seconds for remaining images, plus one2h evaluation GPU, one12h API CPU and one1h report CPU job")


def generation_snapshot(run, plan):
    from .generation import complete
    return {row["case_id"] for row in plan["rows"] if row["selected"] and complete(run, row)}


def observation_sets(views):
    result = {}
    for row in views:
        result.setdefault(row["prompt_id"], set()).add((row["case_id"], row["view_id"], row["image_sha256"], row.get("seed", 0)))
    return result


def validate_monotonic_views(old_views, new_views):
    old, new = observation_sets(old_views), observation_sets(new_views)
    for prompt_id, identities in old.items():
        if identities != new.get(prompt_id):
            raise ValueError("An existing API prompt block would change: " + prompt_id)
    # Core prompt blocks must be introduced with all three methods and all14 views.
    for prompt_id in set(new) - set(old):
        rows = [row for row in new_views if row["prompt_id"] == prompt_id]
        if any(row.get("seed", 0) != 0 for row in rows):
            raise ValueError("The autonomous continuation supports core seed0 only")


def planned_api_requests(manifest, frozen):
    from .vlm import request_records
    repeats = {(row["image_sha256"], row["view_id"]) for row in manifest["repeat_view_identities"]}
    return len(request_records(manifest["views"], frozen)) + len(request_records([row for row in manifest["views"] if (row["image_sha256"], row["view_id"]) in repeats], frozen, "repeat-1"))


def record_event(state, kind, **details):
    event = dict(kind=kind, **details)
    signature = digest(event)
    if signature == state.get("last_event_signature"):
        return
    state.setdefault("events", []).append(dict(event, observed_utc=now()))
    state["events"] = state["events"][-200:]
    state["last_event_signature"] = signature
    print("CONTINUATION", kind, details, flush=True)


def poll(run, state):
    from . import bulk_submission, generation, resources
    run = Path(run)
    scheduler = resources.reconcile(run)
    jobs = scheduler.get("jobs", {})
    live = active_jobs(scheduler)
    state["scheduler_snapshot_utc"] = now()
    state["active_job_keys"] = sorted(live)
    state.setdefault("blockers", {})
    state.setdefault("evaluation_batches", {})
    state.setdefault("api_batches", {})
    state.setdefault("report_batches", {})
    # Recover an interrupted controller after sbatch: the resource journal is authoritative.
    for path in (run / "controller/batches").glob("*.json"):
        batch = read(path)
        key = "confirmation-eval-" + batch["batch_id"]
        if key in jobs:
            state["evaluation_batches"].setdefault(batch["batch_id"], dict(key=key, job_id=jobs[key]["job_id"], case_ids=batch["case_ids"]))
    if not (run / "generation/manifest.json").exists():
        state["phase"] = "waiting_for_frozen_generation_manifest"
        return state
    plan = generation.gate(run)
    if plan["tier"] != "core" or len(plan["selected_indices"]) != 72:
        raise ValueError("Autonomous controller is limited to the frozen72-image core tier")
    completed = generation_snapshot(run, plan)
    state["verified_generation_count"] = len(completed)
    state["verified_case_ids"] = sorted(completed)
    profile_ids = {plan["rows"][i]["case_id"] for i in plan["profile_indices"]}
    live_generation = any(job["role"] == "generation" for job in live.values())
    eligible_methods = {plan["rows"][i]["method_id"] for i in plan["profile_indices"] if plan["rows"][i]["case_id"] in completed}
    profile_job = jobs.get("confirmation-profiles")
    profiles_finished = bool(profile_job and terminal(profile_job)) or profile_ids <= completed
    generation_audit = bulk_submission.audit(run, plan, scheduler, completed, eligible_methods)
    state["generation_batches"] = generation_audit
    pending_indices = generation_audit["pending_indices"]
    if generation_audit["submission_blockers"]:
        state["blockers"]["generation_submission"] = "; ".join(generation_audit["submission_blockers"])
    if generation_audit["failed_case_ids"]:
        state["blockers"]["generation_cases"] = "Submitted images ended without verified outputs; no automatic retry: " + ",".join(generation_audit["failed_case_ids"])
    if pending_indices and profiles_finished and not any(key in state["blockers"] for key in ("generation_budget", "generation_submission")):
        if not live_generation:
            evidence = profile_budget(run, plan, completed, pending_indices=pending_indices)
            evidence["available_su"] = resources.balance()
            write(run / "controller/profile_budget.json", evidence)
            if evidence["available_su"] < evidence["required_su"]:
                state["blockers"]["generation_budget"] = "Insufficient allocation for measured remaining independent methods and next evaluation stages"
            else:
                result = bulk_submission.launch(run, plan, pending_indices, evidence["available_su"])
                if result.get("state") == "insufficient_scheduler_reservation":
                    state["blockers"]["generation_budget"] = "Insufficient allocation for one four-hour generation task and the next evaluation stages"
                else:
                    state["bulk_job"] = result
                    record_event(state, "bulk_submitted", job_id=result.get("job_id"), indices=result["indices"],
                                 eligible_methods=sorted(eligible_methods), measured_remaining_su=evidence["required_su"],
                                 scheduler_batch_reservation_su=result["reservation_su"])
                    return state
    if not profile_ids <= completed and profiles_finished:
        state["blockers"]["generation"] = "Missing profile outputs are not retried; independent verified methods may continue: " + ",".join(sorted(eligible_methods))
    generation_stopped = profiles_finished and not live_generation and (not pending_indices or any(
        key in state["blockers"] for key in ("generation_budget", "generation_submission")))
    evaluated = set()
    for batch_id, batch in state["evaluation_batches"].items():
        path = run / "controller/evaluations" / (batch_id + ".json")
        result = read(path) if path.exists() else {}
        if result.get("state") == "complete":
            evaluated.update(result["completed_case_ids"])
        elif batch["key"] in jobs and terminal(jobs[batch["key"]]):
            state["blockers"]["evaluation"] = "Evaluation job ended incomplete: " + batch["key"] + "; inspect saved stage errors, no automatic scientific retry"
    state["verified_evaluation_count"] = len(evaluated)
    for batch_id, batch in state["report_batches"].items():
        path = run / "controller/finalizations" / (batch_id + ".json")
        if batch["key"] in jobs and terminal(jobs[batch["key"]]) and (not path.exists() or read(path).get("state") != "complete"):
            state["blockers"]["report"] = "Report/finalization worker ended incomplete: " + batch["key"]
    new = completed - evaluated
    busy_writers = writers(live)
    if not busy_writers and "evaluation" not in state["blockers"] and (len(new) >= 6 or (generation_stopped or len(completed) == 72) and new):
        from .confirmation_index import index
        index(run)
        batch_id = digest(sorted(new))[:20]
        key = "confirmation-eval-" + batch_id
        batch_path = run / "controller/batches" / (batch_id + ".json")
        write(batch_path, dict(batch_id=batch_id, case_ids=sorted(new), inventory_records_sha256=sha(run / "generation/inventory_records.json")))
        result = resources.submit(run, key, ROOT / "studies/paper_eval_v2/confirmation_worker.slurm", role="evaluation",
            script_args=("--run", str(run), "--batch", str(batch_path)), reserve_su=292)
        state["evaluation_batches"][batch_id] = dict(key=key, job_id=result["job_id"], case_ids=sorted(new))
        record_event(state, "evaluation_submitted", job_id=result["job_id"], new_images=len(new))
        return state
    # Preparation and submission are protected by the same lock as all paid workers.
    live_api = any(job["role"] == "api" for job in live.values())
    production_path = run / "vlm/production_result.json"
    production = read(production_path) if production_path.exists() else {}
    for batch_id, batch in state["api_batches"].items():
        job = jobs.get(batch["key"])
        if job and terminal(job) and batch_id == state.get("latest_api_batch"):
            if production.get("state") != "complete" or production.get("valid_requests") != batch["planned_requests"]:
                state["blockers"]["api"] = "API continuation incomplete; saved uncertain/failed requests require reconciliation: " + batch["key"]
    dev_api_jobs = [job for key, job in jobs.items() if job["role"] == "api" and key.startswith("api-production")]
    if dev_api_jobs and all(terminal(job) for job in dev_api_jobs) and production.get("state") == "blocked":
        state["blockers"]["api"] = "Saved development API result is blocked: " + str(production.get("blocker"))
    if not live_api and not busy_writers and "api" not in state["blockers"] and (run / "vlm/views.json").exists() and evaluated:
        from .vlm import worker_lock, select_cohort_views
        from .prepare_production import prepare
        try:
            with worker_lock(run):
                view_value = read(run / "vlm/views.json")
                available_views = view_value["views"] if isinstance(view_value, dict) else view_value
                proposed, _ = select_cohort_views(available_views, read(run / "cohorts.json"))
                old_path = run / "vlm/production_views.json"
                old = read(old_path) if old_path.exists() else dict(views=[])
                validate_monotonic_views(old["views"], proposed)
                changed = observation_sets(proposed) != observation_sets(old["views"])
                if changed:
                    prepare(run)
                manifest = read(old_path)
                batch_id = digest(observation_sets_serializable(manifest["views"]))[:20]
                key = "api-confirmation-" + batch_id
                expected = planned_api_requests(manifest, read(run / "vlm/pilot_protocol.json"))
                needs_execution = changed or production.get("state") != "complete" or production.get("valid_requests") != expected
                if needs_execution:
                    if key in jobs and terminal(jobs[key]):
                        state["blockers"]["api"] = "An API job for this exact manifest already ended incomplete: " + key
                    else:
                        state["api_batches"][batch_id] = dict(key=key, planned_requests=expected, production_manifest_sha256=sha(old_path))
                        state["latest_api_batch"] = batch_id
                        write(run / "controller/state.json", state)
                        result = resources.submit(run, key, ROOT / "studies/paper_eval_v2/api.slurm", role="api",
                            script_args=("production", "--run", str(run), "--budget-mode", "provider_managed"), reserve_su=24)
                        state["api_batches"][batch_id]["job_id"] = result["job_id"]
                        record_event(state, "api_submitted", job_id=result["job_id"], total_requests=expected)
                        return state
        except BlockingIOError:
            state["phase"] = "waiting_for_private_api_worker_lock"
            return state
    all_features = len(evaluated) == 72
    reference_path = run / "group2/reference_features.json"
    reference_ready = reference_path.exists() and read(reference_path).get("complete") is True
    run_distribution = all_features and reference_ready and "distribution" not in state["blockers"]
    api_complete = not live_api and production.get("state") == "complete" and production.get("valid_requests") == production.get("planned_requests")
    if not busy_writers and not live_api and evaluated and "report" not in state["blockers"]:
        signature = digest(dict(evaluated=sorted(evaluated), production_sha256=sha(production_path) if production_path.exists() else None,
                                distribution=run_distribution, reference_features_sha256=sha(reference_path) if reference_path.exists() else None))[:20]
        key = "confirmation-finalize-" + signature
        if signature not in state["report_batches"]:
            args = ["--run", str(run), "--batch-id", signature]
            if run_distribution:
                args.append("--distribution")
            result = resources.submit(run, key, ROOT / "studies/paper_eval_v2/confirmation_finalize.slurm", role="cpu", script_args=args, reserve_su=4)
            state["report_batches"][signature] = dict(key=key, job_id=result["job_id"], final_features=all_features, distribution=run_distribution)
            state["latest_report_batch"] = signature
            record_event(state, "report_submitted", job_id=result["job_id"], confirmation_distribution=run_distribution)
            return state
    final_report = state.get("latest_report_batch")
    report_result_path = run / "controller/finalizations" / (str(final_report) + ".json")
    report_result = read(report_result_path) if report_result_path.exists() else {}
    if report_result.get("distribution") and report_result.get("state") == "complete" and report_result.get("confirmation_distribution", {}).get("state") != "complete":
        state["blockers"]["distribution"] = "Confirmation distribution estimates or uncertainty remain incomplete; see group2/confirmation/status.json"
    if all_features and api_complete and not busy_writers and not live_api and report_result.get("state") == "complete" and report_result.get("confirmation_distribution", {}).get("state") == "complete":
        manifest = read(run / "vlm/production_views.json")
        confirmation = next((row for row in manifest["cohorts"] if row["cohort_id"] == "confirmation24"), {})
        if len(confirmation.get("prompt_ids", [])) == 24:
            state["state"] = "complete"
    elif generation_stopped and not busy_writers and not live_api and state["blockers"] and (not new or "evaluation" in state["blockers"]):
        state["state"] = "stopped_with_explicit_pending_work"
    state["phase"] = "monitoring_verified_outputs"
    return state


def observation_sets_serializable(views):
    return {key: sorted(value) for key, value in observation_sets(views).items()}


def finalize(run, batch_id, distribution=False):
    run = Path(run)
    path = run / "controller/finalizations" / (batch_id + ".json")
    result = dict(state="running", started_utc=now(), distribution=distribution)
    write(path, result)
    try:
        if (run / "vlm/production_result.json").exists():
            from .repeat_analysis import evaluate
            result["repeat_analysis"] = evaluate(run)
        if distribution:
            from .confirmation_distribution import evaluate
            result["confirmation_distribution"] = evaluate(run, max_bootstrap_seconds=300)
        from .report import report
        result["report"] = report(run)
        result.update(state="complete", completed_utc=now())
    except Exception:
        result.update(state="failed", error=traceback.format_exc(), completed_utc=now())
        write(path, result)
        raise
    write(path, result)
    return result


def run_controller(run=RUN, hours=24., interval=60.):
    if not 0 < hours <= 24 or not 1 <= interval <= 60:
        raise ValueError("Finite duration must be at most24 hours; poll interval must be1–60 seconds")
    run = Path(run)
    directory = run / "controller"
    directory.mkdir(exist_ok=True)
    with (directory / "process.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / "state.json"
        state = read(path) if path.exists() else dict(state="running", epoch=1, started_utc=now(), duration_seconds=max(1, hours * 3600 - 120))
        started = datetime.fromisoformat(state["started_utc"]).timestamp()
        deadline = started + state["duration_seconds"]
        while time.time() < deadline and state["state"] == "running":
            try:
                state = poll(run, state)
                state["consecutive_controller_errors"] = 0
            except Exception:
                error = traceback.format_exc()
                state["consecutive_controller_errors"] = state.get("consecutive_controller_errors", 0) + 1
                record_event(state, "controller_error", error=error)
                if state["consecutive_controller_errors"] >= 3:
                    state.update(state="stopped_with_controller_error", error=error)
            state["updated_utc"] = now()
            write(path, state)
            if state["state"] == "running":
                time.sleep(min(interval, max(0, deadline - time.time())))
        if state["state"] == "running":
            state.update(state="deadline_reached", updated_utc=now(), reason="Finite controller ended; submitted finite jobs remain recorded and are not claimed complete")
            write(path, state)
        return state


def launch_controller(run=RUN, hours=24., interval=60., resume=False):
    """Explicitly start one new finite epoch; preserve every scientific blocker and batch."""
    from . import resources
    if not 0 < hours <= 24 or not 1 <= interval <= 60:
        raise ValueError("Finite duration must be at most24 hours; poll interval must be1–60 seconds")
    run = Path(run)
    directory = run / "controller"
    directory.mkdir(exist_ok=True)
    with lock(directory / "launch.lock"):
        scheduler = resources.reconcile(run)
        controllers = {key: job for key, job in scheduler["jobs"].items() if key.startswith("continuation-controller-v")}
        active = {key: job for key, job in active_jobs(scheduler).items() if key in controllers}
        if resume and active:
            raise ValueError("An earlier continuation controller is still active or pending; do not overlap controller epochs")
        epoch = 1
        if resume:
            path = directory / "state.json"
            if not path.exists():
                raise ValueError("No previous controller state exists; use launch for the first epoch")
            with (directory / "process.lock").open("a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                state = read(path)
                if state.get("state") == "complete":
                    raise ValueError("Verified continuation is already complete")
                old_epoch = int(state.get("epoch", 1))
                epoch = max([old_epoch] + [int(key.rsplit("v", 1)[1]) for key in controllers]) + 1
                archive = directory / "epochs" / ("epoch-%04d-state.json" % old_epoch)
                if archive.exists() and read(archive) != state:
                    archive = directory / "epochs" / ("epoch-%04d-state-%s.json" % (old_epoch, digest(state)[:12]))
                write(archive, state)
                previous = dict(epoch=old_epoch, state=state["state"], started_utc=state["started_utc"],
                                archived_state=str(archive), archived_state_sha256=sha(archive))
                state.setdefault("epoch_history", []).append(previous)
                state.update(state="running", epoch=epoch, started_utc=now(), duration_seconds=max(1, hours * 3600 - 120),
                             consecutive_controller_errors=0, phase="explicit_finite_resume")
                state.pop("error", None)
                state.pop("reason", None)
                write(path, state)
        return resources.submit(run, "continuation-controller-v" + str(epoch), ROOT / "studies/paper_eval_v2/controller.slurm",
            role="cpu", reserve_su=hours, options=("--time=%d" % max(1, math.ceil(hours * 60)),),
            script_args=("--run", str(run), "--hours", str(hours), "--interval", str(interval)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "launch", "resume", "finalize"))
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--hours", type=float, default=24.)
    parser.add_argument("--interval", type=float, default=60.)
    parser.add_argument("--batch-id")
    parser.add_argument("--distribution", action="store_true")
    args = parser.parse_args()
    if args.action in ("launch", "resume"):
        print(launch_controller(args.run, args.hours, args.interval, resume=args.action == "resume"))
    elif args.action == "finalize":
        if not args.batch_id or not args.batch_id.isalnum():
            raise ValueError("A hexadecimal finalization batch ID is required")
        print(finalize(args.run, args.batch_id, args.distribution))
    else:
        print(run_controller(args.run, args.hours, args.interval))


if __name__ == "__main__":
    main()
