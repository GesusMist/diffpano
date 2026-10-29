"""Explicit finite resume of verified unfinished native-reference groups."""
import argparse
from pathlib import Path
import re

from .common import ROOT, RUN, digest, lock, now, read, sha, write

TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE")


def terminal(job):
    states = job.get("scheduler_state", "unknown")
    if isinstance(states, str):
        states = [states]
    return bool(states) and all(any(str(value).startswith(end) for end in TERMINAL) for value in states)


def generation_conflicts(scheduler):
    """Pending generation also reserves the shared two-worker allowance."""
    jobs = scheduler.get("jobs", {})
    known = {job["job_id"]: (key, job) for key, job in jobs.items()}
    live = scheduler.get("queue_snapshot", [])
    live_ids = {row["job_id"].split("_")[0] for row in live}
    conflicts = [dict(key=key, job_id=job["job_id"], reason="active_or_unreconciled_generation")
                 for key, job in jobs.items() if job["role"] == "generation" and (job["job_id"] in live_ids or not terminal(job))]
    for row in live:
        base = row["job_id"].split("_")[0]
        if base not in known and "gpu" in row.get("resources", "").lower():
            conflicts.append(dict(key=row.get("name", "external"), job_id=row["job_id"], reason="untracked_gpu_allocation"))
    for attempt in scheduler.get("attempts", []):
        if attempt.get("role") == "generation" and attempt.get("state") in ("submission_intent", "ambiguous_submission"):
            conflicts.append(dict(key=attempt["key"], job_id=attempt.get("job_id"), reason="unreconciled_submission_intent"))
    return conflicts


def unfinished_groups(run, manifest):
    from .native import complete
    rows = manifest["rows"]
    groups = manifest["generation_groups"]
    if manifest["expected"] != 240 or len(rows) != 240 or any(row["index"] != index for index, row in enumerate(rows)):
        raise ValueError("Frozen native manifest must contain the original240 indexed images")
    if len(groups) != 8 or any(not group or len(group) > 30 for group in groups) or sorted(index for group in groups for index in group) != list(range(240)):
        raise ValueError("Native generation groups are not the exact240-image partition")
    pending = [index for index, row in enumerate(rows) if not complete(run, row)]
    pending_set = set(pending)
    selected = [index for index, group in enumerate(groups) if pending_set.intersection(group)]
    return selected, pending


def next_version(scheduler):
    keys = set(scheduler.get("jobs", {})) | {row["key"] for row in scheduler.get("attempts", [])}
    versions = [int(match.group(1)) for key in keys if (match := re.fullmatch(r"native-resume-v([0-9]+)", key))]
    return max(versions, default=0) + 1


def resume(run=RUN, execute=False):
    from . import native, resources
    run = Path(run)
    with lock(run / "native/resume.lock"):
        manifest = native.frozen(run)
        groups, pending = unfinished_groups(run, manifest)
        if not groups:
            return dict(state="complete", verified_images=240, pending_images=0, groups=[], submitted=False)
        scheduler = resources.reconcile(run)
        conflicts = generation_conflicts(scheduler)
        reserve = len(groups) * 146.
        available = resources.balance()
        retained_versions = [int(match.group(1)) for path in (run / "native/resumes").glob("native-resume-v*.json")
                             if (match := re.fullmatch(r"native-resume-v([0-9]+)\.json", path.name))]
        version = max([next_version(scheduler)] + [value + 1 for value in retained_versions])
        key = "native-resume-v" + str(version)
        plan = dict(state="ready" if not conflicts and available >= reserve else "blocked", submitted=False,
            key=key, manifest_sha256=sha(run / "native/manifest.json"), helper_source_sha256=sha(__file__),
            groups=groups, unfinished_indices=pending, pending_images=len(pending), verified_images=240-len(pending),
            reservation_su=reserve, available_su=available, conflicting_jobs=conflicts,
            array=",".join(map(str, groups)) + "%2", walltime_per_group="01:00:00",
            resume_policy="Frozen native.execute rechecks each image, skips valid completed outputs, and archives prior incomplete attempts before retry.")
        if not execute:
            return dict(plan, dry_run=True)
        if conflicts:
            raise RuntimeError("Conflicting generation allocation or ambiguous submission; reconcile or wait before native resume: " + str(conflicts))
        if available < reserve:
            raise RuntimeError("Insufficient allocation for the finite native resume: %.2f available, %.2f reserved SU" % (available, reserve))
        path = run / "native/resumes" / (key + ".json")
        if path.exists():
            # A crash before the shared sbatch intent should not silently overwrite provenance.
            previous = read(path)
            if previous.get("state") not in ("submission_planned", "ready"):
                raise RuntimeError("Existing native resume record must be reconciled: " + str(path))
            comparable = {k: v for k, v in previous.items() if k not in ("state", "created_utc")}
            proposed = {k: v for k, v in plan.items() if k not in ("state", "created_utc")}
            if comparable != proposed:
                raise RuntimeError("Pending native resume plan changed before submission; reconcile: " + str(path))
        write(path, dict(plan, state="submission_planned", created_utc=now()))
        result = resources.submit(run, key, ROOT / "studies/paper_eval_v2/native.slurm", role="generation",
            options=("--array=" + plan["array"], "--exclude=fc026"), script_args=("--run", str(run)), reserve_su=reserve)
        record = dict(plan, state="submitted", submitted=True, job_id=result["job_id"], submitted_utc=now(),
                      output_verification="pending", scheduler_record=result)
        write(path, record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--dry-run", action="store_true", help="Only report verified unfinished groups and conflicts; default submits the finite resume")
    args = parser.parse_args()
    import json
    print(json.dumps(resume(args.run, execute=not args.dry_run), indent=2))


if __name__ == "__main__":
    main()
