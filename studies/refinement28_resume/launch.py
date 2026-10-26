"""Submit only the ten unfinished refinement cases with four-hour limits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import math
import re
import subprocess

from studies.refinement28.launch import account_balance
from studies.refinement28.common import ACCOUNT, ROOT, sha, write
from .common import INDICES, OUT, PLAN, gate

PREFIX = "refinement28-resume"
HOURLY_GENERATION_SU = 146.0
GENERATION_WALL_HOURS = 4.0
GENERATION_MAX_SU = 10 * HOURLY_GENERATION_SU * GENERATION_WALL_HOURS
REPORT_MAX_SU = 4 * (80 / 60)
TOTAL_MAX_SU = GENERATION_MAX_SU + REPORT_MAX_SU
DEFAULT_MAX_SU = math.ceil(TOTAL_MAX_SU)
EXPECTED_INDICES = (24, 25, 26, 27, 42, 43, 44, 45, 46, 47)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def estimate():
    return dict(generation_cases=10, retry_indices=list(INDICES),
                maximum_concurrent_gpu_jobs=2, gpu="NVIDIA A100 40GB", cpus=4,
                host_memory="64G", walltime="04:00:00", excluded_nodes=["fc026"],
                generation_hourly_su=HOURLY_GENERATION_SU,
                maximum_gpu_hours=10 * GENERATION_WALL_HOURS,
                generation_max_su=GENERATION_MAX_SU, report_max_su=REPORT_MAX_SU,
                total_max_su=TOTAL_MAX_SU,
                note="Four-hour limits replace the original 90-minute limits for exactly ten unfinished cases. Actual SU charges depend on elapsed time; completed case images are retained.")


def submission_commands(retry_job="<retry_job_id>"):
    scripts = ROOT / "studies" / "refinement28_resume"
    base = ["sbatch", "--parsable", "--chdir=" + str(ROOT)]
    return {
        "retry": base + ["--job-name=" + PREFIX,
                          "--output=" + str(OUT / "logs" / "generation" / "%x-%A_%a.out"),
                          "--array=24-27,42-47%2", "--exclude=fc026",
                          str(scripts / "generate.slurm")],
        "report": base + ["--job-name=" + PREFIX + "-report",
                           "--output=" + str(OUT / "logs" / "report" / "%x-%j.out"),
                           "--dependency=afterany:" + retry_job, "--kill-on-invalid-dep=yes",
                           str(scripts / "report.slurm")],
    }


def checked_plan():
    if tuple(INDICES) != EXPECTED_INDICES:
        raise ValueError("Only the ten inspected unfinished cases may be retried")
    plan = gate()
    if plan["increased_walltime_minutes"] != 240:
        raise ValueError("Retry manifest must authorize the four-hour walltime")
    return plan


def launch(*, execute=False, max_su=DEFAULT_MAX_SU,
           runner=subprocess.run, query=subprocess.check_output):
    if not math.isfinite(max_su) or max_su < TOTAL_MAX_SU:
        raise ValueError("SU limit must cover the reservation cap %.3f" % TOTAL_MAX_SU)
    checked_plan()
    result = dict(estimate(), dry_run=not execute, max_su=max_su, account=ACCOUNT,
                  manifest_sha256=sha(PLAN), commands=submission_commands())
    if not execute:
        return result
    queue = query(["squeue", "-h", "-u", "shig", "-o", "%j"], text=True)
    if any(name.strip().startswith("refinement28") for name in queue.splitlines()):
        raise RuntimeError("Existing refinement28 jobs require inspection before submission")
    balance = account_balance(query(["myproject", "-l"], text=True))
    if balance < TOTAL_MAX_SU:
        raise RuntimeError("Current account balance %.2f SU cannot cover %.3f SU" % (balance, TOTAL_MAX_SU))
    result["available_su_at_submission"] = balance
    with (OUT / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal_path = OUT / "launch.json"
        if journal_path.exists():
            raise RuntimeError("Launch journal already exists; inspect recorded or ambiguous jobs, never resubmit automatically")
        checked_plan()
        for directory in (OUT / "logs" / "generation", OUT / "logs" / "report"):
            directory.mkdir(parents=True, exist_ok=True)
        record = dict(result, created_utc=utc_now(), state="submitting", jobs={}, attempts=[])
        write(journal_path, record)

        def submit(stage, command):
            intent = dict(stage=stage, command=command, state="submission_intent", created_utc=utc_now())
            record["attempts"].append(intent)
            write(journal_path, record)
            try:
                submitted = runner(command, cwd=ROOT, text=True, capture_output=True)
            except BaseException as error:
                intent.update(state="ambiguous_submission", error_type=type(error).__name__)
                record["state"] = "ambiguous_submission"
                write(journal_path, record)
                raise
            intent.update(returncode=submitted.returncode, stdout=submitted.stdout, stderr=submitted.stderr)
            job = submitted.stdout.strip().split(";")[0]
            if submitted.returncode or not re.fullmatch(r"[0-9]+", job):
                intent["state"] = record["state"] = "ambiguous_submission"
                write(journal_path, record)
                raise RuntimeError("Submission requires inspection; journal retained at " + str(journal_path))
            intent.update(state="submitted", job_id=job)
            record["jobs"][stage] = dict(job_id=job, command=command)
            write(journal_path, record)
            return job

        retry = submit("retry", submission_commands()["retry"])
        submit("report", submission_commands(retry_job=retry)["report"])
        record.update(state="submitted", submitted_utc=utc_now())
        write(journal_path, record)
        return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--dry-run", action="store_true", help="Read-only plan; this is the default")
    parser.add_argument("--max-su", type=float, default=DEFAULT_MAX_SU)
    args = parser.parse_args(argv)
    import json
    print(json.dumps(launch(execute=args.execute, max_su=args.max_su), indent=2))


if __name__ == "__main__":
    main()
