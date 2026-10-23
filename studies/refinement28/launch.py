"""Profile-gated Slurm launcher for the 48 authorized refinement panoramas.

Dry-run is the default. Every submission intent is persisted before sbatch;
any existing launch journal requires inspection, including ambiguous failures.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import math
from pathlib import Path
import re
import subprocess

from .common import ACCOUNT, OUT, ROOT, freeze_check, read, sha, validation_gate, write

PREFIX = "refinement28"
HOURLY_GENERATION_SU = 146.0
GENERATION_WALL_HOURS = 1.5
GENERATION_MAX_SU = 48 * HOURLY_GENERATION_SU * GENERATION_WALL_HOURS
REPORT_MAX_SU = 4 * (80 / 60)
TOTAL_MAX_SU = GENERATION_MAX_SU + REPORT_MAX_SU
DEFAULT_MAX_SU = math.ceil(TOTAL_MAX_SU)
PROFILES = (0, 1)
BULK = tuple(range(2, 48))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def estimate():
    # Measured FASTER Uniform89 screened-select FLUX20 mean, scaled to 28
    # steps. Old89/refinement timing remains an extrapolation until profiles.
    expected_hours = 48 * (2138.6820458410502 / 3600) * (28 / 20)
    return dict(generation_cases=48, profile_indices=list(PROFILES), bulk_indices=list(BULK),
                maximum_concurrent_gpu_jobs=2, gpu="NVIDIA A100 40GB", cpus=4, host_memory="64G",
                walltime="01:30:00", generation_hourly_su=HOURLY_GENERATION_SU,
                estimated_gpu_hours=expected_hours, estimated_generation_su=expected_hours * HOURLY_GENERATION_SU,
                estimated_elapsed_hours_at_two_gpus=expected_hours / 2,
                generation_max_su=GENERATION_MAX_SU, report_max_su=REPORT_MAX_SU, total_max_su=TOTAL_MAX_SU,
                note="Runtime extrapolated from 19 completed FASTER Uniform89 screened-select FLUX20 cases; both refinement profiles must validate before bulk. SU cap assumes verified local rates; actual elapsed charges can be lower.")


def account_balance(text):
    for line in text.splitlines():
        fields = [part.strip() for part in line.strip().strip("|").split("|")]
        if fields and fields[0] == ACCOUNT:
            try:
                value = float(fields[5].replace(",", ""))
            except (IndexError, ValueError):
                raise ValueError("Cannot parse current account SU balance") from None
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid account SU balance")
            return value
    raise ValueError("Requested FASTER account is absent from myproject output")


def submission_commands(profile_job="<profile_job_id>", bulk_job="<bulk_job_id>"):
    scripts = ROOT / "studies" / "refinement28"
    base = ["sbatch", "--parsable", "--chdir=" + str(ROOT)]
    logs = OUT / "logs"
    generation_log = "--output=" + str(logs / "generation" / "%x-%A_%a.out")
    return {
        "profiles": base + ["--job-name=" + PREFIX + "-profiles", generation_log,
                            "--array=0,1%2", str(scripts / "generate.slurm"), "--stage", "profile"],
        "bulk": base + ["--job-name=" + PREFIX + "-bulk", generation_log,
                        "--dependency=afterok:" + profile_job, "--kill-on-invalid-dep=yes", "--array=2-47%2",
                        str(scripts / "generate.slurm"), "--stage", "bulk"],
        "report": base + ["--job-name=" + PREFIX + "-report",
                          "--output=" + str(logs / "report" / "%x-%j.out"),
                          "--dependency=afterany:" + profile_job + ":" + bulk_job,
                          "--kill-on-invalid-dep=yes", str(scripts / "report.slurm")],
    }


def checked_manifest():
    manifest = freeze_check()
    validation_gate(manifest)
    rows = manifest["rows"]
    if len(rows) != 48 or [row["index"] for row in rows] != list(range(48)):
        raise ValueError("The authorized manifest must contain exactly 48 ordered cases")
    for index in range(0, 48, 2):
        if rows[index]["prompt"] != rows[index + 1]["prompt"]:
            raise ValueError("Refinement cases must be paired by prompt")
        if [rows[index]["refinement_steps"], rows[index + 1]["refinement_steps"]] != [5, 8]:
            raise ValueError("Only the predeclared 5/28 and 8/28 refinements are authorized")
    return manifest


def launch(*, execute=False, max_su=DEFAULT_MAX_SU, runner=subprocess.run, query=subprocess.check_output):
    if not math.isfinite(max_su) or max_su < TOTAL_MAX_SU:
        raise ValueError("SU limit must cover the complete reservation cap %.3f" % TOTAL_MAX_SU)
    manifest = checked_manifest()
    plan = dict(estimate(), dry_run=not execute, max_su=max_su, account=ACCOUNT,
                source_sha256=manifest["source_sha256"], manifest_sha256=sha(OUT / "manifest.json"),
                commands=submission_commands())
    if not execute:
        # Offline and read-only: no locks, journals, scheduler calls, or mkdirs.
        return plan
    queue = query(["squeue", "-h", "-u", "shig", "-o", "%j"], text=True)
    if any(name.strip().startswith(PREFIX) for name in queue.splitlines()):
        raise RuntimeError("Existing refinement28 jobs require inspection before submission")
    balance = account_balance(query(["myproject", "-l"], text=True))
    if balance < TOTAL_MAX_SU:
        raise RuntimeError("Current account balance %.2f SU cannot cover %.3f SU" % (balance, TOTAL_MAX_SU))
    plan["available_su_at_submission"] = balance
    with (OUT / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal_path = OUT / "launch.json"
        if journal_path.exists():
            raise RuntimeError("Launch journal already exists; inspect recorded or ambiguous jobs, never resubmit automatically")
        # Repeat the immutable gate under the submission lock.
        checked_manifest()
        for directory in (OUT / "logs" / "generation", OUT / "logs" / "report"):
            directory.mkdir(parents=True, exist_ok=True)
        record = dict(plan, created_utc=utc_now(), state="submitting", jobs={}, attempts=[])
        write(journal_path, record)

        def submit(stage, command):
            intent = dict(stage=stage, command=command, state="submission_intent", created_utc=utc_now())
            record["attempts"].append(intent)
            write(journal_path, record)
            try:
                result = runner(command, cwd=ROOT, text=True, capture_output=True)
            except BaseException as error:
                intent.update(state="ambiguous_submission", error_type=type(error).__name__)
                record["state"] = "ambiguous_submission"
                write(journal_path, record)
                raise
            intent.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            job = result.stdout.strip().split(";")[0]
            if result.returncode or not re.fullmatch(r"[0-9]+", job):
                # Even unusual sbatch failure text may follow server acceptance.
                intent["state"] = record["state"] = "ambiguous_submission"
                write(journal_path, record)
                raise RuntimeError("Submission result requires inspection; journal retained at " + str(journal_path))
            intent.update(state="submitted", job_id=job)
            record["jobs"][stage] = dict(job_id=job, command=command)
            write(journal_path, record)
            return job

        profile = submit("profiles", submission_commands()["profiles"])
        bulk = submit("bulk", submission_commands(profile_job=profile)["bulk"])
        submit("report", submission_commands(profile_job=profile, bulk_job=bulk)["report"])
        record.update(state="submitted", submitted_utc=utc_now())
        write(journal_path, record)
        return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true", help="Submit the authorized jobs after all frozen gates pass")
    actions.add_argument("--dry-run", action="store_true", help="Read-only plan; this is the default")
    parser.add_argument("--max-su", type=float, default=DEFAULT_MAX_SU)
    args = parser.parse_args(argv)
    import json
    print(json.dumps(launch(execute=args.execute, max_su=args.max_su), indent=2))


if __name__ == "__main__":
    main()
