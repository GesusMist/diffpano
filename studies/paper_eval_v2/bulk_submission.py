"""Finite confirmation batches with scheduler reservations and no scientific retries."""
import math
from pathlib import Path

from .common import ROOT, digest, now, read, sha, write

MAX_BATCH_IMAGES = 6
GPU_SU_PER_TASK = 4 * 146
NEXT_STAGE_RESERVE_SU = 292 + 24 + 4


def array_indices(command):
    """Read the actual submitted Slurm array, including compact ranges."""
    options = [str(value).split("=", 1)[1] for value in command if str(value).startswith("--array=")]
    if len(options) != 1:
        raise ValueError("A submitted generation batch needs one recorded array specification")
    result = set()
    for item in options[0].split("%", 1)[0].split(","):
        if "-" in item:
            first, last = item.split("-", 1)
            result.update(range(int(first), int(last) + 1))
        else:
            result.add(int(item))
    if not result:
        raise ValueError("A submitted generation batch has no recorded indices")
    return result


def audit(run, plan, scheduler, completed, eligible_methods):
    """Only a successfully submitted job counts as attempted; rejected jobs do not."""
    from .controller import terminal
    run = Path(run)
    selected = {row["index"]: row for row in plan["rows"] if row["selected"]}
    attempted, failed, active_indices = set(), set(), set()
    blockers = []
    journals = {}
    for path in (run / "controller/generation_batches").glob("*.json"):
        record = read(path)
        journals[record["key"]] = record
        if record.get("job_id") and record["key"] not in scheduler.get("jobs", {}):
            blockers.append("Submitted batch journal is missing from scheduler records: " + record["key"])
    for key, job in scheduler.get("jobs", {}).items():
        if not key.startswith("confirmation-bulk"):
            continue
        try:
            indices = array_indices(job.get("command", []))
            if not indices <= set(selected):
                raise ValueError("Submitted array contains indices outside the selected frozen tier")
            if key in journals:
                record = journals[key]
                if indices != set(record["indices"]) or record["case_ids"] != [selected[i]["case_id"] for i in record["indices"]]:
                    raise ValueError("Submitted array differs from its frozen batch journal")
                if record["manifest_sha256"] != sha(run / "generation/manifest.json"):
                    raise ValueError("Submitted batch journal belongs to a different generation manifest")
        except (ValueError, TypeError) as exc:
            blockers.append(key + ": " + str(exc))
            continue
        attempted.update(indices)
        if terminal(job):
            failed.update(selected[index]["case_id"] for index in indices if selected[index]["case_id"] not in completed)
        else:
            active_indices.update(indices)
    # A process may have started before a crash or an externally submitted job.
    # Existing scientific attempt records are never erased or silently retried.
    for index, row in selected.items():
        path = run / "generation/cases" / row["case_id"] / "status.json"
        if index not in plan["profile_indices"] and path.exists():
            attempted.add(index)
            if index not in active_indices and row["case_id"] not in completed:
                failed.add(row["case_id"])
    for attempt in scheduler.get("attempts", []):
        if attempt["key"].startswith("confirmation-bulk") and attempt["state"] in ("submission_intent", "ambiguous_submission"):
            blockers.append("Unresolved submission must be reconciled before more generation: " + attempt["key"])
    pending = [index for index, row in sorted(selected.items()) if index not in plan["profile_indices"]
               and index not in attempted and row["case_id"] not in completed and row["method_id"] in eligible_methods]
    return dict(attempted_indices=sorted(attempted), pending_indices=pending, failed_case_ids=sorted(failed),
                submission_blockers=blockers)


def launch(run, plan, pending_indices, available_su):
    """Retain the frozen four-hour worker while reserving only one small array."""
    from .resources import submit
    run = Path(run)
    if not math.isfinite(available_su) or available_su < 0:
        raise ValueError("Account balance must be finite and nonnegative")
    capacity = min(MAX_BATCH_IMAGES, max(0, math.floor((available_su - NEXT_STAGE_RESERVE_SU) / GPU_SU_PER_TASK)))
    indices = list(pending_indices[:capacity])
    if not indices:
        return dict(state="insufficient_scheduler_reservation", required_su=GPU_SU_PER_TASK + NEXT_STAGE_RESERVE_SU,
                    available_su=available_su)
    case_ids = [plan["rows"][index]["case_id"] for index in indices]
    identity = dict(indices=indices, case_ids=case_ids, manifest_sha256=sha(run / "generation/manifest.json"))
    batch_id = digest(identity)[:20]
    key = "confirmation-bulk-" + batch_id
    path = run / "controller/generation_batches" / (batch_id + ".json")
    reservation = len(indices) * GPU_SU_PER_TASK + NEXT_STAGE_RESERVE_SU
    record = dict(identity, batch_id=batch_id, key=key, array_concurrency=2, walltime_hours=4,
                  scheduler_gpu_reservation_su=len(indices) * GPU_SU_PER_TASK, next_stage_reserve_su=NEXT_STAGE_RESERVE_SU)
    if path.exists():
        old = read(path)
        if any(old.get(field) != value for field, value in record.items()):
            raise ValueError("Frozen generation batch identity changed: " + batch_id)
    else:
        write(path, dict(record, created_utc=now()))
    job = submit(run, key, ROOT / "studies/paper_eval_v2/generation.slurm", role="generation",
                 script_args=("--run", str(run), "--stage", "bulk"),
                 options=("--array=" + ",".join(map(str, indices)) + "%2", "--exclude=fc026"), reserve_su=reservation)
    write(path, dict(read(path), job_id=job["job_id"]))
    return dict(job, indices=indices, case_ids=case_ids, batch_id=batch_id, reservation_su=reservation)
