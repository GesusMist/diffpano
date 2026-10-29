"""Finite incremental confirmation evaluation using the frozen metric implementations."""
import argparse
from pathlib import Path
import traceback

from .common import RUN, now, read, write, sha


def evaluate(run, batch):
    from .cli import evaluate as execute
    from .confirmation_index import aggregate
    run, batch = Path(run), Path(batch)
    request = read(batch)
    result_path = run / "controller/evaluations" / (request["batch_id"] + ".json")
    if sha(run / "generation/inventory_records.json") != request["inventory_records_sha256"]:
        raise ValueError("Confirmation index changed while an evaluation batch was reserved")
    result = dict(state="running", batch_id=request["batch_id"], requested_case_ids=request["case_ids"],
                  started_utc=now(), stages={}, completed_case_ids=[])
    write(result_path, result)
    for group, stage in (("group1", "cpu"), ("group1", "gpu"), ("views", "cpu")):
        try:
            value = execute(run, group, stage)
            failures = value.get("execution_failures", [])
            result["stages"][group + ":" + stage] = dict(state="failed" if failures else "complete", failures=failures)
        except Exception:
            result["stages"][group + ":" + stage] = dict(state="failed", error=traceback.format_exc())
        write(result_path, result)
    try:
        result["confirmation_aggregate"] = aggregate(run)
        values = read(run / "group1/per_panorama_metrics.json")
        metrics = {row["case_id"]: row["metrics"] for row in values if row["namespace"] == "common_v2"}
        required = {"DS", "CS-H", "CS-directional", "MUSIQ"}
        result["completed_case_ids"] = [case for case in request["case_ids"] if required <= set(metrics.get(case, {}))]
        result["missing_case_ids"] = sorted(set(request["case_ids"]) - set(result["completed_case_ids"]))
    except Exception:
        result["aggregate_error"] = traceback.format_exc()
    result["state"] = "complete" if len(result["completed_case_ids"]) == len(request["case_ids"]) and all(item["state"] == "complete" for item in result["stages"].values()) else "failed"
    result["completed_utc"] = now()
    write(result_path, result)
    if result["state"] != "complete":
        raise RuntimeError("Confirmation evaluation incomplete; inspect " + str(result_path))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    print(evaluate(args.run, args.batch))
