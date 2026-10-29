"""Incrementally expose fresh confirmation outputs without rewriting frozen evaluators."""
import argparse
import copy
import hashlib
from pathlib import Path

from .common import RUN, ROOT, atomic_text, csv_write, digest, lock, read, sha, write
from . import generation

METHODS = ("sphere20", "select", "spherope")
COMPARISONS = (("sphere20", "select"), ("spherope", "select"))


def prompt_record(row):
    path = Path(row["prompt_path"])
    if sha(path) != row["prompt_sha256"] or path.read_text().splitlines() != row["directional_descriptions"]:
        raise ValueError("Frozen confirmation prompt bytes differ")
    return dict(prompt_id=row["prompt_id"], prompt_name=row["scene_id"], prompt_path=str(path),
        full_path=str(path), filename=path.name, line_count=5, effective_lines=row["directional_descriptions"],
        effective_sha256=row["prompt_sha256"], original_sha256=row["prompt_sha256"],
        line_sha256=[hashlib.sha256(line.encode()).hexdigest() for line in row["directional_descriptions"]],
        memberships=["confirmation24"], source_group="fresh_real_confirmation", adapted=False)


def records(run, plan):
    """Include declared missing/unselected rows for coverage; only verified images are eligible."""
    run = Path(run)
    manifest_path = run / "generation/manifest.json"
    manifest_sha = sha(manifest_path)
    result = []
    for row in plan["rows"]:
        folder = generation.case_folder(run, row)
        status = read(folder / "status.json") if (folder / "status.json").exists() else {}
        valid = generation.complete(run, row)
        metadata = read(folder / "metadata.json") if valid else {}
        eligible = valid and row["selected"]
        state = "complete" if eligible else ("not_selected" if not row["selected"] else status.get("state", "missing"))
        if state == "complete" and not valid:
            state = "invalid"
        spec = row["specification"]
        prompt = prompt_record(row)
        image_sha = sha(folder / "final.png") if valid else None
        parameters = dict(backend=row["backend"], seed=row["seed"], steps=spec["steps"],
            erp_resolution=[spec["width"], spec["height"]], model=dict(id=spec["checkpoint"],
            revision=spec["revision"], precision=spec["precision"]), configuration=metadata.get("configuration", spec),
            configuration_sha256=row.get("configuration_sha256", digest(spec)),
            camera_geometry_sha256=row.get("camera_geometry_sha256"), schedule_sha256=row.get("schedule_sha256"),
            counts=metadata.get("counts", {}), scheduler=metadata.get("scheduler"),
            refinement_steps=0, specification=spec)
        item = dict(case_id=row["case_id"], panorama_id=digest(dict(method=row["method_id"],
            prompt_id=row["prompt_id"], seed=row["seed"], image_sha256=image_sha)) if valid else None,
            method_id=row["method_id"], family="confirmation24", memberships=["confirmation24"],
            scene_id=row["scene_id"], prompt_id=row["prompt_id"], prompt_name=row["scene_id"],
            prompt_path=row["prompt_path"], effective_prompt_sha256=row["prompt_sha256"], prompt=prompt,
            scoring_prompt=row.get("scoring_prompt"), seed=row["seed"], tier=row["tier"], selected=row["selected"],
            manifest_path=str(manifest_path), manifest_sha256=manifest_sha, manifest_index=row["index"],
            image_path=str(folder / "final.png"), image_sha256=image_sha,
            metadata_path=str(folder / "metadata.json"), metadata_sha256=sha(folder / "metadata.json") if valid else None,
            dimensions=[spec["width"], spec["height"]], dimensions_verified=valid,
            eligible=eligible, state=state, reasons=[] if eligible else ["unselected tier" if not row["selected"] else "No verified complete generation output: " + state],
            parameters=parameters, runtime={key: metadata.get(key) for key in
                ("generation_seconds", "setup_seconds", "stage_seconds", "peak_allocated_gib", "gpu", "node")},
            provenance=dict(reused=False, semantic_sha256=row["semantic_sha256"], source_revision=metadata.get("source_revision"),
                generation_source_sha256=plan["source_sha256"], source_image_sha256=row["source_image_sha256"],
                caption_sha256=row["caption_sha256"], scientific_configuration_source="frozen fresh-generation manifest and retained executed metadata"))
        result.append(item)
    return result


def augment_cohorts(original, rows, plan_sha):
    updated = copy.deepcopy(original)
    by_prompt = {}
    for row in rows:
        if row["tier"] == "core":
            previous = by_prompt.setdefault(row["prompt_id"], row["prompt"])
            if previous != row["prompt"]:
                raise ValueError("Confirmation prompt changed across methods")
    if len(by_prompt) != 24:
        raise ValueError("Exactly 24 frozen confirmation prompts are required")
    prompt_ids = sorted(by_prompt)
    existing_prompts = {row["prompt_id"]: row for row in updated["prompts"]}
    for identity, prompt in sorted(by_prompt.items()):
        if identity in existing_prompts and existing_prompts[identity] != prompt:
            raise ValueError("Confirmation identity collides with a pre-existing prompt")
    updated["prompts"] = [row for row in updated["prompts"] if row["prompt_id"] not in by_prompt] + [by_prompt[p] for p in prompt_ids]
    for cohort_id, cohort in original["cohorts"].items():
        if cohort_id != "confirmation24" and set(cohort.get("prompt_ids", [])) & set(prompt_ids):
            raise ValueError("Fresh confirmation identities overlap a development cohort")
    complete = [row for row in rows if row["eligible"]]
    updated["selected_cells"] = [cell for cell in updated["selected_cells"] if cell["prompt_id"] not in by_prompt]
    updated["selected_cells"].extend({key: row[key] for key in ("case_id", "method_id", "panorama_id", "prompt_id", "seed")} for row in complete)
    seed0 = {(row["prompt_id"], row["method_id"]): row for row in complete if row["seed"] == 0 and row["method_id"] in METHODS}
    matched = [p for p in prompt_ids if all((p, method) in seed0 for method in METHODS)]
    excluded = [dict(prompt_id=p, missing_methods=[method for method in METHODS if (p, method) not in seed0]) for p in prompt_ids if p not in matched]
    updated["cohorts"]["confirmation24"] = dict(prompt_count=24, prompt_ids=prompt_ids,
        state="complete" if len(matched) == 24 else "generation_incomplete", method_ids=list(METHODS),
        seed_policy="Core seed0; declared extensions remain separately identifiable by tier and seed.",
        generation_manifest_sha256=plan_sha, matched=dict(primary=dict(method_ids=list(METHODS),
            prompt_count=len(matched), prompt_ids=matched, excluded=excluded,
            panorama_ids=[seed0[(p, m)]["panorama_id"] for p in matched for m in METHODS])))
    # This assert also protects development matched subsets and any future named cohorts.
    assert {k: v for k, v in updated["cohorts"].items() if k != "confirmation24"} == {k: v for k, v in original["cohorts"].items() if k != "confirmation24"}
    return updated


def index(run=RUN):
    run = Path(run)
    plan = generation.gate(run)
    with lock(run / "generation/confirmation_index.lock"):
        rows = records(run, plan)
        plan_sha = sha(run / "generation/manifest.json")
        targets = (run / "cohorts.json", run / "inventory/cohorts.json")
        snapshots = run / "generation/initial_cohorts"
        snapshots.mkdir(exist_ok=True)
        snapshot_evidence = {}
        pending = []
        for target in targets:
            name = "root_cohorts.json" if target.parent == run else "inventory_cohorts.json"
            snapshot = snapshots / name
            if not snapshot.exists():
                atomic_text(snapshot, target.read_text())
            initial = read(snapshot)
            current = read(target)
            for key, cohort in initial["cohorts"].items():
                if key != "confirmation24" and current["cohorts"].get(key) != cohort:
                    raise ValueError("Development cohort changed since the preserved initial snapshot")
            pending.append((target, augment_cohorts(current, rows, plan_sha)))
            snapshot_evidence[str(target.relative_to(run))] = dict(path=str(snapshot), sha256=sha(snapshot))
        write(run / "generation/inventory_records.json", rows)
        for target, value in pending:
            write(target, value)
        state_counts = {state: sum(row["state"] == state for row in rows) for state in sorted({row["state"] for row in rows})}
        evidence = dict(manifest_sha256=plan_sha, inventory_records_sha256=sha(run / "generation/inventory_records.json"),
            initial_cohorts=snapshot_evidence, selected=sum(row["selected"] for row in rows),
            eligible=sum(row["eligible"] for row in rows), states=state_counts)
        write(run / "generation/confirmation_index.json", evidence)
        return evidence


def summarize_confirmation(rows, prompt_ids):
    """Use frozen prompt-cluster statistics with confirmation-specific paired contrasts."""
    from .statistics import summarize
    output = dict(groups=[], paired_comparisons=[])
    selected = [row for row in rows if row["prompt_id"] in set(prompt_ids) and row["method_id"] in METHODS and row.get("seed", 0) == 0]
    for namespace in sorted({row.get("namespace", "common_v2") for row in selected}):
        subset = [row for row in selected if row.get("namespace", "common_v2") == namespace]
        for metric in sorted({key for row in subset for key in row.get("metrics", {})}):
            policies = sorted({row.get("text_policy_id", "unspecified") for row in subset if metric in row.get("metrics", {})}) if metric.startswith("CS") else ["not_applicable"]
            for policy in policies:
                metric_rows = [row for row in subset if row.get("text_policy_id", "unspecified") == policy] if metric.startswith("CS") else subset
                if metric.startswith("CS"):
                    for prompt_id in prompt_ids:
                        hashes = {row.get("scoring_text_sha256", row.get("effective_prompt_sha256")) for row in metric_rows if row["prompt_id"] == prompt_id}
                        if len(hashes) > 1:
                            raise ValueError("Confirmation methods scored different canonical text")
                groups, pairs = summarize(metric_rows, metric, "confirmation24", prompt_ids, comparisons=COMPARISONS)
                output["groups"].extend(dict(namespace=namespace, text_policy_id=policy, seed_scope="core_seed0", **row) for row in groups)
                output["paired_comparisons"].extend(dict(namespace=namespace, text_policy_id=policy, seed_scope="core_seed0", **row) for row in pairs)
    return output


def aggregate(run=RUN):
    run = Path(run)
    rows = read(run / "group1/per_panorama_metrics.json")
    prompts = read(run / "cohorts.json")["cohorts"]["confirmation24"]["prompt_ids"]
    result = summarize_confirmation(rows, prompts)
    directory = run / "group1/confirmation"
    for name, values in result.items():
        write(directory / (name + ".json"), values)
        csv_write(directory / (name + ".csv"), values, sorted({key for row in values for key in row}))
    return dict(groups=len(result["groups"]), paired_comparisons=len(result["paired_comparisons"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("index", "aggregate"))
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    import json
    print(json.dumps(index(args.run) if args.action == "index" else aggregate(args.run), indent=2))


if __name__ == "__main__":
    main()
