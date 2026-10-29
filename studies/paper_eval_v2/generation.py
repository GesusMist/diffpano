"""Finite fresh-confirmation generation with immutable manifests and safe reuse."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import platform
import tarfile
import time
import traceback

from .common import ROOT, RUN, read, write, sha, digest, freeze, now
from .baselines import specifications, scientific_configuration

CORE_METHODS = ("spherope", "sphere20", "select")
EXPECTED_TIER_COUNTS = {"core": 72, "seeds": 24, "trained": 24, "generality": 32, "full": 152}


def sources():
    files = set()
    for directory in ("diffpano", "studies/all_prompts", "studies/gradient_blending", "studies/tt_cea",
                      "studies/gwtf_noise", "studies/gwtf_erp4k", "studies/original_spherediff"):
        files.update(path for path in (ROOT / directory).rglob("*") if path.is_file() and "__pycache__" not in path.parts
                     and path.suffix in (".py", ".json", ".yaml", ".txt"))
    files.update(ROOT / "studies/paper_eval_v2" / name for name in
                 ("generation.py", "baselines.py", "baseline_validation.py", "generation.slurm"))
    files.update(ROOT / "scripts" / name for name in ("generate.py", "dense_erp_experiment.py"))
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(files)}


def case_definitions(captions, dataset, selected_tier="core"):
    if selected_tier not in EXPECTED_TIER_COUNTS:
        raise ValueError("Unknown generation tier")
    if captions.get("state") != "complete" or captions.get("reviewed") is not True:
        raise ValueError("Generation requires complete, reviewed, frozen real-scene captions")
    caption_rows = captions.get("rows", captions.get("records", []))
    real_rows = dataset.get("rows", dataset.get("records", []))
    if len(caption_rows) != 24 or len(real_rows) != 24:
        raise ValueError("confirmation24 must contain exactly 24 real scenes and caption records")
    real = {row["scene_id"]: row for row in real_rows}
    if len(real) != 24 or {row["scene_id"] for row in caption_rows} != set(real):
        raise ValueError("Real-scene and caption identities differ")
    subset = dataset.get("second_seed_scene_ids", [])
    if len(subset) != 8 or len(set(subset)) != 8 or not set(subset) <= set(real):
        raise ValueError("Eight second-seed scenes must be frozen before generation")
    variants = specifications()
    selected = {"core"} if selected_tier == "core" else ({"core", "seeds", "trained", "generality"}
                  if selected_tier == "full" else {"core", selected_tier})
    cases = []
    for tier in ("core", "seeds", "trained", "generality"):
        scene_captions = sorted(caption_rows, key=lambda row: row["scene_id"])
        if tier in ("seeds", "generality"):
            scene_captions = [row for row in scene_captions if row["scene_id"] in subset]
        methods = CORE_METHODS if tier in ("core", "seeds") else (("dit360",) if tier == "trained" else ("sphere20_sana", "select_sana"))
        seeds = (1,) if tier == "seeds" else ((0, 1) if tier == "generality" else (0,))
        for caption in scene_captions:
            if caption["source_image_sha256"] != real[caption["scene_id"]]["image_sha256"]:
                raise ValueError("Caption source differs from frozen real scene")
            lines = caption["directional_descriptions"]
            if len(lines) != 5 or not all(isinstance(line, str) and line.strip() and "\n" not in line for line in lines):
                raise ValueError("Exactly five nonempty frozen directional descriptions required")
            prompt = caption["generation_prompt"]
            path = Path(prompt["path"])
            if not path.is_absolute():
                path = ROOT / path
            if sha(path) != prompt["effective_sha256"] or path.read_text().splitlines() != lines:
                raise ValueError("Frozen generation prompt bytes differ")
            for seed in seeds:
                for method in methods:
                    base_method = method.removesuffix("_sana")
                    spec = dict(variants[base_method])
                    backend = "sana" if method.endswith("_sana") else "flux"
                    if backend == "sana":
                        spec.update(backend="sana", checkpoint="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
                                    revision="e2b3c0cbffebcd09d83805e88b9f5f106afc74ac", guidance_scale=4.5)
                        if base_method == "sphere20":
                            spec["n_spherical_points"] = 2600
                    case_id = "confirmation24/%s/%s/seed%d" % (caption["scene_id"], method, seed)
                    cases.append({"index": len(cases), "case_id": case_id, "tier": tier, "selected": tier in selected,
                                  "method_id": method, "backend": backend, "scene_id": caption["scene_id"],
                                  "prompt_id": caption["prompt_id"], "seed": seed, "specification": spec,
                                  "prompt_path": str(path.resolve()), "prompt_sha256": prompt["effective_sha256"],
                                  "global_generation_description": caption["global_generation_description"],
                                  "directional_descriptions": lines, "scoring_prompt": caption.get("scoring_prompt"),
                                  "caption_sha256": digest(caption), "source_image_sha256": caption["source_image_sha256"]})
    for tier, count in EXPECTED_TIER_COUNTS.items():
        if tier != "full" and sum(row["tier"] == tier for row in cases) != count:
            raise AssertionError("Nominal generation tier count mismatch")
    assert len(cases) == 152 and len({row["case_id"] for row in cases}) == 152
    return cases


def schedule_identity(schedule):
    # Preserve the historical SANA negative-infinity bound as its existing JSON sentinel.
    from studies.tt_cea.common import historical_scheduler_json
    return digest(historical_scheduler_json(schedule))


def prepare(run=RUN):
    run = Path(run)
    folder = run / "generation"
    folder.mkdir(exist_ok=True)
    config = read(run / "run_config.json")
    caption_path, dataset_path = run / "captions.json", run / "real_dataset_manifest.json"
    captions, dataset = read(caption_path), read(dataset_path)
    cases = case_definitions(captions, dataset, config["selected_generation_tier"])
    evidence_path = folder / "final_method_evidence.json"
    evidence_sha256 = sha(evidence_path)
    final_rule = {"method_id": "select", "configuration": specifications()["select"],
                  "selection": "predeclared_fallback", "confirmation_not_used_for_selection": True,
                  "evidence_path": str(evidence_path), "evidence_sha256": evidence_sha256,
                  "interpretation": "Select is carried forward as the predeclared fallback under unresolved quality/semantics/geometry tradeoffs; it is not declared the best development method.",
                  "rule": "Carry unrefined primary screened-select Old89 forward when available development quality/geometry/semantics/cost do not identify a single winner; never select solely by DS or a favorable judge.",
                  "refinement_included": False, "missing_required_development_cases": [],
                  "development_coverage": "All 24 primary screened-select cases were valid in the Phase 1 inventory."}
    freeze(folder / "final_method.json", final_rule)
    own_sources = sources()
    from studies.original_spherediff.common import official_hashes
    from .baseline_validation import source_evidence
    baseline_sources = source_evidence(run)
    for row in cases:
        if row["method_id"].startswith("select"):
            configuration, cameras, slots, old = scientific_configuration(row)
            row["configuration_sha256"] = digest(configuration.to_dict())
            row["camera_geometry_sha256"] = old["camera_geometry_sha256"]
            row["schedule_sha256"] = schedule_identity(old["prepared_schedule"])
        row["semantic_sha256"] = digest({key: value for key, value in row.items() if key != "index"})
    selected = [row["index"] for row in cases if row["selected"]]
    plan = {"study": "paper_eval_v2_confirmation", "created_utc": now(), "tier": config["selected_generation_tier"],
            "rows": cases, "selected_indices": selected, "nominal_counts": EXPECTED_TIER_COUNTS,
            "selected_erp_slots": len(selected), "native_reference_slots_separate": 240,
            "profile_indices": [0, 1, 2], "bulk_indices": [index for index in selected if index not in (0, 1, 2)],
            "caption_manifest_sha256": sha(caption_path), "real_dataset_manifest_sha256": sha(dataset_path),
            "final_method_sha256": sha(folder / "final_method.json"), "final_method_evidence_sha256": evidence_sha256, "source_hashes": own_sources,
            "checkpoint_equivalence_sha256": sha(folder / "checkpoint_equivalence.json"),
            "baseline_environment_sha256": sha(folder / "baseline_environment.json"),
            "baseline_import_validation_sha256": sha(folder / "baseline_import_validation.json"),
            "source_sha256": digest(own_sources), "official_spherediff_source_hashes": official_hashes(),
            "baseline_sources": baseline_sources, "generation_gpu_limit": 2,
            "walltime_per_case": "04:00:00", "exclude_nodes": ["fc026"],
            "profile_policy": "First real scene per core method is a counted study output; remaining cases require that method's successful complete profile.",
            "blockers": {"lf_multidiffusion": specifications()["lf_multidiffusion"]}}
    path = folder / "manifest.json"
    if path.exists():
        old = read(path)
        plan["created_utc"] = old["created_utc"]
    freeze(path, plan)
    archive = folder / "implementation.tar.gz"
    if not archive.exists():
        with tarfile.open(archive, "w:gz") as handle:
            for relative in sorted(own_sources):
                handle.add(ROOT / relative, arcname=relative)
    return {"selected_cases": len(selected), "all_declared_cases": len(cases), "manifest": str(path), "state": "prepared"}


def gate(run):
    run = Path(run)
    plan = read(run / "generation/manifest.json")
    if sources() != plan["source_hashes"]:
        raise ValueError("Frozen generation source changed")
    for name, key in (("captions.json", "caption_manifest_sha256"), ("real_dataset_manifest.json", "real_dataset_manifest_sha256")):
        if sha(run / name) != plan[key]:
            raise ValueError("Frozen scientific input changed: " + name)
    from studies.original_spherediff.common import official_hashes
    assert official_hashes() == plan["official_spherediff_source_hashes"]
    from .baseline_validation import source_evidence
    assert source_evidence(run) == plan["baseline_sources"]
    for name in ("final_method", "final_method_evidence"):
        if sha(run / ("generation/" + name + ".json")) != plan[name + "_sha256"]:
            raise ValueError("Frozen final-method selection evidence changed: " + name)
    weights = read(run / "generation/checkpoint_equivalence.json")
    if not weights["all_shared_files_identical"]:
        raise ValueError("Pinned mirror/official checkpoint equivalence not validated")
    for name in ("checkpoint_equivalence", "baseline_environment", "baseline_import_validation"):
        if sha(run / ("generation/" + name + ".json")) != plan[name + "_sha256"]:
            raise ValueError("Frozen baseline validation changed: " + name)
    return plan


def case_folder(run, row):
    identity = Path(row["case_id"])
    assert not identity.is_absolute() and ".." not in identity.parts
    return Path(run) / "generation/cases" / identity


def complete(run, row):
    folder = case_folder(run, row)
    try:
        status = read(folder / "status.json")
        metadata = read(folder / "metadata.json")
        if status["state"] != "complete" or status["metadata_sha256"] != sha(folder / "metadata.json"):
            return False
        if metadata["semantic_sha256"] != row["semantic_sha256"] or sha(folder / "final.png") != status["image_sha256"]:
            return False
        from PIL import Image
        with Image.open(folder / "final.png") as image:
            if image.size != (row["specification"]["width"], row["specification"]["height"]) or image.mode != "RGB":
                return False
            image.verify()
        return True
    except (OSError, ValueError, KeyError):
        return False


def run_case(index, run=RUN, profile=False):
    run = Path(run)
    plan = gate(run)
    row = plan["rows"][index]
    if row["index"] != index or index not in plan["selected_indices"]:
        raise ValueError("Requested case is not in the user-selected tier")
    if profile != (index in plan["profile_indices"]):
        raise ValueError("Profile stage does not match the frozen profile indices")
    if complete(run, row):
        return {"state": "reused_valid_output", "case_id": row["case_id"]}
    if not profile and row["method_id"] in CORE_METHODS:
        prerequisite = next(plan["rows"][i] for i in plan["profile_indices"] if plan["rows"][i]["method_id"] == row["method_id"])
        if not complete(run, prerequisite):
            raise RuntimeError("This method's real full-resolution profile has not completed: " + prerequisite["case_id"])
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("A GPU Slurm allocation is required")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible generation GPU is required")
    folder = case_folder(run, row)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "case.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / "status.json").exists() or (folder / "final.png").exists():
            raise RuntimeError("Existing incomplete attempt must be reconciled and archived explicitly before retry")
        status = {"case_id": row["case_id"], "state": "running", "job_id": os.environ["SLURM_JOB_ID"],
                  "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"), "node": platform.node(), "started_utc": now()}
        write(folder / "status.json", status)
        try:
            from .baselines import run_select, run_official
            started = time.perf_counter()
            metadata = run_select(row, folder) if row["method_id"].startswith("select") else run_official(row, folder, run)
            metadata.update(case=row, semantic_sha256=row["semantic_sha256"], source_sha256=plan["source_sha256"],
                            manifest_sha256=sha(run / "generation/manifest.json"), source_revision=os.popen("git rev-parse HEAD").read().strip(),
                            image_sha256=sha(folder / "final.png"), process_generation_seconds=time.perf_counter() - started,
                            gpu=torch.cuda.get_device_name(), node=platform.node(), job_id=os.environ["SLURM_JOB_ID"],
                            prompt_identity=row["prompt_id"], prompt_sha256=row["prompt_sha256"],
                            native_resolution=[row["specification"]["width"], row["specification"]["height"]])
            gate(run)
            write(folder / "metadata.json", metadata)
            write(folder / "status.json", dict(status, state="complete", completed_utc=now(),
                   metadata_sha256=sha(folder / "metadata.json"), image_sha256=metadata["image_sha256"]))
            assert complete(run, row)
            return {"state": "complete", "case_id": row["case_id"], "generation_seconds": metadata["generation_seconds"]}
        except BaseException:
            write(folder / "status.json", dict(status, state="failed", failed_utc=now(), error=traceback.format_exc()))
            raise


def status(run=RUN):
    run = Path(run)
    plan = read(run / "generation/manifest.json")
    cases = []
    for row in plan["rows"]:
        folder = case_folder(run, row)
        record = read(folder / "status.json") if (folder / "status.json").exists() else {}
        valid = complete(run, row)
        cases.append({"case_id": row["case_id"], "index": row["index"], "tier": row["tier"], "method_id": row["method_id"],
                      "state": "complete" if valid else record.get("state", "pending" if row["selected"] else "not_selected"),
                      "image_path": str(folder / "final.png") if valid else None, "selected": row["selected"]})
    result = {"cases": cases, "selected": sum(row["selected"] for row in cases),
              "verified_complete": sum(row["state"] == "complete" for row in cases), "updated_utc": now()}
    write(run / "generation/completion.json", result)
    return result


def launch(run=RUN, stage="profiles", dependency=None):
    from .resources import submit
    run = Path(run)
    plan = gate(run)
    indices = plan["profile_indices"] if stage == "profiles" else plan["bulk_indices"]
    pending = [index for index in indices if not complete(run, plan["rows"][index])]
    if not pending:
        return {"state": "all_requested_cases_complete"}
    options = ["--array=" + ",".join(str(index) for index in pending) + "%2", "--exclude=fc026"]
    if dependency:
        options += ["--dependency=afterany:" + str(dependency), "--kill-on-invalid-dep=yes"]
    # Initial estimate uses historical select/SphereDiff times plus a conservative
    # SpheRoPE allowance. Profile measurements are retained for actual planning.
    estimated = sum({"select": 100., "sphere20": 40., "spherope": 30., "dit360": 30.,
                     "select_sana": 40., "sphere20_sana": 20.}[plan["rows"][i]["method_id"]] for i in pending)
    return submit(run, "confirmation-" + stage, ROOT / "studies/paper_eval_v2/generation.slurm", role="generation",
                  script_args=("--run", str(run), "--stage", stage), options=options, reserve_su=estimated)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status", "launch"))
    parser.add_argument("--run", default=str(RUN))
    parser.add_argument("--stage", choices=("profiles", "bulk"), default="profiles")
    parser.add_argument("--index", type=int)
    parser.add_argument("--dependency")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.run)
    elif args.command == "run":
        index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        result = run_case(index, args.run, profile=args.stage == "profiles")
    elif args.command == "status":
        result = status(args.run)
    else:
        result = launch(args.run, args.stage, args.dependency)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
