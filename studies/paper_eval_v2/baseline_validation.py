"""Finite CPU checks for official baseline source, weights, and isolated imports."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from .common import ROOT, RUN, read, write, sha, now
from .baselines import MIRROR_SNAPSHOT, OFFICIAL_SNAPSHOT, SPHEROPE_COMMIT, DIT360_COMMIT

PACKAGES = ("diffusers==0.37.1", "transformers==4.57.6", "tokenizers==0.22.2", "huggingface-hub==0.36.2",
            "httpx==0.28.1", "httpcore==1.0.9", "anyio==4.9.0", "sniffio==1.3.1", "h11==0.16.0")


def source_evidence(run):
    result = {}
    for name, commit in (("SpheRoPE", SPHEROPE_COMMIT), ("DiT360", DIT360_COMMIT)):
        folder = Path(run) / "external" / name
        manifest = read(folder / "source_manifest.json")
        assert manifest["commit"] == commit
        expected = {item["path"]: item["sha"] for item in manifest["tree"]["tree"] if item["type"] == "blob"}
        import hashlib
        files = {}
        for path in sorted((folder / "source").rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(folder / "source").as_posix()
            raw = path.read_bytes()
            git_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            assert expected[relative] == git_blob, "Official source bytes differ: " + str(path)
            files[relative] = sha(path)
        assert files and "README.md" in files
        result[name] = {"commit": commit, "repository": manifest["repository"], "files": files,
                        "source_manifest_sha256": sha(folder / "source_manifest.json")}
    return result


def weight_audit(run):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Full checkpoint hashing requires a finite CPU allocation")
    evidence_path = ROOT / "outputs/10.9ournew-flux28/seed0-v1/manifest.json"
    historical = read(evidence_path)["weight_audit"]
    assert historical["all_shared_files_identical"]
    checked, shared = [], {}
    for old in historical["files"]:
        row = {"name": old["name"], "methods": {}}
        for method, snapshot in (("diffpano", MIRROR_SNAPSHOT), ("official", OFFICIAL_SNAPSHOT)):
            path = snapshot / old["name"]
            assert path.is_file()
            resolved = str(path.resolve())
            checksum = shared.setdefault(resolved, None)
            if checksum is None:
                checksum = shared[resolved] = sha(path)
            old_method = "spherediff" if method == "official" else method
            assert checksum == old["methods"][old_method]["sha256"], "Checkpoint changed: " + str(path)
            row["methods"][method] = {"path": str(path), "resolved_path": resolved,
                                     "sha256": checksum, "bytes": path.stat().st_size}
        assert row["methods"]["diffpano"]["sha256"] == row["methods"]["official"]["sha256"]
        row["identical"] = True
        checked.append(row)
    result = {"all_shared_files_identical": True, "files": checked, "verified_utc": now(),
              "job_id": os.environ["SLURM_JOB_ID"], "historical_evidence": str(evidence_path),
              "historical_evidence_sha256": sha(evidence_path),
              "comparison": "Full SHA256 of every shared diffusers pipeline JSON/weight file in pinned mirror and official snapshots; official snapshot feeds SphereDiff and SpheRoPE."}
    write(Path(run) / "generation/checkpoint_equivalence.json", result)
    return {"all_shared_files_identical": True, "files": len(checked)}


def environment(run):
    directory = Path(run) / "generation/environment"
    wheels = directory / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    env = {key: value for key, value in os.environ.items() if key != "OPENROUTER_API_KEY"}
    subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
                    "--dest", str(wheels)] + list(PACKAGES), check=True, env=env)
    artifacts = []
    for wheel in sorted(wheels.glob("*.whl")):
        artifacts.append({"path": str(wheel.resolve()), "sha256": sha(wheel)})
    value = {"packages": list(PACKAGES), "artifacts": artifacts,
             "base_environment": "studies/all_prompts/environment.sh diffpano",
             "isolation": "Pinned wheels expand into process-local SLURM_TMPDIR or /tmp, cleaned at normal exit; existing environments untouched"}
    write(Path(run) / "generation/baseline_environment.json", value)
    return value


def imports(run):
    from .baselines import import_sphe
    cls = import_sphe(run)
    import inspect
    required = {"sfc_rope", "enable_circular_encoding", "semantic_distortion_cfg", "semantic_distortion_gamma", "no_cfg_until_timestep"}
    assert required <= set(inspect.signature(cls.__call__).parameters)
    result = {"passed": True, "pipeline_class": cls.__name__, "pipeline_source": inspect.getfile(cls),
              "versions": {name: importlib.metadata.version(name) for name in
                           ("diffusers", "transformers", "tokenizers", "huggingface-hub", "torch", "accelerate")},
              "source_evidence": source_evidence(run)}
    write(Path(run) / "generation/baseline_import_validation.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("environment", "imports", "weights", "sources", "all", "launch"))
    parser.add_argument("--run", default=str(RUN))
    args = parser.parse_args(argv)
    if args.command == "launch":
        from .resources import submit
        result = submit(args.run, "baseline-validation", ROOT / "studies/paper_eval_v2/baseline_validation.slurm",
                        role="cpu", script_args=("--run", args.run), reserve_su=4.)
        print(json.dumps(result, indent=2))
        return
    if args.command == "all":
        checks = {}
        for name, function in (("environment", environment), ("imports", imports), ("weights", weight_audit)):
            try:
                function(args.run)
                checks[name] = {"state": "passed"}
            except Exception as error:
                checks[name] = {"state": "failed", "error": type(error).__name__ + ": " + str(error)}
            write(Path(args.run) / "generation/baseline_validation.json", {"checks": checks, "updated_utc": now()})
        if any(value["state"] != "passed" for value in checks.values()):
            raise RuntimeError("One or more baseline preparation checks failed; see generation/baseline_validation.json")
        result = checks
    else:
        result = {"environment": environment, "imports": imports, "weights": weight_audit, "sources": source_evidence}[args.command](args.run)
    print(json.dumps(result if args.command == "weights" else {"command": args.command, "state": "complete"}, indent=2))


if __name__ == "__main__":
    main()
