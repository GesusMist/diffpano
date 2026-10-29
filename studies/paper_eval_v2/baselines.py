"""Pinned official baseline adapters and unchanged DiffPano scientific operators."""
from contextlib import contextmanager
from dataclasses import replace
import importlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import random
import sys
import time

from .common import ROOT, read, write, sha

MIRROR_ID = "ModelsLab/flux.1-dev"
MIRROR_REVISION = "fa45a9eb6808ba8fdfc7cc2756f7f1a16e0921f4"
OFFICIAL_ID = "black-forest-labs/FLUX.1-dev"
OFFICIAL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
SPHEROPE_COMMIT = "a40a86b49c25dbd4d0191591f28a0d5baf0c4ca4"
DIT360_COMMIT = "3779fe7965473f6824994c663a0ae7a76bc7aafa"
MIRROR_SNAPSHOT = Path("/scratch/user/shig/diffpano/hf_cache/hub/models--ModelsLab--flux.1-dev/snapshots") / MIRROR_REVISION
OFFICIAL_SNAPSHOT = Path("/scratch/user/shig/SphereDiff/hf_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots") / OFFICIAL_REVISION


def specifications():
    return {
        "select": {"backend": "flux", "algorithm": "DiffPano Old89 GWTF screened poisson_select", "steps": 20,
                   "width": 4096, "height": 2048, "guidance_scale": 3.5, "true_cfg_scale": 1., "refinement": 0,
                   "camera_count": 89, "camera_layout": "old89", "local_size": 1024, "fov": 80,
                   "gradient_mode": "poisson_select", "reference_mode": "screened", "lambda_color": .1,
                   "relative_tolerance": 1e-5, "absolute_tolerance": 1e-7, "max_iterations": 200,
                   "checkpoint": MIRROR_ID, "revision": MIRROR_REVISION, "precision": "bfloat16",
                   "prompt_format": "five_directional_lines", "model_cpu_offload": False},
        "sphere20": {"backend": "flux", "algorithm": "unmodified official SphereDiff SphericalFluxPipeline",
                     "steps": 20, "width": 4096, "height": 2048, "guidance_scale": 3.5, "true_cfg_scale": 1.,
                     "n_spherical_points": 26500, "weighted_average_temperature": .1,
                     "checkpoint": OFFICIAL_ID, "revision": OFFICIAL_REVISION, "precision": "bfloat16",
                     "prompt_format": "five_directional_lines", "model_cpu_offload": False,
                     "source_commit": "2c8c68ba088f2803b3dce4b52b7b0d68bc996139"},
        "spherope": {"backend": "flux", "algorithm": "official SpheRoPE ERPFluxPipeline", "steps": 50,
                     "width": 2048, "height": 1024, "guidance_scale": 3.5, "sfc_rope": True,
                     "enable_circular_encoding": True, "semantic_distortion_cfg": True,
                     "semantic_distortion_gamma": 6., "no_cfg_until_timestep": 2, "enable_wrapped_noise": False,
                     "checkpoint": OFFICIAL_ID, "revision": OFFICIAL_REVISION, "precision": "bfloat16",
                     "prompt_format": "global_description", "model_cpu_offload": True,
                     "source_commit": SPHEROPE_COMMIT,
                     "setting_label": "paper_quality_FLUX1_2048x1024_50",
                     "documented_override": "Author public FLUX1 defaults are 1024x512/28; requested paper-quality resolution/steps are explicit.",
                     "public_fast_control": {"width": 1024, "height": 512, "steps": 28, "selected": False}},
        "dit360": {"backend": "flux", "algorithm": "official DiT360 trained panorama LoRA", "steps": 28,
                   "width": 2048, "height": 1024, "guidance_scale": 2.8, "precision": "float16",
                   "checkpoint": OFFICIAL_ID, "revision": OFFICIAL_REVISION,
                   "lora_repo": "Insta360-Research/DiT360-Panorama-Image-Generation", "prompt_prefix": "This is a panorama. ",
                   "prompt_format": "global_description", "model_cpu_offload": False, "source_commit": DIT360_COMMIT,
                   "trained": True},
        "lf_multidiffusion": {"state": "blocked_upstream_release", "official_project": "https://ahykw.github.io/lfmd/",
                              "reason": "Official project says code coming soon; no speculative adapter labeled official."},
    }


def scientific_configuration(row):
    from studies.all_prompts.audit import configuration
    from diffpano.refinement import RefinementConfig
    config, _, cameras, slots, old = configuration(row["backend"], row["prompt_path"])
    config = replace(config, experiment=replace(config.experiment, seed=row["seed"]))
    config.global_pipeline.refinement = RefinementConfig(0.)
    assert config.generation.num_inference_steps == 20 and len(cameras) == 89
    return config, cameras, slots, old


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def use_baseline_environment(run):
    """Expand pinned wheels into job-local scratch, preserving home/scratch inodes."""
    import atexit
    import shutil
    import tempfile
    import zipfile
    evidence = read(Path(run) / "generation/baseline_environment.json")
    for item in evidence["artifacts"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Baseline dependency artifact changed")
    temporary = Path(tempfile.mkdtemp(prefix="paper-eval-baseline-", dir=os.environ.get("SLURM_TMPDIR", "/tmp")))
    atexit.register(shutil.rmtree, temporary, ignore_errors=True)
    for item in evidence["artifacts"]:
        if not str(item["path"]).endswith(".whl"):
            continue
        with zipfile.ZipFile(item["path"]) as archive:
            for name in archive.namelist():
                path = Path(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe pinned wheel member")
            archive.extractall(temporary)
    sys.path.insert(0, str(temporary))
    return evidence


def import_sphe(run):
    use_baseline_environment(run)
    source = Path(run) / "external/SpheRoPE/source"
    sys.path.insert(0, str(source))
    module = importlib.import_module("panorama_diffusers.pipelines.pipeline_flux_erp")
    if source.resolve() not in Path(module.__file__).resolve().parents:
        raise ValueError("SpheRoPE import did not resolve to the pinned official source")
    return module.ERPFluxPipeline


@contextmanager
def pipeline_counts(pipeline):
    """Logging wrappers return identical values and restore original functions."""
    counters = {"denoiser": 0, "vae_encode": 0, "vae_decode": 0, "transformer_input_shapes": {}}
    saved = []
    for owner, attribute, counter in ((pipeline.transformer, "forward", "denoiser"),
                                      (pipeline.vae, "encode", "vae_encode"), (pipeline.vae, "decode", "vae_decode")):
        original = getattr(owner, attribute)
        def wrapped(*args, _original=original, _counter=counter, **kwargs):
            counters[_counter] += 1
            if _counter == "denoiser":
                value = kwargs.get("hidden_states", args[0] if args else None)
                if value is not None:
                    shape = str(list(value.shape))
                    counters["transformer_input_shapes"][shape] = counters["transformer_input_shapes"].get(shape, 0) + 1
            return _original(*args, **kwargs)
        saved.append((owner, attribute, original))
        setattr(owner, attribute, wrapped)
    try:
        yield counters
    finally:
        for owner, attribute, original in saved:
            setattr(owner, attribute, original)


def run_select(row, directory):
    import numpy as np
    import torch
    from diffpano.pipelines import build_view_denoiser
    from scripts.generate import _configure_denoiser
    from diffpano.initialization import load_directional_prompts
    from diffpano.planar_pipeline import _negative_prompt
    from diffpano.dense_consensus import prepare_camera_conditioning
    from diffpano.gradient_fusion import GradientSettings
    from diffpano.erp_noise_initialization import primary_noise_size, native_cameras
    from diffpano.gwtf_noise_initialization import initialize_gwtf_shared_noise, GWTFlowNoiseConfig
    from studies.gradient_blending.operator import StudyPipeline
    from studies.gradient_blending.run import benchmark_schedule
    from studies.gradient_blending.diagnostics import save_rgb
    from studies.all_prompts.runtime import ordinary_run
    from studies.gwtf_noise.run import runtime_audit, count_calls
    from studies.original_spherediff.common import safe
    started = time.perf_counter()
    config, cameras, slots, historical = scientific_configuration(row)
    seed_all(row["seed"])
    backend = build_view_denoiser(config)
    _configure_denoiser(config, backend)
    table = benchmark_schedule(backend, config)
    bank = backend.prepare_prompt_conditioning(load_directional_prompts(row["prompt_path"]), _negative_prompt(config))
    conditions, prompt_slots = prepare_camera_conditioning(backend, bank, cameras, config.dense_consensus.prompt_assignment, 1)
    del bank
    settings = GradientSettings(mode="poisson_select", reference_mode="screened", lambda_color=.1,
                                relative_tolerance=1e-5, absolute_tolerance=1e-7, max_iterations=200)
    pipeline = StudyPipeline(backend, cameras, config, settings, slots)
    provenance, audit = runtime_audit(config, backend, pipeline, conditions, prompt_slots, historical)
    assert set(audit["differences"]) <= {"conditioning_sha256", "prompt_indices"}, audit["differences"]
    height, width = primary_noise_size(native_cameras(backend, cameras))
    with count_calls(backend, row["backend"], forbid=True) as initialization_counts:
        states, initialization = initialize_gwtf_shared_noise(backend, cameras, GWTFlowNoiseConfig(height, width, row["seed"]),
                                                              1, camera_slots=slots)
    assert initialization_counts == {"denoiser": 0, "encode": 0, "decode": 0, "initialize": 89}
    if row["seed"] == 0:
        assert initialization["initial_local_sha256"] == historical["initialization"]["initial_local_sha256"]
    assert initialization["scaling_applications"] == 1
    setup_seconds = time.perf_counter() - started
    setup_peak = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    def progress(step, total, record):
        fusion = pipeline.canvas.records[-1]
        assert fusion["converged"] and fusion["reference_mode"] == "screened"
        assert record["min_contributors"] > 0 and record["minimum_weight_sum"] > 0
        write(directory / "progress.json", safe({"step": step, "total": total, "record": record,
                                                 "fusion": fusion, "elapsed_seconds": time.perf_counter() - started}))
        print("PAPER_EVAL_GENERATION", row["case_id"], step, total, flush=True)
    with torch.no_grad(), count_calls(backend, row["backend"]) as counts:
        holder = [states]
        del states
        _, erp, details = ordinary_run(pipeline, holder.pop(), conditions, table, progress)
    torch.cuda.synchronize()
    duration = time.perf_counter() - started
    assert counts == {"denoiser": 1780, "encode": 3560, "decode": 1869, "initialize": 0}, counts
    assert len(pipeline.canvas.records) == 21 and pipeline.canvas.records[-1]["terminal"]
    assert all(item["converged"] for item in pipeline.canvas.records)
    assert details["backward_calls"] == details["replay_count"] == details["original_noise_reinjections"] == 0
    assert tuple(erp.shape) == (1, 3, 2048, 4096) and bool(torch.isfinite(erp).all())
    save_rgb(directory / "final.png", erp)
    stage_seconds = {}
    for interval in pipeline.interval_records:
        for stage, elapsed in interval["stage_seconds"].items():
            stage_seconds[stage] = stage_seconds.get(stage, 0) + elapsed
    stage_seconds.update(details["terminal_seconds"])
    return safe({"configuration": config.to_dict(), "provenance": provenance, "audit": audit,
                 "initialization": initialization, "initialization_counts": initialization_counts,
                 "counts": counts, "details": details, "fusions": pipeline.canvas.records,
                 "intervals": pipeline.interval_records, "stage_seconds": stage_seconds,
                 "generation_seconds": duration, "setup_seconds": setup_seconds,
                 "peak_allocated_gib": max(setup_peak, torch.cuda.max_memory_allocated()) / 2**30,
                 "generation_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})


def run_official(row, directory, run):
    # The modern SpheRoPE environment must enter sys.path before importing torch/diffusers pipelines.
    if row["method_id"] == "spherope":
        pipeline_class = import_sphe(run)
    elif row["method_id"] == "dit360":
        use_baseline_environment(run)
        sys.path.insert(0, str(Path(run) / "external/DiT360/source"))
        pipeline_class = importlib.import_module("src.pipeline").DiT360Pipeline
    else:
        from studies.original_spherediff.common import import_official
        classes, imports = import_official()
        pipeline_class = classes[row["backend"]]
    import torch
    from studies.original_spherediff.common import safe, static_solver, scheduler_record, specs
    seed_all(row["seed"])
    started = time.perf_counter()
    dtype = torch.float16 if row["method_id"] == "dit360" else torch.bfloat16
    snapshot = OFFICIAL_SNAPSHOT if row["backend"] == "flux" else None
    if snapshot is None:
        from studies.original_spherediff.common import snapshot as original_snapshot
        snapshot = original_snapshot(row["backend"])
    pipeline = pipeline_class.from_pretrained(str(snapshot), torch_dtype=dtype, local_files_only=True,
               **({"variant": "bf16"} if row["backend"] == "sana" else {}))
    if row["method_id"] == "dit360":
        lora = read(Path(run) / "generation/dit360_checkpoint.json")
        pipeline.load_lora_weights(lora["local_path"], local_files_only=True)
    if row["method_id"] == "spherope":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(torch.device("cuda"), dtype=dtype)
    original_order = static_solver(pipeline.scheduler)
    spec = row["specification"]
    if row["method_id"].startswith("sphere"):
        call = dict(specs()[row["backend"]]["call"], num_inference_steps=20,
                    prompt_txt_path=row["prompt_path"], generator=torch.Generator("cuda").manual_seed(row["seed"]))
    elif row["method_id"] == "spherope":
        call = {key: spec[key] for key in ("width", "height", "guidance_scale", "sfc_rope", "enable_circular_encoding",
                  "semantic_distortion_cfg", "semantic_distortion_gamma", "no_cfg_until_timestep", "enable_wrapped_noise")}
        call.update(prompt=row["global_generation_description"], num_inference_steps=50,
                    generator=torch.Generator("cpu").manual_seed(row["seed"]))
    else:
        call = {"prompt": spec["prompt_prefix"] + row["global_generation_description"], "width": 2048, "height": 1024,
                "num_inference_steps": 28, "guidance_scale": 2.8, "generator": torch.Generator("cuda").manual_seed(row["seed"])}
    def progress(pipe, step, timestep, kwargs):
        write(directory / "progress.json", {"step": step + 1, "steps": spec["steps"],
                                             "elapsed_seconds": time.perf_counter() - started})
        print("PAPER_EVAL_GENERATION", row["case_id"], step + 1, spec["steps"], flush=True)
        return kwargs
    call["callback_on_step_end"] = progress
    setup_seconds = time.perf_counter() - started
    setup_peak = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad(), pipeline_counts(pipeline) as counts:
        output = pipeline(**call)
    torch.cuda.synchronize()
    duration = time.perf_counter() - started
    image = output.images[0]
    assert image.size == (spec["width"], spec["height"])
    if row["method_id"].startswith("sphere"):
        assert counts["denoiser"] == 1780
    assert counts["vae_decode"] > 0
    image.save(directory / "final.png")
    return safe({"pipeline_class": pipeline_class.__name__, "pipeline_source": inspect.getfile(pipeline_class),
                 "checkpoint_path": str(snapshot), "specification": spec, "counts": counts,
                 "scheduler": scheduler_record(pipeline.scheduler, original_order),
                 "generation_seconds": duration, "setup_seconds": setup_seconds,
                 "peak_allocated_gib": max(setup_peak, torch.cuda.max_memory_allocated()) / 2**30,
                 "generation_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "environment": {name: importlib.metadata.version(name) for name in
                                 ("torch", "diffusers", "transformers", "accelerate")},
                 "model_cpu_offload": row["method_id"] == "spherope", "vae_tiling": bool(getattr(pipeline.vae, "use_tiling", False)),
                 "timing_scope": "CUDA-synchronized full official pipeline call, including warp/fusion/VAE/offload; loading separately"})
