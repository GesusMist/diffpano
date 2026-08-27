#!/usr/bin/env python3
"""Run the experimental 2D no-warp patch ablation."""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf

from diffpano.config import load_experiment_config
from diffpano.initialization import load_directional_prompts, set_random_seed
from diffpano.metadata import save_run_metadata
from diffpano.pipelines import precision_dtype
from diffpano.pipelines.base import resolve_model_source
from experiments.planar.fusion import PlanarPatchFusionConfig
from experiments.planar.pipeline import PlanarPatchSanaPipeline
from experiments.planar.model_pipelines import PlanarPatchFluxPipeline, PlanarPatchSD2Pipeline

PLANAR_PIPELINES = {
    "planar_sana": PlanarPatchSanaPipeline,
    "planar_flux": PlanarPatchFluxPipeline,
    "planar_sd2": PlanarPatchSD2Pipeline,
}


def run(config):
    config.validate()
    if config.model.pipeline not in PLANAR_PIPELINES:
        raise ValueError(f"Unsupported planar pipeline: {config.model.pipeline}")
    prompts = load_directional_prompts(config.prompt.path)
    config.resolved_prompts = prompts
    set_random_seed(config.experiment.seed)
    planar_config = dict(config.planar)
    planar_config.setdefault("random_seed", config.experiment.seed)
    PlanarPatchFusionConfig.from_any(planar_config)
    pipeline_cls = PLANAR_PIPELINES[config.model.pipeline]

    run_id = config.output.run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.environ.get('SLURM_JOB_ID', 'local')}"
    run_dir = Path(config.output.directory) / config.experiment.name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(config.to_dict()), run_dir / "config.yaml")
    (run_dir / "run.log").write_text(f"started_at={datetime.now().isoformat()}\n", encoding="utf-8")

    source = resolve_model_source(config.model.path, config.model.id)
    pipe = pipeline_cls.from_pretrained(
        source,
        revision=config.model.revision,
        variant=config.model.variant,
        torch_dtype=precision_dtype(config.model.precision),
        **config.model.additional_pipeline_kwargs,
    )
    if config.model.pipeline == "planar_sd2":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if config.model.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required when model.cpu_offload=false")
        pipe.to(torch.device("cuda"), dtype=precision_dtype(config.model.precision))
    if pipe.scheduler.config.get("solver_order", 1) > 1:
        pipe.scheduler.config.solver_order = 1

    call_kwargs = dict(
        prompt_txt_path=config.prompt.path,
        negative_prompt_txt_path=config.prompt.negative_path or "",
        num_inference_steps=config.generation.num_inference_steps,
        guidance_scale=config.generation.guidance_scale,
        height=config.generation.height,
        width=config.generation.width,
        use_resolution_binning=False,
        planar_fusion_config=planar_config,
    )
    call_kwargs.update(config.generation.additional_call_kwargs)
    output = pipe(**call_kwargs)
    result_path = run_dir / "result.png"
    if config.output.save_final:
        output.images[0].save(result_path)
    if config.output.save_metadata:
        save_run_metadata(str(run_dir / "metadata.json"), config, pipe, call_kwargs, str(result_path))
    with (run_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"completed_at={datetime.now().isoformat()}\nresult={result_path}\n")
    print(f"Planar ablation saved to {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated planar patch ablation.")
    parser.add_argument("--config", default="experiments/planar/config.yaml")
    args = parser.parse_args()
    run(load_experiment_config(args.config))


if __name__ == "__main__":
    main()
