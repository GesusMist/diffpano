#!/usr/bin/env python3
"""Run native/fixed-initial-noise/model-implied single-patch controls."""
import argparse
import csv
import hashlib
import json
import os
import platform
import time

import torch
from omegaconf import OmegaConf

from diffpano.config import load_experiment_config
from diffpano.diagnostics import tensor_to_pil
from diffpano.initialization import set_random_seed, load_directional_prompts
from diffpano.planar_pipeline import _negative_prompt
from diffpano.native_multidiffusion import prepare_native_backend
from diffpano.pipelines import build_view_denoiser
from diffpano.trajectory import compare_three_way
from scripts.generate import _configure_denoiser, _run_directory


def run(config):
    config.validate()
    config.output.group = 'endpoint-controls'
    config.experiment.name += '-threeway'
    set_random_seed(config.experiment.seed)
    snapshot = config.to_dict()
    destination = _run_directory(config)
    destination.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(snapshot), destination/'config.yaml')
    started = time.perf_counter()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    backend = build_view_denoiser(config)
    _configure_denoiser(config, backend)
    prepared = prepare_native_backend(config, backend)
    conditioning = backend.conditioning_for_prompt_indices(prepared, [8], batch_size=config.generation.batch_size)
    p = config.native_multidiffusion.patch_size
    epsilon = torch.randn(config.generation.batch_size, backend.native_channels, p, p,
                          generator=torch.Generator(device='cpu').manual_seed(config.experiment.seed)).to(backend.device)
    result = compare_three_way(backend, conditioning, epsilon)
    if config.to_dict() != snapshot:
        raise AssertionError('Config changed during three-way comparison')
    if any(n != config.generation.num_inference_steps for n in result.model_evaluations.values()):
        raise AssertionError('Actual model evaluation count differs from configured steps')
    images = {}
    for name, tensor in result.finals.items():
        folder = destination/name
        folder.mkdir()
        path = folder/(name+'_final.png')
        tensor_to_pil(tensor[0]).save(path)
        images[name] = str(path.relative_to(destination))
    epsilon_cpu = result.initial_epsilon.cpu().contiguous()
    torch.save(epsilon_cpu, destination/'initial_epsilon.pt')
    if torch.cuda.is_available(): torch.cuda.synchronize()
    scheduler = getattr(getattr(backend, 'pipeline', None), 'scheduler', None)
    mean_implied = sum(s['native_vs_implied_mae'] for s in result.steps)/len(result.steps)
    mean_fixed = sum(s['native_vs_fixed_mae'] for s in result.steps)/len(result.steps)
    metadata = dict(schema_version=1, config=snapshot, images=images,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), node=platform.node(),
        runtime_seconds=time.perf_counter()-started,
        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else None,
        peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/1024**3 if torch.cuda.is_available() else None,
        fairness=result.fairness, model_evaluations=result.model_evaluations,
        evaluation_unit='guided prediction; same CFG forwards per prediction across A/B/C',
        diagnostic_extra_model_evaluations=0,
        epsilon_sha256=hashlib.sha256(epsilon_cpu.numpy().tobytes()).hexdigest(),
        native_shape=list(epsilon_cpu.shape), actual_precision=str(backend.dtype),
        prompt=load_directional_prompts(config.prompt.path)[2],
        negative_prompt=_negative_prompt(config) or getattr(backend, 'negative_prompt', ''),
        checkpoint=getattr(backend, 'checkpoint_path', None) or config.model.path or config.model.id,
        official_commit=getattr(backend, 'official_commit', None),
        scheduler_class=type(scheduler).__name__ if scheduler is not None else 'PixelDiT official first-order DPM',
        scheduler_config=dict(scheduler.config) if scheduler is not None else None,
        scheduler_timesteps=backend.timesteps.cpu().tolist(),
        scheduler_sigmas=scheduler.sigmas.cpu().tolist() if scheduler is not None and hasattr(scheduler, 'sigmas') else None,
        pixeldit_flow_schedule=backend.solver.schedule.cpu().tolist() if hasattr(backend, 'solver') else None,
        flow_shift=getattr(getattr(backend, 'solver', None), 'flow_shift', None),
        scheduler_shift_mu=getattr(backend, 'scheduler_shift_mu', None),
        scheduler_image_seq_len=getattr(backend, 'scheduler_image_seq_len', None),
        one_step_summary=dict(native_implied_mae=mean_implied, native_fixed_mae=mean_fixed,
                              ratio_epsilon=1e-12, ratio=mean_fixed/max(mean_implied, 1e-12)),
        steps=result.steps)
    (destination/'trajectory.json').write_text(json.dumps(metadata, indent=2)+'\n')
    fields = list(dict.fromkeys(key for step in result.steps for key in step))
    with (destination/'steps.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(result.steps)
    print(json.dumps({'destination': str(destination), 'counts': result.model_evaluations,
                      'one_step_summary': metadata['one_step_summary']}, indent=2), flush=True)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    run(load_experiment_config(parser.parse_args().config))
