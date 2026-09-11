#!/usr/bin/env python3
"""Experiment H: change G's RGB fusion only; reuse every saved control setting."""
import argparse
import copy
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from diffpano.config import load_experiment_config
from diffpano.implied_endpoint_consensus import PlanarImpliedEndpointConsensusPipeline
from diffpano.initialization import set_random_seed
from diffpano.metadata import save_run_metadata
from diffpano.native_multidiffusion import prepare_native_backend
from diffpano.pipelines import build_view_denoiser
from diffpano.trajectory import conditioning_digest
from diffpano.vae_residual import identity_vae_preflight
from scripts.generate import _configure_denoiser
from scripts.paired_rgb_endpoint import digest, timed, save_result


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def check_configs(spec):
    base = load_experiment_config(spec['base_config'])
    config = load_experiment_config(spec['config'])
    reference = read(spec['reference'])
    if base.to_dict() != reference['config']:
        raise AssertionError('Saved baseline config changed')
    expected = copy.deepcopy(base.to_dict())
    if expected['fusion']['mode'] != 'average':
        raise AssertionError('Expected arithmetic-average baseline')
    expected['fusion']['mode'] = 'detail_preserving_average'
    if config.to_dict() != expected:
        raise AssertionError('H must differ from G only in fusion.mode')
    if config.model.pipeline != spec['backend']:
        raise AssertionError('Backend mismatch')
    return config, reference


def reference_facts(spec, reference):
    if spec['backend'] == 'pixeldit':
        return reference['fairness']['conditioning_sha256'], reference['model_evaluations']['implied_endpoint_consensus']
    if spec['backend'] == 'sd35':
        return reference['shared']['conditioning_sha256'], reference['guided_predictions']
    return reference['conditioning_sha256'], reference['guided_predictions']


def check_runtime(current, baseline):
    keys = ['model', 'scheduler', 'scheduler_timesteps', 'scheduler_sigmas',
            'pixeldit_flow_schedule', 'native_geometry']
    if baseline.get('backend_details') is not None:
        keys.append('backend_details')
    for key in keys:
        if current.get(key) != baseline.get(key):
            raise AssertionError('H/G runtime mismatch: '+key)
    for key in ('python', 'torch', 'cuda'):
        if current['environment'][key] != baseline['environment'][key]:
            raise AssertionError('H/G software mismatch: '+key)
    return keys


def run(spec):
    config, reference = check_configs(spec)
    folder = Path(spec['output'])
    folder.mkdir(parents=True, exist_ok=False)
    write(folder/'spec.json', spec)
    started = time.perf_counter()
    if torch.cuda.get_device_name() != spec['gpu']:
        raise AssertionError('H requires the saved A40 control GPU type')
    set_random_seed(config.experiment.seed)
    backend = build_view_denoiser(config)
    _configure_denoiser(config, backend)
    prepared = prepare_native_backend(config, backend)
    conditioning = backend.conditioning_for_prompt_indices(prepared, [8], batch_size=config.generation.batch_size)
    expected_condition, expected_count = reference_facts(spec, reference)
    if conditioning_digest(conditioning) != expected_condition:
        raise AssertionError('H/G conditioning hash mismatch')
    baseline = read(Path(spec['reference_generation'])/'metadata.json')
    preflight_path = folder/'runtime_preflight.json'
    save_run_metadata(str(preflight_path), config, backend, SimpleNamespace(steps=[]), '')
    runtime_keys = check_runtime(read(preflight_path), baseline)
    initial = torch.load(spec['initial_native'], map_location='cpu', weights_only=True)
    initial_hash = digest(initial)
    if initial_hash != reference['initial_native_sha256']:
        raise AssertionError('H/G saved initial tensor hash mismatch')
    pixel_native = spec['backend'] == 'pixeldit'
    pipe = PlanarImpliedEndpointConsensusPipeline(native_config=config.native_multidiffusion,
        backend=backend, pixel_native=pixel_native, residual_correction=not pixel_native,
        fusion_config=config.fusion)
    if not pixel_native:
        shape = (initial.shape[0], backend.native_channels, pipe.native_layout.patch_size, pipe.native_layout.patch_size)
        write(folder/'real_vae_identity_preflight.json', identity_vae_preflight(backend, shape))
    local = pipe.initialize_local_states(initial.to(backend.device))
    torch.save(initial, folder/'initial_native.pt')
    del initial
    before = getattr(backend, 'guided_prediction_count', 0)
    result, timing = timed(lambda: pipe.run(local, prepared))
    actual_count = backend.guided_prediction_count-before
    if actual_count != expected_count or result.audit['guided_predictions'] != expected_count:
        raise AssertionError('H/G guided prediction count mismatch')
    if conditioning_digest(conditioning) != expected_condition:
        raise AssertionError('Conditioning mutated')
    result.audit['initial_local_states_equal_global_native_crops'] = True
    save_result(folder/'generation', config, backend, result, timing)
    check_runtime(read(folder/'generation/metadata.json'), baseline)
    write(folder/'comparison.json', dict(experiment='H', backend=spec['backend'], config=config.to_dict(),
        reference=spec['reference'], reference_generation=spec['reference_generation'], baseline_label=spec['baseline_label'],
        only_setting_changed='fusion.mode: average -> detail_preserving_average',
        same_initial_model_settings_schedule=True, runtime_fields_checked=runtime_keys,
        initial_native_source=spec['initial_native'], initial_native_sha256=initial_hash,
        conditioning_sha256=expected_condition, guided_predictions=actual_count,
        extra_diagnostic_denoiser_calls=0, training_free=True, audit=result.audit,
        geometry=dict(native=asdict(pipe.native_layout), rgb=asdict(pipe.rgb_layout), factor=backend.native_spatial_factor),
        timing=timing, total_seconds=time.perf_counter()-started,
        job=os.environ.get('SLURM_JOB_ID'), node=platform.node(), gpu=torch.cuda.get_device_name()))
    print('Completed H', spec['backend'], folder, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', default='configs/experiments/vae_residual/h-all-models.json')
    parser.add_argument('--backend', required=True, choices=['sd2', 'sana', 'flux', 'sd35', 'pixeldit'])
    args = parser.parse_args()
    run(read(args.spec)[args.backend])


if __name__ == '__main__':
    main()
