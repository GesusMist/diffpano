"""Audited bookkeeping around the unchanged 28-step DiffPano trajectory."""
from dataclasses import asdict
import fcntl
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from diffpano.refinement import independent_count, resolved_refinement
from studies.all_prompts.runtime import ordinary_run
from studies.followup24.common import prompt, prompt_path
from studies.gradient_blending.operator import StudyPipeline
from studies.gradient_blending.run import benchmark_schedule
from studies.refinement28.common import OUT, digest, expected_counts, read, safe, sha, write

ARTIFACT_NAMES = (
    'final.png', 'final-unclamped-fp32.pt',
    'terminal-rgb-reference.png', 'terminal-rgb-reference-fp32.pt', 'config.json',
)


def validate_execution(row, pipeline, details, counts):
    """Require the complete original schedule, exact native tail, and final solve."""
    steps = row['steps']
    tail_steps = row['refinement_steps']
    coupled = row['coupled_steps']
    assert steps == 28 and tail_steps in (5, 8) and coupled == steps-tail_steps
    assert row['last_fraction'] == tail_steps/steps
    assert independent_count(steps, row['last_fraction']) == tail_steps
    assert len(pipeline.cameras) == 89
    assert len(pipeline.backend.timesteps) == steps
    plan = resolved_refinement(pipeline.backend.timesteps, pipeline.refinement_config)
    assert plan == row['semantic']['refinement']
    assert plan['independent_intervals'] == tail_steps and plan['coupled_intervals'] == coupled
    assert counts == expected_counts(row), counts
    assert details['guided_predictions'] == counts['denoiser'] == 89*steps
    assert details['refinement'] == plan
    assert details['time_travel_enabled'] is False
    assert details['original_noise_bank_retained'] is False
    for key in ('backward_calls', 'replay_count', 'original_noise_reinjections', 'additional_denoising_cycles'):
        assert details[key] == 0, key
    assert details['active_consensus_projection'] == 'erp'
    assert len(pipeline.interval_records) == steps
    intervals = []
    for k, record in enumerate(pipeline.interval_records):
        assert record['k'] == k and record['pass_kind'] == 'initial'
        assert record['model_timestep'] == float(pipeline.backend.timesteps[k])
        independent = k >= coupled
        if independent:
            assert record.get('phase') == 'independent_refinement'
            assert record['fusion'] is None and record['guided_predictions'] == 89
            assert set(record['stage_seconds']) == {'independent_native'}
        else:
            assert record.get('phase', 'coupled') == 'coupled'
            assert record['min_contributors'] > 0 and record['minimum_weight_sum'] > 0
        intervals.append(dict(record, phase='independent_refinement' if independent else 'coupled'))
    tail = pipeline.refinement_tail
    assert tail is not None and tail.next_index == steps
    assert [record['k'] for record in tail.records] == list(range(coupled, steps))
    transition = details['refinement_transition']
    assert transition == tail.transition
    assert transition['original_index'] == coupled and transition['step'] == coupled+1
    assert transition['timestep'] == plan['transition_timestep']
    assert len(transition['native_state_sha256']) == 64
    fusions = pipeline.canvas.records
    assert len(fusions) == coupled+1
    assert [record['stage'] for record in fusions] == list(range(1, coupled+1))+['terminal']
    assert [record['terminal'] for record in fusions] == [False]*coupled+[True]
    for record in fusions:
        assert record['mode'] == row['gradient_mode'] == 'poisson_select'
        assert record['reference_mode'] == row['reference_mode'] == 'screened'
        assert record['converged'] and record['ownership_consistent']
        assert record.get('gradient_mix_alpha') is None
    return intervals, plan


def register_group(row, config):
    """Both fractions must use identical conditioning, noise, geometry, and schedule."""
    key = row.get('prompt_identity', row['prompt'])
    assert isinstance(key, str) and key
    relative = Path(key)
    assert not relative.is_absolute() and all(part not in ('.', '..') for part in relative.parts)
    directory = OUT/'runtime-groups'/relative
    directory.mkdir(parents=True, exist_ok=True)
    value = dict(conditioning_sha256=config['provenance']['conditioning_sha256'],
                 initial_local_sha256=config['initialization']['initial_local_sha256'],
                 source_noise_sha256=config['initialization']['source_sha256'],
                 geometry_sha256=config['geometry']['angular_sha256'],
                 schedule_sha256=config['schedule_sha256'])
    with (directory/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = directory/'identity.json'
        if target.exists():
            assert read(target) == value, 'Paired refinement conditioning/noise/geometry/schedule changed'
        else:
            write(target, value)
    return value


def _raw(image):
    return dict(minimum=float(image.min()), maximum=float(image.max()),
                out_of_range_fraction=float(((image < -1) | (image > 1)).float().mean()),
                mean_rgb=image.mean((0, 2, 3)).tolist())


def _publish(directory, erp, reference, config):
    """Publish only validated final products; completion is committed by run.py."""
    from studies.gradient_blending.diagnostics import save_rgb
    with tempfile.TemporaryDirectory(prefix='.staging-', dir=directory) as name:
        staging = Path(name)
        save_rgb(staging/'final.png', erp)
        torch.save(erp.detach().cpu().contiguous(), staging/'final-unclamped-fp32.pt')
        save_rgb(staging/'terminal-rgb-reference.png', reference)
        torch.save(reference.detach().cpu().contiguous(), staging/'terminal-rgb-reference-fp32.pt')
        write(staging/'config.json', config)
        hashes = {name: sha(staging/name) for name in ARTIFACT_NAMES}
        for name in ARTIFACT_NAMES:
            os.replace(staging/name, directory/name)
    return hashes


@torch.no_grad()
def generate(row, directory):
    """Generate one authorized case; the caller owns locks, freeze, and status."""
    from studies.refinement28.configuration import resolved
    from studies.original_spherediff.common import seed_global
    from studies.camera_patching.runtime import initialize
    from studies.gwtf_noise.run import runtime_audit, count_calls
    from studies.all_prompts.common import emit
    from diffpano.pipelines import build_view_denoiser
    from scripts.generate import _configure_denoiser
    from diffpano.initialization import load_directional_prompts
    from diffpano.planar_pipeline import _negative_prompt
    from diffpano.dense_consensus import prepare_camera_conditioning

    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available(), 'GPU Slurm allocation required'
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    c, cameras, owners, slots, settings, geometry, old = resolved(row)
    semantic = row['semantic']
    assert safe(asdict(c)) == semantic['configuration']
    assert safe(geometry) == semantic['geometry']
    assert settings.matches_record(semantic['solver'])
    info, _ = prompt(row['prompt'])
    assert info == semantic['prompt']
    assert c.generation.num_inference_steps == row['steps'] == 28
    assert c.global_pipeline.refinement.last_fraction == row['last_fraction'] == row['refinement_steps']/28
    assert len(cameras) == len(owners) == len(slots) == 89
    assert geometry['angular_sha256'] == old['camera_geometry_sha256']
    seed_global(torch, np)
    backend = build_view_denoiser(c)
    _configure_denoiser(c, backend)
    table = benchmark_schedule(backend, c)
    assert len(table) == len(backend.timesteps) == 28 and all(not item.eligible for item in table)
    bank = backend.prepare_prompt_conditioning(load_directional_prompts(str(prompt_path(row['prompt']))), _negative_prompt(c))
    conditions, prompt_slots = prepare_camera_conditioning(backend, bank, cameras, c.dense_consensus.prompt_assignment, 1)
    del bank
    assert list(prompt_slots) == geometry['route']['indices']
    pipeline = StudyPipeline(backend, cameras, c, settings, owners)
    assert pipeline.bridge_mode == 'local_identity_preserving'
    provenance, audit = runtime_audit(c, backend, pipeline, conditions, prompt_slots, old)
    allowed = {'prepared_schedule', 'conditioning_sha256', 'prompt_indices'}
    assert set(audit['differences']) <= allowed, audit['differences']
    audit.update(accepted=True, allowed_runtime_differences=sorted(allowed))
    assert provenance['camera_geometry_sha256'] == geometry['angular_sha256']
    assert safe(provenance['prepared_schedule']) == semantic['schedule']
    for key in ('class_name', 'config'):
        assert provenance['prepared_schedule'][key] == old['prepared_schedule'][key], key
    assert len(old['prepared_schedule']['timesteps']) == 20
    assert provenance['prepared_schedule']['timesteps'] != old['prepared_schedule']['timesteps']
    plan = resolved_refinement(backend.timesteps, pipeline.refinement_config)
    assert plan == semantic['refinement']
    init_start = time.perf_counter()
    with count_calls(backend, 'flux', forbid=True) as setup_counts:
        states, initialization = initialize(backend, 'flux', cameras, slots)
    initialization_seconds = time.perf_counter()-init_start
    assert setup_counts == dict(denoiser=0, encode=0, decode=0, initialize=89)
    assert initialization['initial_local_sha256'] == old['initialization']['initial_local_sha256']
    expected_initialization = semantic['expected_initialization']
    for key in ('initial_local_sha256', 'source_sha256', 'source_shape', 'scaling', 'scaling_applications'):
        if key in expected_initialization:
            assert safe(initialization[key]) == expected_initialization[key], key
    assert initialization['scaling_applications'] == 1
    config = safe(dict(configuration=asdict(c), backend='flux', method='diffpano', layout='old89',
        steps=28, refinement_steps=row['refinement_steps'], coupled_steps=row['coupled_steps'],
        prompt=info, geometry=geometry, fusion=settings.to_record(), reference_mode='screened',
        refinement=plan, terminal_fusion_mode='poisson_select', terminal_reference_mode='screened',
        provenance=provenance, audit=audit, initialization=initialization,
        setup_initialization_counts=setup_counts,
        initialization_audit=dict(seconds=initialization_seconds, calls=setup_counts,
                                  historical_initialization_match=True),
        camera_count=89, camera_ids=owners, noise_slots=slots,
        camera_geometry_sha256=provenance['camera_geometry_sha256'], FOV_x=80, FOV_y=80,
        local_RGB_resolution=[1024, 1024], local_native_resolution=[128, 128], ERP_resolution=[2048, 4096],
        schedule_sha256=digest(provenance['prepared_schedule']),
        boundary='horizontal periodic, vertical open; pixel-unit differences; binary edge support'))
    config['same_prompt_group_identity'] = register_group(row, config)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter()-started
    setup_peak = torch.cuda.max_memory_allocated()
    setup_reserved = torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    generation_start = time.perf_counter()

    def progress(step, total, record):
        assert total == 28 and record['k'] == step-1
        independent = step > row['coupled_steps']
        assert (record.get('phase') == 'independent_refinement') == independent
        if independent:
            assert record['fusion'] is None
            assert len(pipeline.canvas.records) == row['coupled_steps']
            fusion = None
        else:
            assert record['min_contributors'] > 0 and record['minimum_weight_sum'] > 0
            fusion = pipeline.canvas.records[-1]
            assert fusion['stage'] == step and fusion['converged']
            assert fusion['mode'] == 'poisson_select' and fusion['reference_mode'] == 'screened'
        current = dict(step=step, total=total, phase='independent_refinement' if independent else 'coupled',
                       elapsed_seconds=time.perf_counter()-generation_start, record=record, fusion=fusion)
        write(directory/'progress.json', safe(current))
        emit('REFINEMENT28_PROGRESS', case=row.get('case_id', row.get('key')), **safe(current))

    with count_calls(backend, 'flux') as counts:
        holder = [states]
        del states
        native, erp, details = ordinary_run(pipeline, holder.pop(), conditions, table, progress)
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter()-generation_start
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    intervals, plan = validate_execution(row, pipeline, details, counts)
    assert details['initial_state_sha256'] == initialization['initial_local_sha256']
    assert erp.shape == (1, 3, 2048, 4096) and erp.dtype == torch.float32 and bool(torch.isfinite(erp).all())
    reference = pipeline.canvas.last_reference
    assert reference.shape == erp.shape and reference.dtype == torch.float32 and bool(torch.isfinite(reference).all())
    assert native.rgb.dtype == torch.float32
    details = dict(details, terminal_semantics='Decode terminal local states; screened poisson_select assembly; export ERP',
                   terminal_fusion_mode='poisson_select', terminal_reference_mode='screened')
    artifacts = _publish(directory, erp, reference, config)
    stage_seconds = {}
    for record in intervals:
        for key, value in record['stage_seconds'].items():
            stage_seconds[key] = stage_seconds.get(key, 0.)+value
    stage_seconds.update(details['terminal_seconds'])
    return safe(dict(**config, counts=counts, details=details, intervals=intervals, fusions=pipeline.canvas.records,
        poisson_solves=row['coupled_steps']+1, generation_seconds=generation_seconds, setup_seconds=setup_seconds,
        total_seconds=time.perf_counter()-started,
        solver_seconds=sum(record.get('solve_seconds', 0.) for record in pipeline.canvas.records),
        stage_seconds=stage_seconds, final_raw=_raw(erp), terminal_reference_raw=_raw(reference),
        generation_peak_allocated_gib=peak/2**30, peak_allocated_gib=max(setup_peak, peak)/2**30,
        generation_peak_reserved_gib=reserved/2**30, peak_reserved_gib=max(setup_reserved, reserved)/2**30,
        artifacts=artifacts, gpu=torch.cuda.get_device_name(),
        terminal_reference_label='Weighted RGB assembly from this run terminal local proposals',
        terminal_reference_is_independent_baseline=False))
