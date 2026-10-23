"""The established Old89 screened Select configuration with two explicit changes."""
from dataclasses import asdict, replace

from studies.followup24.common import prompt, prompt_path
from studies.refinement28.common import ROOT, read, sha, safe

SCHEDULE_SOURCE = ROOT / 'outputs/10.1flux-step-controls-seed0/diffpano/flux/erp/steps28/air_balloons/config.json'


def resolved(row):
    from diffpano.refinement import RefinementConfig, independent_count
    from studies.layout24.configuration import resolved as baseline_configuration
    from studies.gwtf_erp4k.common import differences

    n = row['refinement_steps']
    assert n in (5, 8) and type(n) is int
    assert row['steps'] == 28 and row['coupled_steps'] == 28 - n
    assert row['last_fraction'] == n / 28
    assert row['layout'] == 'old89' and row['camera_count'] == 89
    assert row['backend'] == 'flux' and row['seed'] == 0 and row['projection'] == 'erp'
    assert row['gradient_mode'] == 'poisson_select' and row['reference_mode'] == 'screened'
    assert row['prompt_path'] == str(prompt_path(row['prompt']))
    base, cameras, owners, slots, settings, geometry, old = baseline_configuration(row)
    config = replace(base, generation=replace(base.generation, num_inference_steps=28),
                     global_pipeline=replace(base.global_pipeline, refinement=RefinementConfig(n / 28)))
    # Historical to_dict intentionally omits zero refinement. Compare full
    # dataclass fields so enabling it remains an explicit value change.
    changes = differences(asdict(base), asdict(config))
    assert {change['path'] for change in changes} == {
        'generation.num_inference_steps', 'global_pipeline.refinement.last_fraction'}
    assert independent_count(28, config.global_pipeline.refinement.last_fraction) == n
    assert settings.mode == 'poisson_select' and settings.reference_mode == 'screened'
    assert settings.lambda_color == .1 and settings.gradient_mix_alpha is None
    assert settings.relative_tolerance == 1e-5 and settings.absolute_tolerance == 1e-7
    assert settings.max_iterations == 200
    assert not config.sampling.rotation.enabled
    assert geometry['angular_sha256'] == old['camera_geometry_sha256']
    assert owners == slots == list(range(89))
    return config, cameras, owners, slots, settings, geometry, old


def schedule_evidence():
    """Reuse a verified actual 28-step schedule, rechecking it on every GPU run."""
    from studies.flux_step_controls.common import schedule_valid
    from studies.gwtf_erp4k.common import differences

    record = read(SCHEDULE_SOURCE)
    assert record['state'] == 'complete'
    assert record['row']['steps'] == 28 and record['row']['backend'] == 'flux'
    assert record['initialization_match'] and not record['time_travel_enabled']
    assert record['schedule'] == record['runtime_provenance']['prepared_schedule']
    schedule_valid(record['schedule'], 28)
    # Compare the saved configuration against the same baseline, including model,
    # guidance, geometry-related settings, and the entire scheduler configuration.
    from studies.all_prompts.audit import configuration
    baseline, _, _, _, old = configuration('flux', record['configuration']['prompt']['path'])
    changes = differences(baseline.to_dict(), record['configuration'])
    assert changes == [{'path': 'generation.num_inference_steps', 'before': 20, 'after': 28}]
    for key in ('class_name', 'config'):
        assert record['schedule'][key] == old['prepared_schedule'][key]
    assert record['initialization']['initial_local_sha256'] == old['initialization']['initial_local_sha256']
    return safe(record['schedule']), dict(path=str(SCHEDULE_SOURCE), sha256=sha(SCHEDULE_SOURCE),
                                        verified=True, configuration_differences=changes)


def semantic(row, schedule):
    config, _, _, _, settings, geometry, old = resolved(row)
    n, coupled = row['refinement_steps'], row['coupled_steps']
    expected = {key: safe(old['initialization'][key]) for key in (
        'initial_local_sha256', 'source_sha256', 'source_shape', 'scaling', 'scaling_applications')}
    return safe(dict(configuration=asdict(config), geometry=geometry, solver=settings.to_record(),
                     prompt=prompt(row['prompt'])[0], schedule=schedule, expected_initialization=expected,
                     refinement=dict(last_fraction=n / 28, total_intervals=28,
                                     coupled_intervals=coupled, independent_intervals=n,
                                     transition_step=coupled + 1,
                                     transition_timestep=float(schedule['timesteps'][coupled])),
                     model=dict(id=config.model.id, revision=config.model.revision),
                     initializer='unchanged saved Old89 GWTF, seed 0',
                     case={key: value for key, value in row.items() if key not in ('index', 'case_id')}))
