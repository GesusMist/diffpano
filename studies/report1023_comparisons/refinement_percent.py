"""Authenticated 0/20/30 display-label sources; the zero baseline has 20 steps.

This collector reads retained PNGs and metadata. Previously authenticated full
artifact checks are reused; raw tensors are deliberately not read again.
"""
import copy
import struct
from pathlib import Path

from .refinement import GEOMETRY, ROOT, SOLVER, configuration, differences, digest, read, sha

REPORT = ROOT / 'outputs/10.23report'
BASELINE_KEY = 'diffpano_old89_screened_select'
REFINEMENT_KEYS = ('refine05of28', 'refine08of28')
LABELS = ('0% (no refinement)', '20%', '30%')


def _inventory(index, group, kind):
    directory = REPORT / group
    matches = [item for item in index['groups'] if item['directory'] == str(directory)]
    assert len(matches) == 1, group
    pinned = matches[0]
    manifest_path = directory / 'manifest.json'
    assert pinned['manifest'] == str(manifest_path)
    assert sha(manifest_path) == pinned['manifest_sha256']
    manifest = read(manifest_path)
    assert manifest['complete'] is True and manifest['all_source_hashes_preserved'] is True
    assert manifest['all_exported_panels_pixel_verified'] is True
    assert manifest['figure_count'] == pinned['figure_count'] == 24
    inventory_path = directory / 'provenance/sources.json'
    inventory_hash = sha(inventory_path)
    assert inventory_hash == manifest['source_inventory_sha256']
    inventory = read(inventory_path)
    assert inventory['kind'] == kind and len(inventory['figures']) == 24
    assert len({row['prompt'] for row in inventory['figures']}) == 24
    return inventory, dict(manifest_path=str(manifest_path),
        manifest_sha256=pinned['manifest_sha256'], source_inventory_path=str(inventory_path),
        source_inventory_sha256=inventory_hash)


def _source(panel):
    image = Path(panel['image_path'])
    metadata = Path(panel['metadata_path'])
    assert image.is_absolute() and metadata.is_absolute()
    assert sha(image) == panel['image_sha256'], str(image)
    assert sha(metadata) == panel['metadata_sha256'], str(metadata)
    with image.open('rb') as stream:
        header = stream.read(26)
    assert header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR'
    assert struct.unpack('>II', header[16:24]) == (4096, 2048)
    assert header[24:26] == b'\x08\x02', 'Expected an eight-bit RGB panorama'
    record = read(metadata)
    assert record['artifacts']['final.png'] == panel['image_sha256']
    return record


def _prompt(figure, metadata):
    path = Path(figure['prompt_path'])
    assert path.is_absolute() and sha(path) == figure['prompt_sha256']
    prompt = metadata['prompt']
    assert prompt['name'] == figure['prompt'] and prompt['adapted'] is False
    assert prompt['original_sha256'] == prompt['effective_sha256'] == figure['prompt_sha256']
    assert path.read_text().splitlines() == prompt['effective_lines']
    assert len(prompt['effective_lines']) == 5
    configured = Path(metadata['configuration']['prompt']['path'])
    assert configured == path


def _method(metadata, steps, tail):
    config = configuration(metadata['configuration'])
    assert config['generation']['num_inference_steps'] == steps
    assert config['global_pipeline']['refinement']['last_fraction'] == tail / steps
    assert config['experiment']['seed'] == 0 and config['model']['pipeline'] == 'flux'
    assert metadata['camera_geometry_sha256'] == GEOMETRY
    camera_ids = metadata.get('camera_ids')
    assert camera_ids == list(range(89)) if camera_ids is not None else metadata['camera_count'] == 89
    for key, value in SOLVER.items():
        assert metadata['fusion'][key] == value, key
    # Original 10.7 metadata predates this field. Its screened interpretation is
    # pinned by the authenticated uniform inventory and migration source record.
    assert metadata['fusion'].get('reference_mode', 'screened') == 'screened'
    coupled = steps - tail
    assert metadata['counts'] == dict(denoiser=89*steps, encode=178*coupled,
                                      decode=89*(coupled+1), initialize=0)
    assert len(metadata['intervals']) == steps and len(metadata['fusions']) == coupled+1
    schedule = metadata['provenance']['prepared_schedule']
    assert len(schedule['timesteps']) == steps and len(schedule['sigmas']) == steps+1
    assert metadata['schedule_sha256'] == digest(schedule)
    for k, interval in enumerate(metadata['intervals']):
        assert interval['k'] == k and interval['model_timestep'] == schedule['timesteps'][k]
        if k >= coupled:
            assert interval['phase'] == 'independent_refinement' and interval['fusion'] is None
    for fusion in metadata['fusions']:
        assert fusion['mode'] == 'poisson_select' and fusion['lambda_color'] == .1
        assert fusion.get('reference_mode', 'screened') == 'screened'
        assert fusion['converged'] is True
        assert fusion['final_true_residual'] <= fusion['stopping_threshold']
    if tail:
        for key, value in dict(total_intervals=steps, coupled_intervals=coupled,
                independent_intervals=tail, last_fraction=tail/steps).items():
            assert metadata['refinement'][key] == value
    return config


def collect(*, allow_mixed_steps=False):
    """Return 24 triples only when the 20-versus-28-step comparison is accepted."""
    if allow_mixed_steps is not True:
        raise ValueError('Explicit allow_mixed_steps=True is required: the 0% baseline has 20 steps; both refinement runs have 28.')
    index_path = REPORT / 'comparison-index.json'
    index = read(index_path)
    assert index['complete'] is True and index['source_images_unchanged'] is True
    current, current_evidence = _inventory(index, 'refinement', 'refinement28')
    baseline, baseline_evidence = _inventory(index, 'uniform', 'uniform')
    baselines = {row['prompt']: row for row in baseline['figures']}
    assert set(baselines) == {row['prompt'] for row in current['figures']}
    figures, comparisons = [], []
    for previous in current['figures']:
        keys = [panel['key'] for panel in previous['panels']]
        assert keys == list(REFINEMENT_KEYS), 'Current refinement source already has a baseline or unexpected panels'
        baseline_figure = baselines[previous['prompt']]
        assert previous['prompt_path'] == baseline_figure['prompt_path']
        assert previous['prompt_sha256'] == baseline_figure['prompt_sha256']
        candidates = [p for p in baseline_figure['panels'] if p['key'] == BASELINE_KEY]
        assert len(candidates) == 1
        panels = [copy.deepcopy(candidates[0])] + copy.deepcopy(previous['panels'])
        wrapper = _source(panels[0])
        scientific = panels[0]['scientific']
        for key, value in dict(backend='flux', steps=20, seed=0, camera_count=89,
                layout='old89', gradient_mode='poisson_select', reference_mode='screened',
                refinement_fraction=0.).items():
            assert scientific[key] == value
        assert wrapper['reused'] is True and wrapper['prompt'] == previous['prompt']
        original_path = Path(wrapper['original_metadata_path'])
        assert sha(original_path) == wrapper['original_metadata_sha256']
        original = read(original_path)
        source = wrapper['source']
        assert source['image_sha256'] == original['artifacts']['final.png'] == panels[0]['image_sha256']
        assert source['original_metadata_sha256'] == wrapper['original_metadata_sha256']
        assert configuration(source['config']) == configuration(original['configuration'])
        assert source['fusion'] == original['fusion']
        assert source['refinement'] == 0. and source['steps'] == 20 and source['layout'] == 'old89'
        metadata = [original] + [_source(panel) for panel in panels[1:]]
        configs = []
        for panel, record, steps, tail, percent, label in zip(panels, metadata,
                (20, 28, 28), (0, 5, 8), (0, 20, 30), LABELS):
            _prompt(previous, record)
            configs.append(_method(record, steps, tail))
            panel['label'] = label
            panel['display_percent'] = percent
            panel['scientific'].update(total_steps=steps, coupled_steps=steps-tail,
                refinement_steps=tail, exact_refinement_fraction=tail/steps,
                exact_refinement_ratio='%d/%d' % (tail, steps), display_percent=percent,
                actual_refinement_percent=100*tail/steps, reference_mode='screened',
                solver=copy.deepcopy(SOLVER))
        expected_changes = ['generation.num_inference_steps', 'global_pipeline.refinement.last_fraction']
        assert differences(configs[0], configs[1]) == expected_changes
        assert differences(configs[0], configs[2]) == expected_changes
        assert differences(configs[1], configs[2]) == ['global_pipeline.refinement.last_fraction']
        assert metadata[1]['provenance'] == metadata[2]['provenance']
        provenance = [copy.deepcopy(record['provenance']) for record in metadata]
        schedules = [record.pop('prepared_schedule') for record in provenance]
        assert provenance[0] == provenance[1] == provenance[2]
        assert schedules[0]['class_name'] == schedules[1]['class_name']
        assert schedules[0]['config'] == schedules[1]['config']
        for key in ('initial_local_sha256', 'source_sha256', 'camera_slots', 'scaling', 'scaling_applications'):
            assert metadata[0]['initialization'][key] == metadata[1]['initialization'][key] == metadata[2]['initialization'][key]
        baseline_source = dict(path=str(original_path), sha256=wrapper['original_metadata_sha256'],
            legacy_screened_field_inferred_from_authenticated_inventory='reference_mode' not in original['fusion'])
        panels[0]['scientific']['original_metadata'] = baseline_source
        figures.append(dict(kind='refinement_percent', prompt=previous['prompt'],
            title=previous['title'], subtitle='FLUX | Old89 | seed 0 | screened poisson_select | lambda 0.1',
            prompt_path=previous['prompt_path'], prompt_sha256=previous['prompt_sha256'], panels=panels))
        comparisons.append(dict(prompt=previous['prompt'], baseline_original_metadata=baseline_source,
            configuration_changed_fields=expected_changes,
            unchanged_runtime_provenance_except_schedule=True,
            schedule_sha256=[record['schedule_sha256'] for record in metadata],
            schedule_step_counts=[20, 28, 28], initial_native_sha256=original['initialization']['initial_local_sha256'],
            conditioning_sha256=original['provenance']['conditioning_sha256']))
    assert len(figures) == 24
    return dict(figures=figures, evidence=dict(allow_mixed_steps=True, generation_performed=False,
        source_images_modified=False, prompt_count=24, verified_panel_count=72,
        comparison_index_path=str(index_path), comparison_index_sha256=sha(index_path),
        previous_refinement=current_evidence, previous_uniform=baseline_evidence,
        validation='Prior inventories authenticate full-artifact validation; this collector rechecks PNG, metadata, original baseline metadata, prompt bytes, and actual generator settings. Raw tensors are not rehashed.',
        baseline_note='The 0% baseline uses 20 total steps. The 20% and 30% display labels designate exact 5/28 and 8/28 independent terminal intervals in 28-step runs.',
        label_interpretation=dict(display_percent=[0, 20, 30], exact_refinement_ratios=['0/20', '5/28', '8/28'],
            total_steps=[20, 28, 28], coupled_steps=[20, 23, 20], independent_steps=[0, 5, 8]),
        comparisons=comparisons))
