"""Read-only, factor-matched sources for current and historical refinement figures."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/shig/diffpano')
CURRENT = ROOT/'outputs/refinement28-flux48/seed0-v1'
HISTORICAL = ROOT/'outputs/10.8gradient-refinement-flux/seed0-v1'
OLD_PROMPTS = ('air_balloons', 'underwater', 'ruins', 'firework')
GEOMETRY = 'b55b4eb5051cdfb8c579e81d4f51ece1f15aeb0e8d7e1fde6b7b74a49154435c'
SOLVER = dict(mode='poisson_select', lambda_color=.1, relative_tolerance=1e-5,
              absolute_tolerance=1e-7, max_iterations=200)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def differences(first, second, prefix=''):
    if isinstance(first, dict) and isinstance(second, dict):
        assert first.keys() == second.keys(), prefix
        return [path for key in first for path in differences(
            first[key], second[key], prefix+'.'+key if prefix else key)]
    return [] if first == second else [prefix]


def configuration(value):
    value = copy.deepcopy(value)
    # Historical serialization omitted the default zero refinement field.
    value['global_pipeline'].setdefault('refinement', {'last_fraction': 0.})
    return value


def prompt_record(metadata, semantic):
    record = metadata['prompt']
    assert record == semantic['prompt'] and not record['adapted']
    path = Path(record.get('full_path', record['path']))
    path = path if path.is_absolute() else ROOT/path
    raw = path.read_bytes()
    assert raw.decode('utf-8').splitlines() == record['effective_lines']
    assert len(record['effective_lines']) == 5
    assert sha(path) == record['original_sha256'] == record['effective_sha256']
    return dict(prompt_path=str(path), prompt_sha256=record['effective_sha256'])


def panel(directory, metadata, key, label, steps, tail):
    from PIL import Image
    image = directory/'final.png'
    image_hash = sha(image)
    assert image_hash == metadata['artifacts']['final.png']
    with Image.open(image) as opened:
        assert opened.format == 'PNG' and opened.mode == 'RGB' and opened.size == (4096, 2048)
        opened.verify()
    return dict(key=key, label=label, image_path=str(image), image_sha256=image_hash,
                metadata_path=str(directory/'metadata.json'), metadata_sha256=sha(directory/'metadata.json'),
                scientific=dict(backend='flux', total_steps=steps, coupled_steps=steps-tail,
                    refinement_steps=tail, layout='old89', camera_count=89, seed=0,
                    reference_mode='screened', solver=SOLVER,
                    camera_geometry_sha256=metadata['camera_geometry_sha256'],
                    schedule_sha256=metadata['schedule_sha256'],
                    initial_native_sha256=metadata['initialization']['initial_local_sha256'],
                    source_noise_sha256=metadata['initialization']['source_sha256']))


def pair_checks(rows, metadata, steps):
    changed = differences(configuration(metadata[0]['configuration']), configuration(metadata[1]['configuration']))
    assert changed == ['global_pipeline.refinement.last_fraction'], changed
    for record, row in zip(metadata, rows):
        config = configuration(record['configuration'])
        assert config == configuration(row['semantic']['configuration'])
        assert config['generation']['num_inference_steps'] == steps
        assert config['experiment']['seed'] == 0
        assert config['model']['pipeline'] == 'flux'
        assert record['camera_geometry_sha256'] == GEOMETRY
        for key, value in SOLVER.items():
            assert record['fusion'][key] == value, key
        assert record['reference_mode'] == 'screened'
    # Actual model, conditioning, scheduler, and camera records must match, not
    # merely filenames or the declared experiment labels.
    assert metadata[0]['provenance'] == metadata[1]['provenance']
    assert metadata[0]['camera_ids'] == metadata[1]['camera_ids'] == list(range(89))
    for key in ('initial_local_sha256', 'source_sha256', 'camera_slots', 'scaling', 'scaling_applications'):
        assert metadata[0]['initialization'][key] == metadata[1]['initialization'][key]
    prompts = [prompt_record(record, row['semantic']) for record, row in zip(metadata, rows)]
    assert prompts[0] == prompts[1]
    return prompts[0], changed


def historical_case(row):
    directory = HISTORICAL/'cases'/row['key']
    metadata = read(directory/'metadata.json')
    semantic = row['semantic']
    assert row['semantic_sha256'] == metadata['semantic_sha256'] == digest(semantic)
    assert semantic['backend'] == 'flux' and semantic['steps'] == 20 and semantic['seed'] == 0
    assert row['reference_mode'] == 'screened' and row['gradient_mode'] == 'poisson_select'
    tail = 2 if row['last_fraction'] == .1 else 0
    assert row['last_fraction'] in (0., .1)
    coupled = 20-tail
    for key, value in dict(total_intervals=20, coupled_intervals=coupled,
                           independent_intervals=tail, last_fraction=row['last_fraction']).items():
        assert metadata['refinement'][key] == value
    assert metadata['counts'] == dict(denoiser=1780, encode=2*89*coupled, decode=89*(coupled+1), initialize=0)
    assert metadata['initialization']['initial_local_sha256'] == semantic['initial_native_sha256']
    assert metadata['camera_geometry_sha256'] == semantic['camera_geometry_sha256'] == GEOMETRY
    assert metadata['provenance']['prepared_schedule'] == semantic['schedule']
    assert metadata['provenance']['conditioning_sha256'] == semantic['conditioning_sha256']
    assert metadata['schedule_sha256'] == digest(semantic['schedule'])
    details = metadata['details']
    assert details['guided_predictions'] == 1780 and details['active_consensus_projection'] == 'erp'
    assert details['initial_state_sha256'] == semantic['initial_native_sha256']
    assert details['time_travel_enabled'] is False and details['original_noise_bank_retained'] is False
    for key in ('backward_calls', 'replay_count', 'original_noise_reinjections', 'additional_denoising_cycles'):
        assert details[key] == 0
    assert len(metadata['intervals']) == 20
    for index, interval in enumerate(metadata['intervals']):
        assert interval['k'] == index and interval['pass_kind'] == 'initial'
        assert interval['model_timestep'] == float(semantic['schedule']['timesteps'][index])
        if index >= coupled:
            assert interval['phase'] == 'independent_refinement' and interval['fusion'] is None
            assert interval['guided_predictions'] == 89
            assert set(interval['stage_seconds']) == {'independent_native'}
        else:
            assert interval.get('phase') != 'independent_refinement'
    assert len(metadata['fusions']) == coupled+1
    for index, fusion in enumerate(metadata['fusions']):
        assert fusion['stage'] == ('terminal' if index == coupled else index+1)
        assert fusion['terminal'] == (index == coupled) and fusion['converged'] is True
        assert fusion['mode'] == 'poisson_select' and fusion['ownership_consistent'] is True
        assert fusion.get('reference_mode', metadata['reference_mode']) == 'screened'
        assert fusion['lambda_color'] == .1
        assert fusion['final_true_residual'] <= fusion['stopping_threshold']
    if tail:
        assert details['refinement'] == metadata['refinement']
        assert details['refinement_transition']['original_index'] == coupled
        assert details['refinement_transition']['step'] == coupled+1
        assert details['refinement_transition']['timestep'] == float(semantic['schedule']['timesteps'][coupled])
        assert digest(metadata['source_hashes']) == row['implementation_sha256']
        assert metadata['fusion'] == semantic['fusion']
    else:
        audit = metadata['reuse_audit']
        assert metadata['reused'] is True and audit['passed'] and all(audit['checks'].values())
        source = Path(metadata['reused_from'])
        assert sha(source/'metadata.json') == audit['metadata_sha256']
        assert sha(source/'final.png') == audit['image_sha256'] == metadata['artifacts']['final.png']
    retained, absent = {}, []
    for name, expected in metadata['artifacts'].items():
        path = directory/name
        if path.exists():
            assert sha(path) == expected, str(path)
            retained[name] = expected
        else:
            absent.append(name)
    assert 'final.png' in retained and 'terminal-rgb-reference.png' in retained
    status = directory/'status.json'
    if status.exists():
        record = read(status)
        assert record['state'] in ('complete', 'reused') and record['metadata_sha256'] == sha(directory/'metadata.json')
    evidence = dict(case_id=row['key'], validation='retained PNGs, semantic identity, actual configuration and execution records',
                    status_file_present=status.exists(), retained_artifact_hashes=retained,
                    absent_recorded_artifacts=absent, metadata_sha256=sha(directory/'metadata.json'))
    return directory, metadata, evidence


def collect():
    from studies.refinement28 import common as current
    from studies.followup24.common import PROMPTS24

    manifest = current.freeze_check()
    assert current.OUT == CURRENT
    lookup = {(row['prompt'], row['refinement_steps']): row for row in manifest['rows']}
    assert len(lookup) == 48 and set(lookup) == {(name, tail) for name in PROMPTS24 for tail in (5, 8)}
    figures, comparisons, historical_evidence = [], [], []
    for name in PROMPTS24:
        rows = [lookup[(name, tail)] for tail in (5, 8)]
        for row in rows:
            assert current.complete(row, verify=True), row['case_id']
        paths = [current.folder(row) for row in rows]
        metadata = [read(path/'metadata.json') for path in paths]
        prompt, changed = pair_checks(rows, metadata, 28)
        left, right = [row['semantic'] for row in rows]
        assert differences(left['configuration'], right['configuration']) == changed
        assert differences(left['case'], right['case']) == ['coupled_steps', 'last_fraction', 'refinement_steps']
        assert {key: value for key, value in left.items() if key not in ('configuration', 'case', 'refinement')} == {
            key: value for key, value in right.items() if key not in ('configuration', 'case', 'refinement')}
        assert metadata[0]['same_prompt_group_identity'] == metadata[1]['same_prompt_group_identity']
        panels = [panel(path, record, 'refine%02dof28' % tail,
                        'Refinement %d/28 (%d + %d steps)' % (tail, 28-tail, tail), 28, tail)
                  for path, record, tail in zip(paths, metadata, (5, 8))]
        figures.append(dict(kind='refinement28', prompt=name, title=name.replace('_', ' ').title(),
            subtitle='FLUX28 | Old89 | seed 0 | screened poisson_select | lambda 0.1', panels=panels, **prompt))
        comparisons.append(dict(kind='refinement28', prompt=name, configuration_changed_fields=changed,
                                completion='original complete(row, verify=True), including all recorded artifacts'))
    old_manifest = read(HISTORICAL/'manifest.json')
    for name in OLD_PROMPTS:
        rows = [next(row for row in old_manifest['rows'] if row['prompt'] == name and
                     row['reference_mode'] == 'screened' and row['gradient_mode'] == 'poisson_select' and
                     row['last_fraction'] == fraction) for fraction in (0., .1)]
        items = [historical_case(row) for row in rows]
        prompt, changed = pair_checks(rows, [item[1] for item in items], 20)
        assert differences(rows[0]['semantic'], rows[1]['semantic']) == ['configuration.global_pipeline.refinement.last_fraction']
        labels = ('No refinement (20 + 0 steps)', 'Refinement 2/20 (18 + 2 steps)')
        panels = [panel(item[0], item[1], 'refine%02dof20' % tail, label, 20, tail)
                  for item, tail, label in zip(items, (0, 2), labels)]
        figures.append(dict(kind='refinement20', prompt=name, title=name.replace('_', ' ').title(),
            subtitle='FLUX20 | Old89 | seed 0 | screened poisson_select | lambda 0.1', panels=panels, **prompt))
        historical_evidence.extend(item[2] for item in items)
        comparisons.append(dict(kind='refinement20', prompt=name, configuration_changed_fields=changed,
                                completion='retained-artifact validation; missing historical status files are not reconstructed'))
    assert len(figures) == 28
    return dict(figures=figures, evidence=dict(
        manifests=[dict(path=str(root/'manifest.json'), sha256=sha(root/'manifest.json')) for root in (CURRENT, HISTORICAL)],
        current_verified_cases=48, current_pairs=24, historical_retained_cases=8, historical_pairs=4,
        comparisons=comparisons, historical_cases=historical_evidence,
        historical_note='Historical status files and some recorded raw tensors are absent. Retained PNG hashes, semantic identities, configurations, schedules, noise, operation counts, interval phases and terminal convergence are verified; no full historical completion status is claimed.',
        baseline_note='The supplemental 20-step off/on comparison is separate from the 28-step 5/8 comparison.'))
