"""Login-safe identities, provenance, and strict completion checks for 48 cases."""
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from studies.followup24.common import NEW4, PROMPTS24

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/refinement28-flux48/seed0-v1'
ACCOUNT = '142684268708'
PROFILE_INDICES = (0, 1)
ARTIFACT_NAMES = (
    'config.json', 'final.png', 'final-unclamped-fp32.pt',
    'terminal-rgb-reference.png', 'terminal-rgb-reference-fp32.pt',
)


def safe(value):
    if is_dataclass(value):
        return safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [safe(item) for item in (sorted(value) if isinstance(value, set) else value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return '-Infinity' if value < 0 else 'Infinity' if value > 0 else 'NaN'
    if hasattr(value, 'detach'):
        return safe(value.detach().cpu().tolist())
    if hasattr(value, 'item'):
        return safe(value.item())
    return value


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    """Replace one JSON record atomically; never leave a partially written gate."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(safe(value), sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def definitions():
    result = []
    for name in PROMPTS24:
        relative = Path('prompts') / ('ournew' if name in NEW4 else '') / (name + '.txt')
        identity = relative.with_suffix('').as_posix()
        for tail in (5, 8):
            result.append(dict(
                index=len(result),
                case_id=identity + '/old89/screened/poisson_select/refine%02dof28' % tail,
                prompt=name, prompt_path=str(ROOT / relative), prompt_identity=identity,
                backend='flux', method='diffpano', layout='old89', camera_count=89,
                projection='erp', seed=0, steps=28, refinement_steps=tail,
                coupled_steps=28 - tail, last_fraction=tail / 28.0,
                gradient_mode='poisson_select', reference_mode='screened',
                gradient_mix_alpha=None, time_travel=False, camera_rotation=False))
    assert len(result) == 48 and len({row['case_id'] for row in result}) == 48
    return result


def validate_row(row):
    index = row.get('index')
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 48:
        raise ValueError('Case index is outside the exact 48-case study')
    expected = definitions()[index]
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError('Case differs from the authorized prompt/layout/tail definition')
    return row


def folder(row):
    validate_row(row)
    return OUT / 'cases' / row['case_id']


def expected_counts(row):
    validate_row(row)
    cameras = row['camera_count']
    coupled = row['coupled_steps'] * cameras
    return dict(denoiser=row['steps'] * cameras, encode=2 * coupled,
                decode=coupled + cameras, initialize=0)


def sources():
    """Hash code used by the new execution path without editing inherited gates."""
    directories = (
        'diffpano', 'scripts', 'configs', 'studies/refinement28', 'studies/gradient_blending',
        'studies/gradient_refinement', 'studies/all_prompts', 'studies/tt_cea',
        'studies/gwtf_noise', 'studies/gwtf_erp4k', 'studies/camera_patching',
        'studies/camera_count_v2', 'studies/layout24', 'studies/ournew_comparison',
        'studies/original_spherediff', 'studies/flux_step_controls',
        'studies/mean_max_old89', 'studies/gradient_refinement_retry',
        'studies/gradient_score_refresh', 'studies/panorama_metrics',
    )
    values = {}
    for directory in directories:
        for path in sorted((ROOT / directory).rglob('*')):
            if (path.is_file() and '__pycache__' not in path.parts
                    and path.suffix in ('.py', '.sh', '.slurm', '.txt', '.yaml', '.yml', '.json')):
                values[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ('studies/followup24/common.py', 'studies/followup24/__init__.py'):
        values[name] = sha(ROOT / name)
    return values


def prompt_record(name):
    from studies.followup24.common import prompt
    return prompt(name)[0]


def freeze_check():
    manifest = read(OUT / 'manifest.json')
    current = sources()
    assert manifest['source_hashes'] == current, 'Scientific source changed after freeze'
    assert manifest['source_sha256'] == digest(current), 'Invalid manifest source digest'
    if 'outputs_symlink' in manifest:
        assert os.readlink(ROOT / 'outputs') == manifest['outputs_symlink'], 'Outputs symlink changed'
    rows = manifest['rows']
    assert len(rows) == 48, 'Manifest does not contain the exact 48 cases'
    prompts = {name: prompt_record(name) for name in PROMPTS24}
    if 'prompts' in manifest:
        assert manifest['prompts'] == prompts, 'Frozen prompts changed'
    for index, row in enumerate(rows):
        validate_row(row)
        assert row['index'] == index, 'Manifest case order changed'
        assert row['source_sha256'] == manifest['source_sha256'], 'Case source differs from manifest'
        assert row['semantic_sha256'] == digest(row['semantic']), 'Case semantic digest changed'
        assert row['semantic']['prompt'] == prompts[row['prompt']], 'Prompt changed after freeze'
    return manifest


def preserved():
    record = read(OUT / 'startup-preservation.json')
    assert isinstance(record['hashes'], dict) and record['hashes'], 'Missing startup preservation hashes'
    for name, expected in record['hashes'].items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        assert sha(path) == expected, 'Preserved source/evidence changed: ' + name
    if 'outputs_symlink' in record:
        assert os.readlink(ROOT / 'outputs') == record['outputs_symlink'], 'Outputs symlink changed'
    for key, command in (('branch', ['git', 'branch', '--show-current']),
                         ('git_head', ['git', 'rev-parse', 'HEAD'])):
        if key in record:
            assert subprocess.check_output(command, cwd=str(ROOT)).decode().strip() == record[key], (
                'Repository ' + key + ' changed')
    return record


def validation_gate(manifest=None):
    manifest = freeze_check() if manifest is None else manifest
    gate = read(OUT / 'validation.json')
    assert gate['passed'] is True, 'Validation did not pass'
    assert gate['source_sha256'] == manifest['source_sha256'], 'Stale validation source hash'
    assert gate['manifest_sha256'] == sha(OUT / 'manifest.json'), 'Stale validation manifest hash'
    return gate


def validate_metadata(row, metadata):
    """Validate scientific completion independently of an optimistic status flag."""
    validate_row(row)
    semantic = row['semantic']
    assert row['semantic_sha256'] == digest(semantic), 'Invalid case semantic digest'
    assert metadata['semantic_sha256'] == row['semantic_sha256'], 'Metadata semantic mismatch'
    assert metadata['source_sha256'] == row['source_sha256'], 'Metadata source mismatch'
    assert metadata['configuration'] == semantic['configuration'], 'Resolved config changed'
    assert metadata['fusion'] == semantic['solver'], 'Poisson settings changed'
    assert metadata['prompt'] == semantic['prompt'], 'Prompt provenance changed'
    assert metadata['camera_count'] == row['camera_count'] == 89
    assert metadata['camera_geometry_sha256'] == semantic['geometry']['angular_sha256']
    assert metadata['counts'] == expected_counts(row), 'Unexpected generation operation counts'
    assert metadata['terminal_fusion_mode'] == 'poisson_select'
    assert metadata['terminal_reference_mode'] == 'screened'
    assert metadata.get('reused', False) is False, 'This study requires new, distinct trajectories'
    initialization = metadata['initialization']
    for key, value in semantic['expected_initialization'].items():
        assert initialization[key] == value, 'Initialization mismatch: ' + key
    assert metadata['initialization_audit']['calls'] == dict(
        denoiser=0, encode=0, decode=0, initialize=89), 'Unexpected setup calls'
    assert metadata['schedule_sha256'] == digest(semantic['schedule']), 'Prepared schedule mismatch'
    plan = metadata['refinement']
    assert plan == semantic['refinement'], 'Refinement differs from frozen plan'
    expected_plan = dict(last_fraction=row['last_fraction'], total_intervals=28,
                         coupled_intervals=row['coupled_steps'],
                         independent_intervals=row['refinement_steps'],
                         transition_step=row['coupled_steps'] + 1)
    for key, value in expected_plan.items():
        assert plan[key] == value, 'Incorrect resolved refinement: ' + key
    details = metadata['details']
    assert details['refinement'] == plan
    assert details['guided_predictions'] == expected_counts(row)['denoiser']
    assert details['time_travel_enabled'] is False and details['original_noise_bank_retained'] is False
    assert details['active_consensus_projection'] == 'erp'
    assert details['initial_state_sha256'] == initialization['initial_local_sha256']
    for key in ('backward_calls', 'replay_count', 'original_noise_reinjections', 'additional_denoising_cycles'):
        assert details[key] == 0, 'Unexpected trajectory operation: ' + key
    transition = details['refinement_transition']
    assert transition['original_index'] == row['coupled_steps']
    assert transition['step'] == plan['transition_step']
    assert transition['timestep'] == plan['transition_timestep']
    assert len(transition['native_state_sha256']) == 64
    intervals = metadata['intervals']
    assert len(intervals) == 28, 'Missing or extra original denoising interval'
    for index, interval in enumerate(intervals):
        assert interval['k'] == index, 'Intervals reordered or repeated'
        assert interval['pass_kind'] == 'initial', 'Unexpected replay'
        assert interval['model_timestep'] == float(semantic['schedule']['timesteps'][index])
        if index < row['coupled_steps']:
            assert interval['phase'] == 'coupled', 'Early native refinement'
        else:
            assert interval['phase'] == 'independent_refinement', 'Missing independent tail interval'
            assert 'fusion' in interval and interval['fusion'] is None, 'Fusion during independent tail'
            assert interval['guided_predictions'] == 89
            assert set(interval['stage_seconds']) == {'independent_native'}, 'Consensus operation during native tail'
    fusions = metadata['fusions']
    assert len(fusions) == row['coupled_steps'] + 1, 'Incorrect Poisson solve count'
    if 'poisson_solves' in metadata:
        assert metadata['poisson_solves'] == len(fusions), 'Poisson solve counter mismatch'
    for index, fusion in enumerate(fusions):
        assert fusion['mode'] == 'poisson_select' and fusion['reference_mode'] == 'screened'
        assert fusion.get('gradient_mix_alpha') is None, 'Unexpected gradient mixing'
        assert fusion['converged'] is True, 'A Poisson solve failed to converge'
        assert fusion['ownership_consistent'] is True, 'Poisson selection ownership changed'
        assert fusion['stage'] == ('terminal' if index == len(fusions) - 1 else index + 1)
        assert bool(fusion.get('terminal', False)) == (index == len(fusions) - 1), 'Incorrect terminal fusion'
    assert set(metadata['artifacts']) == set(ARTIFACT_NAMES), 'Incomplete artifact set'
    return True


def complete(row, verify=True):
    try:
        directory = folder(row)
        status = read(directory / 'status.json')
        if status['state'] != 'complete':
            return False
        metadata = read(directory / 'metadata.json')
        validate_metadata(row, metadata)
        assert status['semantic_sha256'] == row['semantic_sha256']
        assert status['source_sha256'] == row['source_sha256']
        assert status['metadata_sha256'] == sha(directory / 'metadata.json')
        assert status['image_sha256'] == metadata['artifacts']['final.png']
        assert status['artifacts'] == metadata['artifacts']
        if verify:
            for name, expected in metadata['artifacts'].items():
                path = directory / name
                assert path.is_file() and not path.is_symlink(), 'Artifact must be a local regular file'
                assert sha(path) == expected, 'Artifact hash changed: ' + name
            config = read(directory / 'config.json')
            assert config['configuration'] == metadata['configuration']
            assert config['fusion'] == metadata['fusion']
            from PIL import Image
            for name in ('final.png', 'terminal-rgb-reference.png'):
                with Image.open(directory / name) as image:
                    assert image.format == 'PNG' and image.mode == 'RGB' and image.size == (4096, 2048)
                    image.verify()
        return True
    except (AssertionError, KeyError, ValueError, TypeError, IndexError, OSError):
        return False
