"""One retry identity; original failed artifacts and original freeze stay intact."""
import contextlib
import copy
import os
import subprocess
from pathlib import Path
from studies.mean_max_old89 import common as base

ROOT = base.ROOT
OUT = ROOT / 'outputs/10.15mean-max-firefly-retry/seed0-v1'
BASE_INDEX = 27
OLD_RTOL = 1e-4
NEW_RTOL = 2e-4
MODE = base.MODE
COUNTS = base.COUNTS
PROMPTS = ('firefly',)
AUTHORIZATION = '2026-10-15 user: well, please raise the threshold to get firefy results and make the figure'
sha, digest, read, write, safe = base.sha, base.digest, base.read, base.write, base.safe
prompt, prompt_path = base.prompt, base.prompt_path


@contextlib.contextmanager
def effective_prompt(name):
    assert name == 'firefly'
    with base.effective_prompt(name) as value:
        yield value


def base_row():
    row = read(base.OUT / 'manifest.json')['rows'][BASE_INDEX]
    assert row['index'] == BASE_INDEX and row['prompt'] == 'firefly'
    assert row['case_id'] == 'prompts/firefly/coarse_color/poisson_mean_max/alpha025/refine000'
    assert row['reference_mode'] == 'coarse_color' and row['gradient_mix_alpha'] == .25
    assert row['semantic']['solver']['relative_tolerance'] == OLD_RTOL
    return row


def definitions():
    row = copy.deepcopy(base_row())
    row['semantic']['solver']['relative_tolerance'] = NEW_RTOL
    row['semantic_sha256'] = digest(row['semantic'])
    row['source_sha256'] = digest(sources())
    return [row]


def sources():
    values = base.sources()
    for path in sorted((ROOT / 'studies/mean_max_firefly_retry').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.slurm', '.sh', '.txt'):
            values[str(path.relative_to(ROOT))] = sha(path)
    return values


def folder(row):
    return OUT / 'cases' / row['case_id']


def retry_record():
    original = base_row()
    return dict(authorization=AUTHORIZATION, base_case_id=original['case_id'],
                base_case_index=BASE_INDEX, base_case_directory=str(base.folder(original)),
                base_manifest_sha256=sha(base.OUT / 'manifest.json'),
                base_semantic_sha256=original['semantic_sha256'],
                previous_relative_tolerance=OLD_RTOL, relative_tolerance=NEW_RTOL,
                changed_settings=['solver.relative_tolerance'],
                absolute_tolerance=1e-7, constrained_max_iterations=400,
                original_failed_case_preserved=True, automatic_relaxation=False)


def check_scope(row):
    original = base_row()
    expected = copy.deepcopy(original['semantic'])
    expected['solver']['relative_tolerance'] = NEW_RTOL
    assert row['semantic'] == expected, 'Retry may change only solver.relative_tolerance'
    assert row['semantic_sha256'] == digest(expected)
    for key in set(original) - {'semantic', 'semantic_sha256', 'source_sha256'}:
        assert row[key] == original[key], key


def preserved():
    base.preserved()
    record = read(OUT / 'history/original-failure.json')
    for path, value in record['source_artifacts'].items():
        assert sha(path) == value, 'Original failed artifact changed: ' + path


def freeze_check():
    manifest = read(OUT / 'manifest.json')
    assert manifest['source_hashes'] == sources(), 'Retry source changed after freeze'
    assert manifest['base_manifest_sha256'] == sha(base.OUT / 'manifest.json')
    assert len(manifest['rows']) == 1
    row = manifest['rows'][0]
    check_scope(row)
    assert row['source_sha256'] == digest(manifest['source_hashes'])
    assert prompt('firefly')[0] == row['semantic']['prompt']
    assert manifest['tolerance_retry'] == retry_record()
    return manifest


def base_validation():
    """Reuse unchanged model/geometry/guidance checks; tolerance change is explicit."""
    manifest = freeze_check()
    base_manifest = read(base.OUT / 'manifest.json')
    assert base_manifest['source_hashes'] == base.sources()
    records = {}
    for name in ('cpu.json', 'weights.json', 'metrics-cpu.json', 'gpu.json'):
        path = base.OUT / 'validation' / name
        assert sha(path) == manifest['base_validation_hashes'][name]
        record = read(path)
        assert record['passed']
        assert record['source_sha256'] == digest(base_manifest['source_hashes'])
        assert record['manifest_sha256'] == manifest['base_manifest_sha256']
        records[name] = record
    return records['gpu.json']


def complete(row, verify=True):
    path = folder(row)
    try:
        check_scope(row)
        status = read(path / 'status.json')
        if status['state'] != 'complete':
            return False
        meta = read(path / 'metadata.json')
        if meta['semantic_sha256'] != row['semantic_sha256'] or meta['source_sha256'] != row['source_sha256']:
            return False
        if status['semantic_sha256'] != row['semantic_sha256']:
            return False
        if meta['tolerance_retry'] != retry_record() or meta['fusion'] != row['semantic']['solver']:
            return False
        required = {'final.png', 'terminal-rgb-reference.png', 'final-unclamped-fp32.pt', 'terminal-rgb-reference-fp32.pt', 'config.json'}
        if set(meta['artifacts']) != required or meta['artifacts']['final.png'] != status['image_sha256']:
            return False
        if meta['counts'] != COUNTS or meta['poisson_solves'] != 21 or len(meta['fusions']) != 21:
            return False
        for fusion in meta['fusions']:
            if fusion['mode'] != MODE or fusion['reference_mode'] != row['reference_mode'] or fusion['gradient_mix_alpha'] != .25 or not fusion['converged']:
                return False
        if not meta['fusions'][-1]['terminal']:
            return False
        if verify:
            if sha(path / 'metadata.json') != status['metadata_sha256']:
                return False
            for name, value in meta['artifacts'].items():
                if sha(path / name) != value:
                    return False
        return True
    except (AssertionError, KeyError, ValueError, OSError):
        return False
