"""Read-only preservation checks and immutable identities for the retry batch."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

from studies.refinement28 import common as original

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_OUT = ROOT / 'outputs/refinement28-flux48/seed0-v1'
OUT = RESUME_OUT = ORIGINAL_OUT / 'recovery-v1'
PLAN = OUT / 'manifest.json'
AUDIT = OUT / 'audit.json'
ACCOUNT = '142684268708'
INDICES = ALLOWED_INDICES = (24, 25, 26, 27, 42, 43, 44, 45, 46, 47)
read, write, sha, digest, safe = original.read, original.write, original.sha, original.digest, original.safe


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sources():
    return {path.relative_to(ROOT).as_posix(): sha(path)
            for path in sorted((ROOT / 'studies/refinement28_resume').rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts
            and path.suffix in ('.py', '.sh', '.slurm', '.txt', '.md', '.json', '.yaml', '.yml')}


def definitions():
    manifest = read(ORIGINAL_OUT / 'manifest.json')
    assert len(manifest['rows']) == 48
    result = [manifest['rows'][index] for index in INDICES]
    assert [row['index'] for row in result] == list(INDICES)
    return result


def case_directory(row):
    identity = Path(row['case_id'])
    assert not identity.is_absolute() and '..' not in identity.parts
    return ORIGINAL_OUT / 'cases' / identity


def snapshot(directory):
    """Record bytes and relative names; timestamps and inode numbers are immaterial."""
    directory = Path(directory)
    assert not directory.is_symlink(), 'Case directory must not be a symlink'
    if not directory.exists():
        value = dict(exists=False, files={})
    else:
        assert directory.is_dir(), 'Case path is not a directory'
        files = {}
        for path in sorted(directory.rglob('*')):
            assert not path.is_symlink(), 'Unexpected symlink in case: ' + str(path)
            if path.is_file():
                files[path.relative_to(directory).as_posix()] = dict(
                    sha256=sha(path), size_bytes=path.stat().st_size)
            else:
                assert path.is_dir(), 'Unexpected special file in case: ' + str(path)
        value = dict(exists=True, files=files)
    return dict(value, digest=digest(value))


def verify_snapshot(directory, expected):
    assert expected['digest'] == digest({key: expected[key] for key in ('exists', 'files')})
    assert snapshot(directory) == expected, 'Case bytes changed: ' + str(directory)
    return True


def _git(*arguments):
    return subprocess.check_output(['git'] + list(arguments), cwd=str(ROOT), text=True).strip()


def git_evidence(original_head):
    head = _git('rev-parse', 'HEAD')
    assert subprocess.run(['git', 'merge-base', '--is-ancestor', original_head, head],
                          cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0, (
        'Original revision is no longer an ancestor of the observed commit')
    delta = _git('diff', '--name-status', '--no-renames', original_head, head)
    return dict(original_git_head=original_head, observed_git_head=head,
                branch=_git('branch', '--show-current'), original_is_ancestor=True,
                name_status=delta.splitlines() if delta else [])


def _check_hashes(hashes):
    for name, expected in hashes.items():
        path = Path(name)
        path = path if path.is_absolute() else ROOT / path
        assert path.is_file() and sha(path) == expected, 'Protected bytes changed: ' + str(path)


def audit_original_inputs():
    """Check all old content gates while treating Git HEAD as provenance only."""
    manifest_path = ORIGINAL_OUT / 'manifest.json'
    startup_path = ORIGINAL_OUT / 'startup-preservation.json'
    validation_path = ORIGINAL_OUT / 'validation.json'
    manifest, startup, validation = read(manifest_path), read(startup_path), read(validation_path)
    assert validation['passed'] is True
    assert validation['manifest_sha256'] == sha(manifest_path), 'Original manifest changed'
    assert validation['source_sha256'] == manifest['source_sha256'] == digest(manifest['source_hashes'])
    _check_hashes(startup['hashes'])
    _check_hashes(manifest['source_hashes'])
    # This independently checks the source file set, all full prompt records,
    # ordered case definitions, semantic hashes, and per-case source hashes.
    assert original.freeze_check() == manifest
    assert os.readlink(ROOT / 'outputs') == startup['outputs_symlink'], 'Outputs symlink changed'
    observed = git_evidence(startup['git_head'])
    assert observed['branch'] == startup['branch'], 'Repository branch changed'
    return dict(
        original_manifest_sha256=sha(manifest_path),
        original_startup_preservation_sha256=sha(startup_path),
        original_validation_sha256=sha(validation_path),
        original_source_sha256=manifest['source_sha256'],
        original_source_count=len(manifest['source_hashes']),
        preservation_count=len(startup['hashes']), prompt_count=len(manifest['prompts']),
        outputs_symlink=startup['outputs_symlink'], branch=startup['branch'],
        original_git_head=startup['git_head'], observed_git_head=observed['observed_git_head'],
        git=observed, content_verified=True)


def _check_helper_sources(plan):
    current = sources()
    assert current == plan['source_hashes'], 'Recovery implementation changed after freeze'
    assert digest(current) == plan['source_sha256'], 'Invalid recovery source digest'


def preserve_original():
    """Replacement for the old HEAD equality check; no original file is rewritten."""
    audit = read(AUDIT)
    current = audit_original_inputs()
    expected = audit['original_inputs']
    for key in ('original_manifest_sha256', 'original_startup_preservation_sha256',
                'original_validation_sha256', 'original_source_sha256', 'original_source_count',
                'preservation_count', 'prompt_count', 'outputs_symlink', 'branch', 'original_git_head'):
        assert current[key] == expected[key], 'Original preservation identity changed: ' + key
    if PLAN.exists():
        plan = read(PLAN)
        assert sha(AUDIT) == plan['audit_sha256'], 'Recovery audit changed'
        _check_helper_sources(plan)
    current['preparation_git_head'] = expected['observed_git_head']
    return current


def gate():
    plan = read(PLAN)
    assert plan['indices'] == plan['allowed_indices'] == list(INDICES), 'Recovery whitelist changed'
    assert sha(AUDIT) == plan['audit_sha256'], 'Recovery audit changed'
    _check_helper_sources(plan)
    validation = read(OUT / 'validation.json')
    assert validation['passed'] is True
    assert validation['plan_sha256'] == sha(PLAN), 'Recovery manifest changed after validation'
    assert validation['audit_sha256'] == plan['audit_sha256']
    assert validation['source_sha256'] == plan['source_sha256']
    audit = read(AUDIT)
    assert plan['cases'] == audit['cases'], 'Case snapshots differ from immutable audit'
    assert plan['original_manifest_sha256'] == audit['original_inputs']['original_manifest_sha256']
    assert plan['protected_completed'] == audit['protected_completed']
    rows = definitions()
    assert [case['index'] for case in plan['cases']] == list(INDICES)
    for case, row in zip(plan['cases'], rows):
        for key in ('index', 'case_id', 'semantic_sha256', 'source_sha256'):
            assert case[key] == row[key], 'Retry case differs from frozen original: ' + key
        assert case['directory'] == str(case_directory(row))
        value = case['snapshot']
        assert value['digest'] == digest({key: value[key] for key in ('exists', 'files')})
    preserve_original()
    return plan


def verify_protected_outputs(plan=None):
    """Full retained-output audit for preparation/final CPU checks, not every GPU step."""
    plan = gate() if plan is None else plan
    assert len(plan['protected_completed']) == 38
    assert {case['index'] for case in plan['protected_completed']} == set(range(48)) - set(INDICES)
    for case in plan['protected_completed']:
        verify_snapshot(case['directory'], case['snapshot'])
    return True
