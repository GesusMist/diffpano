"""Audit all old inputs and freeze a separate, narrowly scoped recovery plan."""
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

from studies.refinement28_resume import common


def create_json(path, value):
    """Publish an immutable record atomically, refusing any existing destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(common.safe(value), handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def collect_cases():
    manifest = common.read(common.ORIGINAL_OUT / 'manifest.json')
    cases, protected = [], []
    assert len(manifest['rows']) == 48
    for row in manifest['rows']:
        directory = common.case_directory(row)
        value = dict(index=row['index'], case_id=row['case_id'], directory=str(directory),
                     semantic_sha256=row['semantic_sha256'], source_sha256=row['source_sha256'],
                     snapshot=common.snapshot(directory))
        if row['index'] in common.INDICES:
            assert not common.original.complete(row, verify=True), 'Whitelisted case is already complete'
            status_path = directory / 'status.json'
            value['original_status'] = common.read(status_path) if status_path.exists() else None
            cases.append(value)
        else:
            assert common.original.complete(row, verify=True), 'Unexpected unfinished case outside whitelist'
            protected.append(value)
    assert [case['index'] for case in cases] == list(common.INDICES)
    assert len(protected) == 38
    return cases, protected


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run recovery preparation in a Slurm CPU allocation')
    if os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'):
        raise RuntimeError('Recovery preparation requires a CPU allocation')
    common.OUT.mkdir(parents=True, exist_ok=True)
    for path in (common.PLAN, common.AUDIT, common.OUT / 'validation.json'):
        if path.exists() or path.is_symlink():
            raise FileExistsError('Recovery records are immutable; existing record: ' + str(path))
    for name in ('logs/generation', 'logs/report', 'validation', 'attempts', 'execution'):
        (common.OUT / name).mkdir(parents=True, exist_ok=True)
    source_hashes = common.sources()
    inputs = common.audit_original_inputs()
    assert all(line.split('\t', 1)[0] == 'A' for line in inputs['git']['name_status']), (
        'Initial revision drift is not the expected addition-only commit')
    cases, protected = collect_cases()
    with (common.OUT / 'validation/tests.log').open('w') as stream:
        suite = unittest.defaultTestLoader.discover(str(common.ROOT / 'studies/refinement28_resume'),
                                                   pattern='tests_*.py', top_level_dir=str(common.ROOT))
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('Recovery tests failed; see validation/tests.log')
    assert common.sources() == source_hashes, 'Recovery implementation changed during preparation'
    current = common.audit_original_inputs()
    # Commit identity may advance while bytes remain fixed; record both observations.
    for key in inputs:
        if key not in ('observed_git_head', 'git'):
            assert current[key] == inputs[key], 'Original inputs changed during preparation'
    audit = dict(
        schema=1, created_utc=common.utc_now(),
        authorization=dict(user_instruction='please allow more time and retry',
                           scope='Retry only the ten unfinished original refinement28 cases with unchanged semantics',
                           regeneration_indices=list(common.INDICES),
                           rendered_case_policy='Archive incomplete existing artifacts intact before regenerating'),
        original_inputs=inputs, end_of_preparation_git=current['git'],
        head_policy='Accept commit drift only while every frozen source, prompt, evidence, helper byte, branch and outputs link is unchanged; original revision remains an ancestor',
        cases=cases, protected_completed=protected)
    create_json(common.AUDIT, audit)
    plan = dict(
        schema=1, study='refinement28_resume', created_utc=common.utc_now(),
        indices=list(common.INDICES), allowed_indices=list(common.INDICES), cases=cases,
        protected_completed=protected, original_manifest_sha256=inputs['original_manifest_sha256'],
        original_source_sha256=inputs['original_source_sha256'],
        original_git_head=inputs['original_git_head'], preparation_git_head=inputs['observed_git_head'],
        source_hashes=source_hashes, source_sha256=common.digest(source_hashes),
        audit_sha256=common.sha(common.AUDIT), increased_walltime_minutes=240,
        walltime='04:00:00', exclude_nodes=['fc026'], concurrency=2,
        generation_case_count=10, protected_completed_count=38)
    create_json(common.PLAN, plan)
    with tarfile.open(common.OUT / 'implementation.tar.gz', 'x:gz') as archive:
        for name in source_hashes:
            archive.add(common.ROOT / name, arcname=name)
    create_json(common.OUT / 'validation.json', dict(
        passed=True, tests_run=result.testsRun, source_sha256=plan['source_sha256'],
        plan_sha256=common.sha(common.PLAN), audit_sha256=plan['audit_sha256'],
        original_inputs_verified=True, protected_completed_verified=38))
    common.gate()
    print('RECOVERY_PREPARED: 10 retries, 38 protected results;', result.testsRun, 'tests passed', flush=True)


if __name__ == '__main__':
    main()
