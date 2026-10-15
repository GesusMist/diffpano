"""Validate scope and freeze exactly one authorized Firefly tolerance retry."""
import shutil
import tarfile
import unittest
from studies.mean_max_firefly_retry.common import *


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Preparation requires a Slurm CPU allocation'
    assert not (OUT / 'manifest.json').exists(), 'Existing freeze must not be overwritten'
    base.freeze_check()
    source = sources()
    rows = definitions()
    test_suite = unittest.defaultTestLoader.loadTestsFromName('studies.mean_max_firefly_retry.tests')
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    assert result.wasSuccessful()
    assert source == sources(), 'Source changed during preparation'
    original = base_row()
    original_folder = base.folder(original)
    assert read(original_folder / 'status.json')['state'] == 'failed'
    protected = {}
    for path in original_folder.rglob('*'):
        if path.is_file() and path.name != 'lock':
            protected[str(path)] = sha(path)
    for name in ('status.json', 'config.json', 'progress.json'):
        path = original_folder / name
        if path.exists():
            target = OUT / 'history/original-failure' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(path), str(target))
    write(OUT / 'history/original-failure.json', dict(case_id=original['case_id'],
          case_directory=str(original_folder), source_artifacts=protected))
    validations = {}
    for name in ('cpu.json', 'weights.json', 'metrics-cpu.json', 'gpu.json'):
        base.gate(name)
        validations[name] = sha(base.OUT / 'validation' / name)
    manifest = dict(rows=rows, logical_cases=1, expected_new=1, source_hashes=source,
                    base_manifest_sha256=sha(base.OUT / 'manifest.json'),
                    base_semantic_sha256=original['semantic_sha256'], base_validation_hashes=validations,
                    tolerance_retry=retry_record(), expected_generation_counts=COUNTS,
                    expected_poisson_solves=21,
                    implementation_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT)).decode().strip(),
                    working_tree_status=subprocess.check_output(['git', 'status', '--short'], cwd=str(ROOT)).decode(),
                    validation_reuse='Unchanged base model, weights, camera, prompt, guidance and runtime gates; new scope/runner checks below.',
                    cpu_tests=dict(passed=True, tests=result.testsRun))
    write(OUT / 'manifest.json', manifest)
    with tarfile.open(str(OUT / 'history/frozen-source.tar.gz'), 'w:gz') as archive:
        for name in source:
            archive.add(str(ROOT / name), arcname=name)
    write(OUT / 'validation/cpu.json', dict(passed=True, tests=result.testsRun,
          source_sha256=digest(source), manifest_sha256=sha(OUT / 'manifest.json'), job=os.environ['SLURM_JOB_ID']))
    freeze_check()
    preserved()
    base_validation()
    print('FIREFLY_RETRY_READY', rows[0]['case_id'], OLD_RTOL, '->', NEW_RTOL, flush=True)


if __name__ == '__main__':
    main()
