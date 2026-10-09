"""Archive failed attempts, then activate only the five validated retry rows."""
import argparse, fcntl, shutil
from studies.gradient_refinement_retry.common import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activate', action='store_true')
    args = parser.parse_args()
    lock = (OUT / 'launch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.activate:
        plan = read(RETRY / 'plan.json')
        original = check_scope(plan)
        gate = read(RETRY / 'validation.json')
        assert gate['passed'] and gate['source_hashes'] == plan['source_hashes']
        assert gate['plan_sha256'] == sha(RETRY / 'plan.json')
        assert read(OUT / 'manifest.json') == original, 'Manifest changed since retry preparation'
        updated = copy.deepcopy(original)
        for row in plan['rows']:
            assert read(folder(row) / 'status.json')['state'] == 'failed'
            updated['rows'][row['index']] = row
        updated['tolerance_retries'] = dict(indices=list(INDICES), relative_tolerance=RTOL,
            previous_relative_tolerance=1e-5, absolute_tolerance=1e-7,
            plan=str(RETRY / 'plan.json'), validation=str(RETRY / 'validation.json'),
            note='User-authorized tolerance-only reruns; other 59 rows retain their original settings.')
        write(OUT / 'manifest.json', updated)
        print('ACTIVATED', list(INDICES), flush=True)
        return
    assert not (RETRY / 'plan.json').exists(), 'Retry plan already exists'
    original = read(OUT / 'manifest.json')
    assert original['source_hashes'] == original_source_hashes()
    for gate_name in ('validation.json', 'gpu-smoke.json'):
        gate = read(OUT / gate_name)
        assert gate['passed'] and gate['source_hashes'] == original['source_hashes']
    for index in INDICES:
        status = read(folder(original['rows'][index]) / 'status.json')
        assert status['state'] == 'failed'
        assert status['solver']['failure'] == 'Final FP32 output fails normal equation or reference constraint'
    RETRY.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(OUT / 'manifest.json'), str(RETRY / 'original-manifest.json'))
    sources = retry_source_hashes()
    rows, attempts = [], []
    for index in INDICES:
        old = original['rows'][index]
        directory = folder(old)
        archive = RETRY / 'failed-attempts' / old['key']
        archive.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.name != 'lock':
                target = archive / path.name
                shutil.copy2(str(path), str(target))
                artifacts[str(target)] = sha(target)
                assert artifacts[str(target)] == sha(path)
        status = read(archive / 'status.json')
        attempts.append(dict(index=index, key=old['key'], previous_job=status['job'],
            archived_artifacts=artifacts, failure=status['solver']))
        row = copy.deepcopy(old)
        row['semantic']['fusion']['relative_tolerance'] = RTOL
        row['semantic_sha256'] = digest(row['semantic'])
        row['implementation_sha256'] = digest(sources)
        rows.append(row)
    plan = dict(indices=list(INDICES), rows=rows, source_hashes=sources, failed_attempts=attempts,
        relative_tolerance=RTOL, previous_relative_tolerance=1e-5, absolute_tolerance=1e-7,
        original_manifest_sha256=sha(RETRY / 'original-manifest.json'),
        authorization='User: restart the 5 failed cases with higher tolerance',
        unchanged='All settings except solver relative tolerance; original generation and numerical code remain frozen.')
    write(RETRY / 'plan.json', plan)
    check_scope(plan)
    print('PREPARED', list(INDICES), 'rtol', RTOL, flush=True)

if __name__ == '__main__':
    main()
