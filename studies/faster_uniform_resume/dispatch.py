"""Run one whitelisted missing case using the unchanged scientific entry point."""
import argparse
from datetime import datetime, timezone
import fcntl
import os
import socket
import time
import traceback

from studies.layout24 import common as original
from studies.faster_uniform_resume.common import OUT, PLAN, ACCOUNT, load_plan


def validate_selection(plan, stage, index):
    if index not in plan['pending_indices'] or index in plan['protected_indices']:
        raise ValueError('Case is outside the authorized missing Uniform subset')
    allowed = ([plan['profile_index']] if stage == 'profile' else
               plan['bulk_indices'] if stage == 'bulk' else [])
    if index not in allowed:
        raise ValueError('Case is outside this launch stage')


def safe_to_generate(row, complete=original.complete, folder=original.folder):
    if complete(row, verify=True):
        return False
    directory = folder(row)
    if (directory / 'final.png').exists() or (directory / 'final.png').is_symlink():
        raise ValueError('Refusing to overwrite an existing final image')
    if (directory / 'status.json').exists() or (directory / 'status.json').is_symlink():
        raise ValueError('Partial or failed case requires explicit review before retry')
    return True


def check_profile(plan, manifest):
    gate = original.read(OUT / 'profile-complete.json')
    row = manifest['rows'][plan['profile_index']]
    assert gate['passed'] and gate['index'] == row['index']
    assert gate['plan_sha256'] == original.sha(PLAN)
    assert gate['image_sha256'] == original.sha(original.folder(row) / 'final.png')
    assert gate['metadata_sha256'] == original.sha(original.folder(row) / 'metadata.json')
    assert original.complete(row, verify=True), 'FASTER profile no longer validates'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('profile', 'bulk'), required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    plan = load_plan()
    index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    validate_selection(plan, args.stage, index)
    assert os.environ.get('SLURM_JOB_ID'), 'Use a Slurm GPU allocation'
    assert os.environ.get('SLURM_CLUSTER_NAME', '').lower() == 'faster'
    assert os.environ.get('SLURM_JOB_ACCOUNT') == ACCOUNT
    manifest = original.freeze_check()
    row = manifest['rows'][index]
    assert row['method'] == 'diffpano' and row['layout'].startswith('uniform')
    assert row['semantic_sha256'] == next(
        case['semantic_sha256'] for case in plan['pending_cases'] if case['index'] == index)
    if args.stage == 'bulk':
        check_profile(plan, manifest)
    execution = OUT / 'execution'
    with (execution / ('case-%03d.lock' % index)).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        generate = safe_to_generate(row)
        started = time.monotonic()
        provenance = dict(index=index, case_id=row['case_id'], stage=args.stage,
                          job=os.environ['SLURM_JOB_ID'],
                          array_job=os.environ.get('SLURM_ARRAY_JOB_ID'),
                          cluster=os.environ['SLURM_CLUSTER_NAME'], hostname=socket.gethostname(),
                          plan_sha256=original.sha(PLAN),
                          started_utc=datetime.now(timezone.utc).isoformat())
        path = execution / ('case-%03d-job-%s.json' % (index, os.environ['SLURM_JOB_ID']))
        original.write(path, dict(provenance, state='running'))
        try:
            import torch
            assert torch.cuda.is_available()
            assert torch.cuda.device_count() == 1
            provenance['gpu_name'] = torch.cuda.get_device_name(0)
            assert 'A100' in provenance['gpu_name']
            if generate:
                print('RESUME_CASE', index, row['case_id'], flush=True)
                from studies.layout24.run import run
                run(index, retry=False)
            assert original.complete(row, verify=True)
            load_plan()
            metadata = original.read(original.folder(row) / 'metadata.json')
            result = dict(provenance, state='complete', generated=generate,
                          elapsed_seconds=time.monotonic() - started,
                          image_sha256=original.sha(original.folder(row) / 'final.png'),
                          metadata_sha256=original.sha(original.folder(row) / 'metadata.json'),
                          runtime={key: metadata.get(key) for key in (
                              'setup_seconds', 'generation_seconds', 'solver_seconds',
                              'peak_allocated_gib', 'generation_peak_allocated_gib', 'gpu')})
            original.write(path, result)
            if args.stage == 'profile':
                original.write(OUT / 'profile-complete.json', dict(result, passed=True))
            print('FASTER_RESUME_COMPLETE', index, result['elapsed_seconds'], flush=True)
        except BaseException:
            original.write(path, dict(provenance, state='failed',
                                      elapsed_seconds=time.monotonic() - started,
                                      error=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
