"""Guarded execution of one authorized case; profile outputs count toward 48."""
import argparse
from datetime import datetime, timezone
import fcntl
import os
import socket
import time
import traceback

from studies.refinement28 import common


def validate_selection(stage, index):
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError('An integer study index is required')
    allowed = common.PROFILE_INDICES if stage == 'profile' else range(2, 48) if stage == 'bulk' else ()
    if index not in allowed:
        raise ValueError('Case is outside this launch stage')


def require_allocation():
    assert os.environ.get('SLURM_JOB_ID'), 'Use a Slurm GPU allocation'
    assert os.environ.get('SLURM_CLUSTER_NAME', '').lower() == 'faster', 'Use the FASTER cluster'
    assert os.environ.get('SLURM_JOB_ACCOUNT') == common.ACCOUNT, 'Incorrect Slurm account'
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one GPU is required'
    assert 'A100' in torch.cuda.get_device_name(0), 'Use the profiled A100 GPU type'


def safe_to_generate(row, retry=False):
    if common.complete(row, verify=True):
        return False
    directory = common.folder(row)
    for name in common.ARTIFACT_NAMES:
        if name.endswith('.png') and ((directory / name).exists() or (directory / name).is_symlink()):
            raise ValueError('Refusing to overwrite an existing image: ' + str(directory / name))
    prior = [path for path in directory.iterdir() if path.name != 'case.lock'] if directory.exists() else []
    if prior and not retry:
        raise ValueError('Partial or failed case requires explicit --retry')
    if retry and (directory / 'status.json').exists():
        status = common.read(directory / 'status.json')
        if status.get('state') == 'running':
            raise ValueError('Case still says running; inspect the job and mark failure before explicit retry')
    return True


def run(index, stage='profile', retry=False):
    validate_selection(stage, index)
    require_allocation()
    manifest = common.freeze_check()
    common.validation_gate(manifest)
    common.preserved()
    row = manifest['rows'][index]
    if stage == 'bulk':
        assert all(common.complete(manifest['rows'][value], verify=True)
                   for value in common.PROFILE_INDICES), 'Both actual profile cases must complete before bulk'
    directory = common.folder(row)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'case.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not safe_to_generate(row, retry=retry):
            print('VALID_CACHE', row['case_id'], flush=True)
            return common.read(directory / 'metadata.json')
        started = time.monotonic()
        provenance = dict(index=index, case_id=row['case_id'], stage=stage,
                          job=os.environ['SLURM_JOB_ID'],
                          array_job=os.environ.get('SLURM_ARRAY_JOB_ID'),
                          cluster=os.environ['SLURM_CLUSTER_NAME'],
                          account=os.environ['SLURM_JOB_ACCOUNT'], hostname=socket.gethostname(),
                          started_utc=datetime.now(timezone.utc).isoformat(),
                          semantic_sha256=row['semantic_sha256'], source_sha256=row['source_sha256'],
                          manifest_sha256=common.sha(common.OUT / 'manifest.json'))
        if (directory / 'status.json').exists():
            prior = common.read(directory / 'status.json')
            common.write(directory / ('attempt-before-' + str(time.time_ns()) + '.json'), prior)
        common.write(directory / 'status.json', dict(provenance, state='running'))
        try:
            from studies.refinement28.runtime import generate
            metadata = common.safe(generate(row, directory))
            metadata.update(semantic_sha256=row['semantic_sha256'], source_sha256=row['source_sha256'],
                            execution=provenance, job=provenance['job'], reused=False)
            common.validate_metadata(row, metadata)
            current = common.freeze_check()
            assert current == manifest, 'Manifest changed during generation'
            common.preserved()
            for name, expected in metadata['artifacts'].items():
                assert common.sha(directory / name) == expected, 'Runtime artifact hash mismatch'
            common.write(directory / 'metadata.json', metadata)
            common.write(directory / 'status.json', dict(
                provenance, state='complete', elapsed_seconds=time.monotonic() - started,
                image_sha256=metadata['artifacts']['final.png'],
                metadata_sha256=common.sha(directory / 'metadata.json'), artifacts=metadata['artifacts']))
            assert common.complete(row, verify=True), 'Post-write completion verification failed'
            print('CASE_COMPLETE', row['case_id'], metadata['counts'],
                  'seconds', time.monotonic() - started, flush=True)
            return metadata
        except BaseException as error:
            common.write(directory / 'status.json', dict(
                provenance, state='failed', elapsed_seconds=time.monotonic() - started,
                error=traceback.format_exc(), solver=getattr(error, 'diagnostics', None)))
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int)
    parser.add_argument('--stage', choices=('profile', 'bulk'), required=True)
    parser.add_argument('--retry', action='store_true')
    args = parser.parse_args()
    index = int(os.environ['SLURM_ARRAY_TASK_ID']) if args.index is None else args.index
    run(index, stage=args.stage, retry=args.retry)


if __name__ == '__main__':
    main()
