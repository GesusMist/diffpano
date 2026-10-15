"""Execute one full, fresh Firefly trajectory with a declared tolerance change."""
import argparse
import fcntl
import time
import traceback
from studies.mean_max_firefly_retry.common import *


def run(retry=False):
    assert os.environ.get('SLURM_JOB_ID'), 'Generation requires a Slurm GPU allocation'
    manifest = freeze_check()
    row = manifest['rows'][0]
    evidence = read(OUT / 'validation/cpu.json')
    assert evidence['passed'] and evidence['source_sha256'] == digest(manifest['source_hashes'])
    assert evidence['manifest_sha256'] == sha(OUT / 'manifest.json')
    validation = base_validation()
    preserved()
    directory = folder(row)
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if complete(row):
        print('VALID_CACHE', row['case_id'], flush=True)
        return
    if (directory / 'status.json').exists():
        if not retry:
            raise RuntimeError('Incomplete retry case requires explicit --retry; tolerance will not change')
        history = directory / 'attempts' / str(time.time_ns())
        history.mkdir(parents=True)
        for name in ('status.json', 'metadata.json', 'progress.json', 'config.json'):
            path = directory / name
            if path.exists():
                os.replace(str(path), str(history / name))
    job = os.environ['SLURM_JOB_ID']
    write(directory / 'status.json', dict(state='running', job=job,
          semantic_sha256=row['semantic_sha256'], tolerance_retry=retry_record()))
    try:
        from studies.mean_max_firefly_retry.runtime import generate
        metadata = generate(row, directory, validation)
        freeze_check()
        preserved()
        metadata.update(case={k: v for k, v in row.items() if k != 'semantic'},
                        semantic_sha256=row['semantic_sha256'], source_sha256=row['source_sha256'],
                        source_hashes=manifest['source_hashes'], implementation_revision=manifest['implementation_revision'],
                        reused=False, job=job, tolerance_retry=retry_record())
        write(directory / 'metadata.json', metadata)
        write(directory / 'status.json', dict(state='complete', job=job,
              semantic_sha256=row['semantic_sha256'], metadata_sha256=sha(directory / 'metadata.json'),
              image_sha256=sha(directory / 'final.png'), tolerance_retry=retry_record()))
        assert complete(row)
        print('CASE_COMPLETE', row['case_id'], flush=True)
    except BaseException as error:
        write(directory / 'status.json', dict(state='failed', job=job, semantic_sha256=row['semantic_sha256'],
              error=traceback.format_exc(), solver=getattr(error, 'diagnostics', None), tolerance_retry=retry_record()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retry', action='store_true')
    run(parser.parse_args().retry)
