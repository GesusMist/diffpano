"""One missing control, guarded by immutable provenance and an exclusive lock."""
import argparse,fcntl,traceback,time
from studies.mean_max_controls.common import *


def run(index,retry=False):
    assert os.environ.get('SLURM_JOB_ID'),'Generation requires a Slurm GPU allocation'
    manifest=freeze_check();row=manifest['rows'][index]
    evidence=validation();base.preserved()
    directory=folder(row);directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if complete(row):print('VALID_CACHE',row['case_id'],flush=True);return
    if (directory/'status.json').exists():
        if not retry:raise RuntimeError('Incomplete case requires explicit --retry')
        history=directory/'attempts'/str(time.time_ns());history.mkdir(parents=True)
        for name in ('status.json','metadata.json','progress.json','config.json'):
            path=directory/name
            if path.exists():os.replace(str(path),str(history/name))
    job=os.environ['SLURM_JOB_ID']
    write(directory/'status.json',dict(state='running',job=job,semantic_sha256=row['semantic_sha256']))
    try:
        from studies.mean_max_controls.runtime import generate
        metadata=generate(row,directory,evidence)
        freeze_check();base.preserved()
        metadata.update(case={key:value for key,value in row.items() if key!='semantic'},semantic_sha256=row['semantic_sha256'],
            source_sha256=row['source_sha256'],source_hashes=manifest['source_hashes'],
            implementation_revision=manifest['implementation_revision'],reused=False,job=job)
        write(directory/'metadata.json',metadata)
        write(directory/'status.json',dict(state='complete',job=job,semantic_sha256=row['semantic_sha256'],
            metadata_sha256=sha(directory/'metadata.json'),image_sha256=sha(directory/'final.png')))
        assert complete(row);print('CASE_COMPLETE',row['case_id'],flush=True)
    except BaseException as error:
        write(directory/'status.json',dict(state='failed',job=job,semantic_sha256=row['semantic_sha256'],
            error=traceback.format_exc(),solver=getattr(error,'diagnostics',None)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int);parser.add_argument('--retry',action='store_true');args=parser.parse_args()
    run(int(os.environ['SLURM_ARRAY_TASK_ID']) if args.index is None else args.index,args.retry)
