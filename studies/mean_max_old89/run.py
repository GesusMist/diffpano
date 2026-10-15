"""One exact matrix row, using the existing ordinary_run trajectory."""
import argparse,fcntl,traceback,time
from studies.mean_max_old89.common import *


def run(index,retry=False):
    assert os.environ.get('SLURM_JOB_ID'),'Generation requires a Slurm GPU allocation'
    manifest=freeze_check();row=manifest['rows'][index]
    for name in ('cpu.json','weights.json','metrics-cpu.json'):gate(name)
    validation=gate('gpu.json');preserved()
    directory=folder(row);directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if complete(row):print('VALID_CACHE',row['case_id'],flush=True);return
    if (directory/'status.json').exists():
        if not retry:raise RuntimeError('Incomplete case requires explicit --retry')
        history=directory/'attempts'/str(time.time_ns());history.mkdir(parents=True)
        for name in ('status.json','metadata.json','progress.json','config.json'):
            path=directory/name
            if path.exists():os.replace(str(path),str(history/name))
    job=os.environ['SLURM_JOB_ID'];write(directory/'status.json',dict(state='running',job=job,semantic_sha256=row['semantic_sha256']))
    try:
        from studies.mean_max_old89.runtime import generate
        metadata=generate(row,directory,validation)
        freeze_check();preserved()
        metadata.update(case={k:v for k,v in row.items() if k!='semantic'},semantic_sha256=row['semantic_sha256'],source_sha256=row['source_sha256'],
            source_hashes=manifest['source_hashes'],implementation_revision=manifest['implementation_revision'],reused=False,job=job)
        write(directory/'metadata.json',metadata)
        write(directory/'status.json',dict(state='complete',job=job,semantic_sha256=row['semantic_sha256'],metadata_sha256=sha(directory/'metadata.json'),image_sha256=sha(directory/'final.png')))
        assert complete(row);print('CASE_COMPLETE',row['case_id'],flush=True)
    except BaseException as error:
        write(directory/'status.json',dict(state='failed',job=job,semantic_sha256=row['semantic_sha256'],error=traceback.format_exc(),solver=getattr(error,'diagnostics',None)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int);p.add_argument('--retry',action='store_true');a=p.parse_args()
    run(int(os.environ['SLURM_ARRAY_TASK_ID']) if a.index is None else a.index,a.retry)
