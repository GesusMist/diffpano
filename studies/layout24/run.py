"""One missing target slot, with immutable resume and explicit retry."""
import argparse,fcntl,time,traceback
from studies.layout24.common import *


def run(index,retry=False):
    assert os.environ.get('SLURM_JOB_ID'),'Generation requires a Slurm GPU allocation'
    manifest=freeze_check();row=manifest['rows'][index];evidence=gate()
    target=next(r for r in read(OUT/'generation_plan.json')['rows'] if r['index']==index)
    if target['state']=='reused':
        assert complete(row);print('VALID_REUSED',row['case_id'],flush=True);return
    assert target['state']=='generate','Blocked row must not be generated'
    directory=folder(row);directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if complete(row):print('VALID_CACHE',row['case_id'],flush=True);return
    if (directory/'status.json').exists():
        if not retry:raise RuntimeError('Incomplete case requires explicit --retry')
        history=directory/'attempts'/str(time.time_ns());history.mkdir(parents=True)
        for name in ('status.json','metadata.json','progress.json','config.json','final.png','final-unclamped-fp32.pt'):
            path=directory/name
            if path.exists():os.replace(str(path),str(history/name))
    job=os.environ['SLURM_JOB_ID'];write(directory/'status.json',dict(state='running',job=job,semantic_sha256=row['semantic_sha256'],reused=False))
    try:
        if row['method']=='diffpano':
            from studies.layout24.runtime import generate
            metadata=generate(row,directory,evidence)
        else:
            from studies.layout24.official import generate
            metadata=generate(row,directory)
        freeze_check()
        metadata.update(case={k:v for k,v in row.items() if k!='semantic'},semantic_sha256=row['semantic_sha256'],source_sha256=row['source_sha256'],
            source_hashes=manifest['source_hashes'],implementation_revision=manifest['implementation_revision'],reused=False,job=job)
        write(directory/'metadata.json',metadata)
        write(directory/'status.json',dict(state='complete',job=job,semantic_sha256=row['semantic_sha256'],metadata_sha256=sha(directory/'metadata.json'),image_sha256=sha(directory/'final.png'),reused=False))
        assert complete(row);print('CASE_COMPLETE',row['case_id'],flush=True)
    except BaseException as error:
        write(directory/'status.json',dict(state='failed',job=job,semantic_sha256=row['semantic_sha256'],error=traceback.format_exc(),solver=getattr(error,'diagnostics',None),reused=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int);p.add_argument('--retry',action='store_true');args=p.parse_args()
    run(int(os.environ['SLURM_ARRAY_TASK_ID']) if args.index is None else args.index,args.retry)
