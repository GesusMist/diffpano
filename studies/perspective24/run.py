"""Group independent batch-one cases in one loaded-model Slurm allocation."""
import argparse,fcntl,time,traceback
from studies.perspective24.common import *

def parse_indices(value):
    result=[]
    for part in value.split(','):
        ends=part.split('-')
        if len(ends)==1:result.append(int(part))
        elif len(ends)==2:result.extend(range(int(ends[0]),int(ends[1])+1))
        else:raise ValueError(value)
    if not result or len(set(result))!=len(result) or any(i<0 or i>=120 for i in result):raise ValueError(value)
    return result

def select_indices(args):
    if args.indices:return parse_indices(args.indices)
    if args.group_size is None:raise ValueError('Use --indices or --group-size with SLURM_ARRAY_TASK_ID')
    assert 1<=args.group_size<=120
    group=int(os.environ['SLURM_ARRAY_TASK_ID']);return list(range(group*args.group_size,min(120,(group+1)*args.group_size)))

def run(indices,retry=False,profile=False):
    assert os.environ.get('SLURM_JOB_ID'),'Generation requires a Slurm GPU allocation'
    manifest=freeze_check();gate('cpu.json');gate('weights.json')
    if not profile:gate('profile.json')
    if profile:assert indices==[0],'The bounded profile occupies actual target slot 0 only'
    locks=[];rows=[];job=os.environ['SLURM_JOB_ID']
    for index in indices:
        row=manifest['rows'][index];directory=folder(row);directory.mkdir(parents=True,exist_ok=True)
        lock=(directory/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        if complete(row):print('VALID_CACHE',row['case_id'],flush=True);continue
        if (directory/'status.json').exists():
            if not retry:raise RuntimeError('Failed/interrupted case requires explicit --retry: '+row['case_id'])
            previous=read(directory/'status.json')
            if previous.get('state')=='running':
                active=subprocess.check_output(['squeue','-h','-j',str(previous['job']),'-o','%i']).decode().strip()
                if active:raise RuntimeError('Case still belongs to an active allocation: '+active)
            history=directory/'attempts'/str(time.time_ns());history.mkdir(parents=True)
            for name in ('status.json','metadata.json','progress.json','config.json','final.png'):
                p=directory/name
                if p.exists():os.replace(str(p),str(history/name))
        rows.append(row)
    if not rows:return
    from studies.perspective24.runtime import load_backend,generate
    b,encoded,setup=load_backend(rows)
    for row in rows:
        directory=folder(row)
        write(directory/'status.json',dict(state='running',job=job,semantic_sha256=row['semantic_sha256']))
        try:
            meta=generate(b,row,encoded.pop(row['index']),directory,setup)
            freeze_check()
            meta.update(case={k:v for k,v in row.items() if k!='semantic'},semantic_sha256=row['semantic_sha256'],source_sha256=row['source_sha256'],
                source_hashes=manifest['source_hashes'],implementation_revision=manifest['implementation_revision'],job=job,reused=False)
            write(directory/'metadata.json',meta)
            write(directory/'status.json',dict(state='complete',job=job,semantic_sha256=row['semantic_sha256'],metadata_sha256=sha(directory/'metadata.json'),image_sha256=sha(directory/'final.png')))
            assert complete(row);print('CASE_COMPLETE',row['index'],row['case_id'],meta['generation_seconds'],flush=True)
            if profile:
                write(OUT/'validation/profile.json',dict(passed=True,index=row['index'],case_id=row['case_id'],counts=meta['counts'],guided_steps=meta['guided_steps'],
                    generation_seconds=meta['generation_seconds'],shared_setup_seconds=setup['seconds'],generation_peak_allocated_gib=meta['generation_peak_allocated_gib'],
                    initialization=meta['initialization'],schedule_sha256=meta['schedule_sha256'],source_sha256=row['source_sha256'],manifest_sha256=sha(OUT/'manifest.json'),job=job))
        except BaseException:
            write(directory/'status.json',dict(state='failed',job=job,semantic_sha256=row['semantic_sha256'],error=traceback.format_exc()));raise
    from studies.perspective24.status import refresh
    refresh()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--indices');parser.add_argument('--group-size',type=int)
    parser.add_argument('--retry',action='store_true');parser.add_argument('--profile',action='store_true');args=parser.parse_args()
    run(select_indices(args),args.retry,args.profile)
