"""One global five-GPU production array; final image/config pairs are completion keys."""
import argparse
from .common import *
from .report import attempts,report

def launch(submit=False,retry=False):
    validation=validation_gate()
    with lock('camera-patching-submission'):
        queue=shell_command(['squeue','-u','shig','-h','-o','%i|%j|%T|%R'])
        emit('QUEUE',text=queue)
        prior=attempts();missing=[]
        for row in rows():
            done=complete(row,verbose=True)
            status='complete' if done else 'blocked' if row['strategy'] in validation['blocked_strategies'] else 'missing'
            emit('REQUIRED_ROW',**row,status=status)
            if status=='missing':missing.append(row)
        if not missing:return
        if any('|campatch-run|' in line for line in queue.splitlines()):
            emit('ACTIVE_ARRAY_NO_DUPLICATE',queue=queue);return
        for row in missing:
            previous=prior.get(row['index'],[])
            if previous:
                for a in previous:
                    for suffix in ('out','err'):
                        path=REPO/'logs'/('campatch-run.'+a['job']+'.'+suffix)
                        emit('PRE_RETRY_LOG',row=row,attempt=a,stream=suffix,tail=path.read_text()[-3500:] if path.exists() else '')
                if submit and not retry:raise RuntimeError('Existing attempts require inspected --retry-failed; no automatic quality retries')
        if submit:
            for name in BACKENDS:preflight_gate(name)
            gpu=shell_command(['squeue','-u','shig','-p','gpu','-h','-o','%i|%j|%T'])
            assert not gpu,('Wait for existing GPU work before a new global %5 array',gpu)
        indices=','.join(str(r['index']) for r in missing)+'%5'
        cmd=['sbatch','--parsable','--array='+indices,str(STUDY/'run.slurm')]
        emit('SUBMISSION_INTENT',command=cmd,missing=len(missing),max_concurrent_gpus=5,dry_run=not submit)
        if submit:
            job=shell_command(cmd).split(';')[0];assert job.isdigit(),job
            emit('ARRAY_SUBMITTED',job=job,indices=[r['index'] for r in missing],max_concurrency=5)
            final=shell_command(['sbatch','--parsable','--dependency=afterany:'+job,str(STUDY/'report.slurm')]).split(';')[0]
            emit('REPORT_SUBMITTED',job=final,array=job)

if __name__=='__main__':
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',action='store_true');mode.add_argument('--submit',action='store_true');mode.add_argument('--status',action='store_true')
    p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    report() if a.status else launch(a.submit,a.retry_failed)
