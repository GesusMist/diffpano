"""One global %5 array; missing images determine work. Evidence only in stdout."""
import argparse
from studies.all_prompts.common import *
from studies.all_prompts.summary import accounting

def main(job,code_hash,prompt_hash,submit=False,retry_failed=False):
    require_validation(job,code_hash,prompt_hash)
    with lock('global-submission'):
        matrix=rows()
        queue=shell_command(['squeue','-u','shig','-h','-o','%i|%j|%T|%R'])
        attempts,raw=accounting()
        emit('PRE_SUBMISSION_STATE',squeue=queue,sacct=raw)
        active=[line for line in queue.splitlines() if '|allp21-run|' in line]
        if active:
            emit('EXISTING_ARRAY_NO_DUPLICATE',jobs=active);return
        existing=[r for r in matrix if image_valid(r['output'],True)]
        missing=[r for r in matrix if r not in existing]
        emit('SUBMISSION_COUNTS',prompt_count=21,total_required=210,already_valid=len(existing),missing_new=len(missing),
            erp_missing=sum(r['projection']=='erp' for r in missing),cea_missing=sum(r['projection']=='cea' for r in missing),
            spherediff_missing=sum(r['method']=='spherediff' for r in missing),expected_concurrency=5)
        for r in missing:
            previous=attempts.get(r['index'],[])
            if previous:
                emit('PREVIOUS_ATTEMPT',row=r,attempts=previous)
                for a in previous:
                    for suffix in ('out','err'):
                        p=REPO/'logs'/('allp21-run.'+a['job']+'.'+suffix)
                        if p.exists():
                            with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3000));tail=f.read().decode('utf-8',errors='replace')
                            emit('PRE_RETRY_LOG',job=a['job'],stream=suffix,tail=tail)
                if not retry_failed:raise RuntimeError('Inspect the recorded failures before explicit --retry-failed')
        if not missing:return
        indices=','.join(str(r['index']) for r in missing)+'%5'
        cmd=['sbatch','--parsable','--array='+indices,str(STUDY/'run.slurm'),
            '--validation-job',str(job),'--code-hash',code_hash,'--prompt-hash',prompt_hash]
        emit('SUBMISSION_INTENT',command=cmd,missing_indices=[r['index'] for r in missing],dry_run=not submit)
        if not submit:return
        # On an ambiguous sbatch failure, inspect squeue/sacct; never automatically retry.
        array=shell_command(cmd).split(';')[0];assert array.isdigit(),array
        emit('ARRAY_SUBMITTED',job=array,max_concurrency=5,cases=len(missing))
        final=shell_command(['sbatch','--parsable','--dependency=afterany:'+array,str(STUDY/'summary.slurm')]).split(';')[0]
        emit('FINAL_TEXT_SUMMARY_SUBMITTED',job=final,dependency=array)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--validation-job',required=True);p.add_argument('--code-hash',required=True)
    p.add_argument('--prompt-hash',required=True);p.add_argument('--submit',action='store_true');p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args();main(a.validation_job,a.code_hash,a.prompt_hash,a.submit,a.retry_failed)
