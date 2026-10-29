"""One locked submission journal and shared generation/evaluation GPU allowance."""
import os
import re
import subprocess
from .common import *


def queue():
    lines = command(['squeue','-h','-r','-u',os.environ.get('USER','shig'),'-o','%i|%j|%T|%b']).splitlines()
    return [dict(zip(('job_id','name','state','resources'),line.split('|',3))) for line in lines if line]


def balance():
    import math
    for line in command(['myproject','-l']).splitlines():
        fields=[part.strip() for part in line.strip().strip('|').split('|')]
        if fields and fields[0]==ACCOUNT:
            value=float(fields[5].replace(',',''))
            if not math.isfinite(value) or value<0:raise ValueError('Invalid account balance')
            return value
    raise ValueError('Account missing from myproject')


def reconcile(run=RUN):
    run=Path(run)
    with locked_state(run/'scheduler.json',dict(jobs={},attempts=[])) as ledger:
        live=queue()
        ledger['queue_snapshot']=live
        for key,job in ledger['jobs'].items():
            matches=[row for row in live if row['job_id'].split('_')[0]==job['job_id']]
            if matches:
                job['scheduler_state']=sorted({m['state'] for m in matches})
            else:
                text=command(['sacct','-j',job['job_id'],'--format=JobID,State,ExitCode,Elapsed,AllocTRES','-n','-P'])
                job['accounting']=text
                job['scheduler_state']=text.split('|')[1] if '|' in text else 'unknown'
            job['last_checked_utc']=now()
        ledger['updated_utc']=now()
        return dict(ledger)


def submit(run,key,script,role='cpu',script_args=(),options=(),reserve_su=0.):
    run=Path(run)
    assert role in ('cpu','generation','evaluation','api')
    with locked_state(run/'scheduler.json',dict(jobs={},attempts=[])) as ledger:
        if key in ledger['jobs']:
            return ledger['jobs'][key]
        pending=[a for a in ledger['attempts'] if a['key']==key and a['state'] in ('submission_intent','ambiguous_submission')]
        if pending:
            raise RuntimeError('Reconcile prior ambiguous submission before retry: '+key)
        jobs=queue()
        known={j['job_id']:j for j in ledger['jobs'].values()}
        generation=[];evaluation=[]
        for job in jobs:
            if 'gpu' not in job['resources'].lower():continue
            record=known.get(job['job_id'].split('_')[0],{})
            (evaluation if record.get('role')=='evaluation' else generation).append(job)
        dependent=any(str(option).startswith('--dependency=') for option in options)
        if role=='generation' and generation and not dependent:
            raise RuntimeError('Generation allocations already active/pending; use a recorded dependency: '+str(generation))
        if role=='evaluation' and evaluation and not dependent:
            raise RuntimeError('Evaluation GPU already reserved')
        # Each generation array is capped at two tasks, and distinct generation
        # arrays must be chained. A conservative external-GPU check prevents overlap.
        running_generation=[job for job in generation if job['state'] in ('RUNNING','COMPLETING','CONFIGURING')]
        if role=='evaluation' and len(running_generation)>2 and not dependent:
            raise RuntimeError('External GPU usage exceeds shared study cap')
        available=balance()
        if available < reserve_su:
            raise RuntimeError('Insufficient allocation: available %.2f SU, reservation %.2f SU'%(available,reserve_su))
        logs=run/'logs';logs.mkdir(exist_ok=True)
        name='pev2-'+key
        args=['sbatch','--parsable','--job-name='+name,'--output='+str(logs/(name+'-%A_%a.log')),'--chdir='+str(ROOT)]+list(options)+[str(script)]+list(script_args)
        intent=dict(key=key,role=role,state='submission_intent',command=args,created_utc=now(),available_su=available,reservation_su=reserve_su)
        ledger['attempts'].append(intent)
        write(run/'scheduler.json',ledger)
        env={k:v for k,v in os.environ.items() if k!='OPENROUTER_API_KEY'}
        try:
            result=subprocess.run(args,cwd=str(ROOT),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
            response=result.stdout.strip().split(';')[0]
            intent.update(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
            if result.returncode or not re.fullmatch(r'[0-9]+',response):
                intent['state']='rejected' if 'Batch job submission failed' in result.stderr or 'sbatch: error:' in result.stderr else 'ambiguous_submission'
                write(run/'scheduler.json',ledger)
                raise RuntimeError('Scheduler submission did not return a valid job ID; inspect journal')
        except BaseException:
            if intent['state']=='submission_intent':intent['state']='ambiguous_submission'
            write(run/'scheduler.json',ledger)
            raise
        intent.update(state='submitted',job_id=response)
        record=dict(key=key,job_id=response,role=role,submitted_utc=now(),command=args,scheduler_state='submitted',output_verification='pending',evaluation='pending')
        ledger['jobs'][key]=record
        return record
