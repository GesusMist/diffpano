"""Dry-run-first durable Slurm submissions; never duplicate recorded attempts."""
import argparse
import time
from studies.original_spherediff.common import *


def command(args):
    env=dict(os.environ)
    venv=env.pop('VIRTUAL_ENV',None);env.pop('VIRTUAL_ENV_PROMPT',None)
    if venv:env['PATH']=':'.join(p for p in env.get('PATH','').split(':') if p.rstrip('/')!=str(Path(venv)/'bin'))
    return subprocess.check_output(args,cwd=str(REPO),env=env,universal_newlines=True).strip()


def observations(ledger):
    ids=[a['job_id'] for a in ledger['attempts'] if a.get('job_id')]
    return dict(squeue=command(['squeue','-u',os.environ.get('USER','shig'),'-h','-o','%i|%j|%T']),
        sacct=command(['sacct','-n','-P','--format=JobIDRaw,JobName,State,ExitCode']+(['-j',','.join(ids)] if ids else ['-u','shig','-S','today'])),
        statuses={str(p.relative_to(ROOT)):read(p) for p in ROOT.glob('*/*/seed0/status.json')},
        manifest_sha256=sha(ROOT/'manifest.json') if (ROOT/'manifest.json').exists() else None)


def main(phase,submit):
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'.submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=ROOT/'execution.json';ledger=read(path) if path.exists() else dict(study='original-spherediff',attempts=[])
        obs=observations(ledger)
        if phase=='status':print(json.dumps(obs,indent=2));return
        if phase=='validate':tasks=[dict(key='validate',script='validate.slurm',args=[])]
        elif phase=='report':
            require_gate();tasks=[dict(key='report',script='report.slurm',args=[])]
        else:
            plan=require_gate();tasks=[dict(key=x['backend']+'/'+x['prompt'],script='run.slurm',args=['--backend',x['backend'],'--prompt',x['prompt']]) for x in plan['rows'] if not x['reuse']]
            assert len(tasks)<=2,'Additional cases require a serial dependency plan'
        new=[]
        for task in tasks:
            previous=[a for a in ledger['attempts'] if a['key']==task['key']]
            if previous:
                print('Recorded attempt; no duplicate submission:',task['key'],flush=True);continue
            if task['script']=='run.slurm':
                n,p=task['key'].split('/')
                if folder(n,p).exists():raise RuntimeError('Existing output not resubmitted: '+task['key'])
            task['name']='orig-sphere-'+task['key'].replace('/','-');new.append(task)
        print('DRY-RUN',json.dumps(new,indent=2),flush=True)
        if not submit:return
        for task in new:
            cmd=['sbatch','--parsable','--job-name='+task['name'],str(STUDY/task['script'])]+task['args']
            entry=dict(task,submission_state='intent',observed_state=obs,command=cmd,unix=time.time())
            ledger['attempts'].append(entry);atomic(path,ledger)
            result=command(cmd);job=result.split(';')[0];assert job.isdigit(),result
            entry.update(submission_state='submitted',job_id=job);atomic(path,ledger)
            print('SUBMITTED',task['key'],job,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['validate','generate','report','status']);p.add_argument('--submit',action='store_true')
    a=p.parse_args();main(a.phase,a.submit)
