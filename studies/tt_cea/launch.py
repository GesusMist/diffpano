"""Durable dry-run-first Slurm launcher with five backend lanes."""
import fcntl
import os
import subprocess
from pathlib import Path
from studies.tt_cea.common import *

LEDGER=ROOT.parent/'seed0-v1-submissions.json'

def submission_environment():
    # activate_venv's /usr/bin/env python3 helper must not resolve to an inherited
    # venv Python after the batch script purges its required shared libraries.
    env=dict(os.environ);venv=env.pop('VIRTUAL_ENV',None);env.pop('VIRTUAL_ENV_PROMPT',None)
    if venv:env['PATH']=':'.join(p for p in env.get('PATH','').split(':') if p.rstrip('/')!=str(Path(venv)/'bin'))
    return env

def command(args):
    return subprocess.check_output(args,text=True,cwd=REPO,
        env=submission_environment() if args[0]=='sbatch' else None).strip()

def inspect_state(ledger):
    queue=command(['squeue','-u',os.environ.get('USER','shig'),'-h','-o','%i|%j|%T'])
    ids=[x['job_id'] for x in ledger['attempts'] if x.get('job_id')]
    accounting=command(['sacct','-n','-P','--format=JobIDRaw,JobName,State,ExitCode','-j',','.join(ids)]) if ids else command(['sacct','-u',os.environ.get('USER','shig'),'-n','-P','--format=JobIDRaw,JobName,State,ExitCode','-S','today'])
    statuses={str(p.relative_to(ROOT)):read(p) for p in ROOT.glob('samples/*/*/*/status.json')}
    return dict(squeue=queue,sacct=accounting,outputs=statuses)

def save_ledger(value):
    atomic(LEDGER,value)
    if (ROOT/'manifest.json').exists():atomic(ROOT/'execution.json',value)

def phase_plan(phase,ledger,development=False):
    latest={a['key']:a for a in ledger['attempts'] if a.get('job_id')}
    def jid(key):return latest.get(key,{}).get('job_id')
    plans=[]
    if phase=='validate':
        key='development-validation' if development else 'validation'
        plans.append(dict(key=key,name='ttcea-'+key,script='validate.slurm',args=['--development'] if development else [],deps=[]))
    elif phase=='preflight':
        plans.append(dict(key='geometry',name='ttcea-geometry',script='preflight.slurm',args=['--phase','geometry'],deps=[('afterok',jid('validation'))]))
        for n in BACKENDS:plans.append(dict(key='preflight-'+n,name='ttcea-preflight-'+n,script='preflight.slurm',args=['--phase','backend','--backend',n],dep_keys=[('afterany','geometry')],deps=[]))
    elif phase in ('time-travel','projection'):
        cases=('T1','TB') if phase=='time-travel' else ('C0','P0','C1')
        wave1=[a['job_id'] for a in ledger['attempts'] if a.get('case') in ('T1','TB') and a.get('job_id')]
        for n in BACKENDS:
            previous=None
            for prompt in PROMPTS:
                for case in cases:
                    key=case+'-'+n+'-'+prompt
                    deps=[('afterok',jid('preflight-'+n))] if previous is None else []
                    if phase=='projection' and previous is None:deps += [('afterany',j) for j in wave1]
                    plans.append(dict(key=key,name='ttcea-'+key,script='run.slurm',args=['--backend',n,'--prompt',prompt,'--case',case],
                        deps=deps,dep_keys=[('afterany',previous)] if previous else [],backend=n,prompt=prompt,case=case))
                    previous=key
    elif phase=='report':
        plans.append(dict(key='report',name='ttcea-report',script='report.slurm',args=[],deps=[('afterany',a['job_id']) for a in ledger['attempts'] if a.get('case') and a.get('job_id')]))
    return plans

def main(args):
    load_settings(args.config);LEDGER.parent.mkdir(parents=True,exist_ok=True)
    with (LEDGER.parent/'seed0-v1-submissions.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=read(LEDGER) if LEDGER.exists() else dict(study='tt-cea-seed0-v1',attempts=[])
        state=inspect_state(ledger)
        if args.phase=='status':print(json.dumps(dict(ledger=ledger,**state),indent=2));return
        plan=phase_plan(args.phase,ledger,args.development)
        print('DRY-RUN PLAN',json.dumps(plan,indent=2),flush=True)
        if not args.submit:return
        if args.phase!='validate':require_gate()
        for task in plan:
            state=inspect_state(ledger)  # Before every individual submission, including retries.
            matches=[a for a in ledger['attempts'] if a['key']==task['key']]
            if matches:
                last=matches[-1]
                if last.get('job_id'):
                    repeat_reason=(args.repeat_validation if args.phase=='validate' else getattr(args,'retry_preflight',None) if args.phase=='preflight' else getattr(args,'repeat_report',None) if args.phase=='report' else None)
                    if repeat_reason:
                        if any(line.split('|')[0]==last['job_id'] for line in state['squeue'].splitlines()):raise RuntimeError('Previous validation still active')
                        terminal=[line.split('|')[2] for line in state['sacct'].splitlines() if line.split('|')[0]==last['job_id']]
                        if not terminal or terminal[0] not in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','NODE_FAIL','OUT_OF_MEMORY'):raise RuntimeError('Previous validation not confirmed terminal')
                        if args.phase=='preflight':
                            log=REPO/'logs'/(last['name']+'.'+last['job_id']+'.err')
                            if terminal[0]=='FAILED' and ('libpython3.11.so.1.0' not in log.read_text()):raise RuntimeError('Preflight retry requires audited infrastructure failure')
                            if (ROOT/'preflight'/((task.get('args',[''])[-1])+'.json')).exists():raise RuntimeError('Preflight result exists; do not overwrite')
                        task['previous_job']=last['job_id'];task['repeat_reason']=repeat_reason
                    else:
                        print('Already submitted; no duplicate',task['key'],last['job_id'],flush=True);continue
                # Resolve an ambiguous submission by durable intent/job name, never resubmit blindly.
                if not last.get('job_id'):
                    found=[line.split('|')[0] for line in state['squeue'].splitlines() if '|'+task['name']+'|' in line]
                    if len(found)==1:last['job_id']=found[0];save_ledger(ledger);continue
                    raise RuntimeError('Unresolved submission intent; inspect squeue/sacct before retry: '+task['key'])
            if task.get('case'):
                try:require_gate(task['backend'],task['case'])
                except (AssertionError,FileNotFoundError) as e:
                    print('BLOCKED BY PREFLIGHT',task['key'],str(e),flush=True);continue
                status=sample_dir(task['backend'],task['prompt'],task['case'])/'status.json'
                if status.exists():raise RuntimeError('Existing sample status requires explicit audit, not automatic retry: '+str(status))
            latest={a['key']:a['job_id'] for a in ledger['attempts'] if a.get('job_id')}
            deps=list(task['deps'])
            if task.get('case'):
                # Follow the last actually submitted case in this backend lane, even when a feature is blocked.
                lane=[a for a in ledger['attempts'] if a.get('case') and a.get('backend')==task['backend'] and a.get('job_id')]
                if lane:deps.append(('afterany',lane[-1]['job_id']))
            for kind,key in task.get('dep_keys',[]):
                if key in latest:deps.append((kind,latest[key]))
            if any(j is None for _,j in deps):raise RuntimeError('Missing prerequisite submission: '+task['key'])
            # Completed jobs may have aged out of slurmctld even though sacct retains them.
            accounted={line.split('|')[0]:line.split('|')[2].split()[0] for line in state['sacct'].splitlines() if len(line.split('|'))>=3}
            active={line.split('|')[0] for line in state['squeue'].splitlines()}
            resolved_deps=[]
            for kind,j in dict.fromkeys(deps):
                outcome=accounted.get(j)
                if j not in active and outcome in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED'):
                    if kind=='afterok' and outcome!='COMPLETED':raise RuntimeError('Failed prerequisite '+j+': '+outcome)
                else:resolved_deps.append((kind,j))
            deps=resolved_deps
            cmd=['sbatch','--parsable','--job-name='+task['name']]
            if deps:cmd+=['--dependency='+','.join(kind+':'+j for kind,j in deps)]
            cmd+=['studies/tt_cea/'+task['script']]+task['args']
            record=dict(task,command=cmd,submission_state='intent',observed_state=state)
            ledger['attempts'].append(record);save_ledger(ledger)
            raw=command(cmd);job=raw.split(';')[0]
            if not job.isdigit():raise RuntimeError('Ambiguous sbatch response: '+raw)
            record.update(job_id=job,submission_state='submitted');save_ledger(ledger)
            print('SUBMITTED',task['key'],job,flush=True)

if __name__=='__main__':
    p=parser(__doc__);p.add_argument('phase',choices=['validate','preflight','time-travel','projection','report','status'])
    p.add_argument('--retry-preflight',help='Audited infrastructure retry reason');p.add_argument('--repeat-report',help='Report refresh reason');p.add_argument('--repeat-validation',help='Reason for a separate validation attempt after confirming the previous job is terminal');p.add_argument('--submit',action='store_true');p.add_argument('--development',action='store_true');main(p.parse_args())
