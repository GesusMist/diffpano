"""Stdlib-only, duplicate-safe Slurm launcher/resumer for the fixed study."""
import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path

REPO=Path('/home/shig/diffpano')
ROOT=REPO/'outputs/bridge-factorial-ruins/20260918'
ACTIVE={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','REQUEUED','RESIZING'}
FAILED={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}


def read(p):return json.loads(Path(p).read_text())


def command(args):return subprocess.check_output(args,cwd=str(REPO),universal_newlines=True).strip()


def queue():
    raw=command(['squeue','-u',os.environ.get('USER','shig'),'--noheader','--format=%i|%j|%T'])
    print('Current queue:\n'+(raw or '(empty)'),flush=True)
    return {v[1]:dict(id=v[0],state=v[2]) for v in (line.split('|') for line in raw.splitlines()) if len(v)==3}


def state(job):
    rows=command(['sacct','-X','-j',str(job),'--noheader','--parsable2','--format=JobID,State']).splitlines()
    found=[r.split('|')[1].split()[0] for r in rows if r.split('|')[0]==str(job)]
    return found[0] if found else 'UNKNOWN'


def complete(folder):
    p=folder/'metadata.json'
    if not p.exists():return False
    m=read(p)
    return (folder/'final_result.png').exists() and ('runtime_seconds' in m)


def submit(phase):
    manifest=read(ROOT/'manifest.json');gate=read(ROOT/'validation.json')
    if not gate['passed']:raise AssertionError('CPU gate required')
    tasks=[]
    if phase=='preflight':
        for name in manifest['models']:
            tasks.append(dict(key='preflight/'+name,name='bfr-pre-'+name,output=ROOT/'preflight'/(name+'.json'),script='slurm/bridge_factorial_cell.slurm',args=['preflight',name],wall='00:25:00'))
    elif phase=='references':
        for name in manifest['references']:
            tasks.append(dict(key='reference/'+name,name='bfr-ref-'+name,output=ROOT/'references'/name,script='slurm/bridge_factorial_reference.slurm',args=[name],wall='03:00:00' if name=='flux' else '01:30:00'))
    else:
        if not all(read(ROOT/'preflight'/(name+'.json'))['passed'] for name in manifest['models']):raise AssertionError('All five real-backend gates required')
        if phase=='remaining' and (not read(ROOT/'pilot-audit.json')['passed'] or not read(ROOT/'execution.json').get('pilot_reviewed')):
            raise AssertionError('Completed numerical and visual pilot review required')
        times={'sd2':(25,40),'sana':(35,50),'flux':(65,90),'sd35':(85,120),'pixeldit':(40,65)}
        for row in manifest['cells']:
            if row['pilot']!=(phase=='pilot'):continue
            mins=times[row['backend']][row['B']]
            tasks.append(dict(key=row['backend']+'/'+row['cell'],name='bfr-'+row['backend']+'-'+row['cell'],output=REPO/row['output'],script='slurm/bridge_factorial_cell.slurm',args=['run',row['backend'],row['cell']],wall='%02d:%02d:00'%(mins//60,mins%60)))
    with (ROOT/'.submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        q=queue();execution=read(ROOT/'execution.json')
        for task in tasks:
            output=task['output']
            done=read(output).get('passed',False) if phase=='preflight' and output.exists() else (complete(output) if phase!='preflight' else False)
            if done:print('Completed; reuse',task['key']);continue
            if task['name'] in q:print('Already active',task['key'],q[task['name']]);continue
            previous=[j for j in execution['jobs'] if j.get('key')==task['key']]
            if previous:
                current=state(previous[-1]['id']);previous[-1]['state']=current
                if current not in FAILED:raise RuntimeError('Refusing uncertain/duplicate submission: '+task['key']+' '+current)
            if output.exists():
                if phase=='preflight':raise RuntimeError('Invalid existing preflight: '+str(output))
                target=ROOT/'failed_attempts'/task['key']/(previous[-1]['id'] if previous else 'unknown')
                target.parent.mkdir(parents=True,exist_ok=True);output.rename(target)
            job=command(['sbatch','--parsable','--job-name='+task['name'],'--time='+task['wall'],task['script']]+task['args']).split(';')[0]
            execution['jobs'].append(dict(id=job,key=task['key'],phase=phase,name=task['name'],state='SUBMITTED',walltime=task['wall']))
            temporary=ROOT/'execution.tmp';temporary.write_text(json.dumps(execution,indent=2)+'\n');temporary.replace(ROOT/'execution.json')
            print('SUBMITTED',job,task['key'],flush=True)


def status():
    manifest=read(ROOT/'manifest.json');q=queue();execution=read(ROOT/'execution.json');counts=dict(expected=80,completed=0,failed=0,pending=0,missing=0)
    for row in manifest['cells']:
        key=row['backend']+'/'+row['cell'];name='bfr-'+row['backend']+'-'+row['cell']
        if complete(REPO/row['output']):counts['completed']+=1
        elif name in q:counts['pending']+=1
        elif any(j.get('key')==key for j in execution['jobs']):counts['failed']+=1
        else:counts['missing']+=1
    print(json.dumps(counts,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','references','pilot','remaining','status']);a=p.parse_args()
    status() if a.phase=='status' else submit(a.phase)
