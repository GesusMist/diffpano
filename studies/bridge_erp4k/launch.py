"""Stdlib-only duplicate-safe submission for the requested 22-cell subset."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO=Path('/home/shig/diffpano');ROOT=REPO/'outputs/bridge-erp4k-ruins/20260918'
FAILED={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
def read(p):return json.loads(p.read_text())
def cmd(args):return subprocess.check_output(args,cwd=str(REPO),universal_newlines=True).strip()
def save(x):
    p=ROOT/'execution.tmp';p.write_text(json.dumps(x,indent=2)+'\n');p.replace(ROOT/'execution.json')
def state(job):
    for line in cmd(['sacct','-X','-j',job,'--noheader','--parsable2','--format=JobID,State']).splitlines():
        a=line.split('|')
        if a[0]==job:return a[1].split()[0]
    return 'UNKNOWN'
def completed(path):
    if path.suffix=='.json':return path.exists() and read(path).get('passed',read(path).get('audit',{}).get('passed',False))
    return (path/'metadata.json').exists() and (path/'final_result.png').exists()
def main(phase):
    with (ROOT/'.submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);x=read(ROOT/'execution.json')
        q=cmd(['squeue','-u',os.environ.get('USER','shig'),'--noheader','--format=%i|%j|%T']);print('Current queue:\n'+(q or '(empty)'),flush=True)
        active={a[1]:a for a in (line.split('|') for line in q.splitlines()) if len(a)==3}
        if phase=='status':
            m=read(ROOT/'manifest.json');done=sum(completed(REPO/r['output']) for r in m['cells']);print('Completed',done,'of',len(m['cells']));return
        tasks=[]
        if phase=='validate':tasks=[dict(key='validation',name='erp4k-validation',script='validate',args=[],output=ROOT/'validation.json',wall='01:00:00')]
        else:
            gate=read(ROOT/'validation.json');assert gate['passed']
            assert all(hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h for p,h in gate['source_hashes'].items())
            m=read(ROOT/'manifest.json')
            if phase=='preflight':
                tasks=[dict(key='preflight/'+b,name='erp4k-pre-'+b,script='cell',args=['preflight',b],output=ROOT/'preflight'/(b+'.json'),wall='00:35:00') for b in m['models']]
            elif phase=='run':
                assert all(read(ROOT/'preflight'/(b+'.json'))['passed'] for b in m['models'])
                times={'sd2':(40,60),'sana':(45,65),'flux':(65,85),'sd35':(80,100),'pixeldit':(45,65)}
                for r in m['cells']:
                    mins=times[r['backend']][r['B']]
                    tasks.append(dict(key=r['backend']+'/'+r['cell'],name='erp4k-'+r['backend']+'-'+r['code'],script='cell',args=['run',r['backend'],r['cell']],output=REPO/r['output'],wall='%02d:%02d:00'%(mins//60,mins%60)))
            elif phase=='report':
                dependencies=[]
                for r in m['cells']:
                    if completed(REPO/r['output']):continue
                    prior=[j for j in x['jobs'] if j['key']==r['backend']+'/'+r['cell']]
                    if not prior or state(prior[-1]['id']) in FAILED:raise RuntimeError('Missing/failed generation before report submission')
                    dependencies.append(prior[-1]['id'])
                tasks=[dict(key='report',name='erp4k-report',script='report',args=[],output=ROOT/'paired-summary.json',wall='01:00:00',dependencies=dependencies)]
        for t in tasks:
            if completed(t['output']):print('Completed; reuse',t['key']);continue
            if t['name'] in active:print('Already active',t['key'],active[t['name']]);continue
            prior=[j for j in x['jobs'] if j['key']==t['key']]
            if prior and state(prior[-1]['id']) not in FAILED:raise RuntimeError('Refusing duplicate/uncertain submission: '+t['key'])
            if t['output'].exists():
                if not prior:raise RuntimeError('Unexpected existing incomplete output: '+str(t['output']))
                dst=ROOT/'failed_attempts'/t['key']/prior[-1]['id'];dst.parent.mkdir(parents=True,exist_ok=True);t['output'].rename(dst)
            args=['sbatch','--parsable','--job-name='+t['name'],'--time='+t['wall']]
            if t.get('dependencies'):args+=['--dependency=afterok:'+':'.join(t['dependencies'])]
            args+=['studies/bridge_erp4k/'+t['script']+'.slurm']+t['args']
            job=cmd(args).split(';')[0];x['jobs'].append(dict(id=job,key=t['key'],phase=phase,name=t['name'],state='SUBMITTED',walltime=t['wall'],dependencies=t.get('dependencies',[])));x['status']=phase+' submitted';save(x);print('SUBMITTED',job,t['key'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['validate','preflight','run','report','status']);main(p.parse_args().phase)
