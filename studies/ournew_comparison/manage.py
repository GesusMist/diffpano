"""Login-safe status, dry run and missing-case-only Slurm submission."""
import argparse,fcntl,subprocess,collections
from studies.ournew_comparison.common import *


def status():
    if not (OUT/'manifest.json').exists():
        print('Manifest not prepared. Expected: 28 new cases; none submitted by this manager.');return
    m=read(OUT/'manifest.json');counts=collections.Counter();failed=[]
    for row in m['rows']:
        path=folder(row)/'status.json';s=read(path) if path.exists() else dict(state='missing')
        state='complete' if complete(row) else s['state']
        if s['state']=='complete' and not complete(row):state='invalid'
        counts[state]+=1
        if state in ('failed','invalid'):failed.append(dict(index=row['index'],case_id=row['case_id'],state=state,error=s.get('error','')[-1500:]))
    print(json.dumps(dict(expected=28,reused=0,new=28,counts=dict(counts),failed=failed),indent=2))
    for name in ('cpu.json','weights.json','metrics-cpu.json','gpu.json','official.json'):
        p=OUT/'validation'/name;print('GATE',name,read(p).get('passed') if p.exists() else 'pending')
    if (OUT/'jobs.json').exists():print('JOBS',json.dumps(read(OUT/'jobs.json'),indent=2))
    if (OUT/'evaluation/status.json').exists():print('METRICS',json.dumps(read(OUT/'evaluation/status.json'),indent=2))
    print(subprocess.check_output(['squeue','-u','shig','-o','%.20i %.28j %.10T %.10M %.20R'],universal_newlines=True))


def launch(dry_run=False,retry=False):
    m=freeze_check();preserved()
    for n in ('cpu.json','weights.json','metrics-cpu.json','gpu.json','official.json'):gate(n)
    lock=(OUT/'submission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    queue=subprocess.check_output(['squeue','-u','shig','-h','-r','-o','%i|%j|%T|%b'],universal_newlines=True)
    active=[line.split('|') for line in queue.splitlines()]
    assert not any(x[1].startswith('ournew-') for x in active),'Suite jobs already active; inspect status before another submission'
    missing=[]
    for row in m['rows']:
        if complete(row):continue
        path=folder(row)/'status.json'
        if path.exists() and not retry:raise RuntimeError('Use --retry for an incomplete/failed existing case: '+row['case_id'])
        missing.append(row['index'])
    unrelated=[x[0] for x in active if 'gpu' in x[3]]
    print(json.dumps(dict(missing_indices=missing,missing_cases=[m['rows'][i]['case_id'] for i in missing],generation_throttle=2,wait_for_unrelated_gpu_jobs=unrelated,dry_run=dry_run),indent=2))
    if dry_run:return
    for d in ('generation','evaluation','reporting'): (OUT/'logs'/d).mkdir(parents=True,exist_ok=True)
    def submit(args):
        value=subprocess.check_output(['sbatch','--parsable']+args,cwd=str(ROOT),universal_newlines=True).strip().split(';')[0]
        assert value.isdigit(),value
        return value
    jobs={}
    if missing:
        args=['--array='+','.join(map(str,missing))+'%2']
        if unrelated:args+=['--dependency=afterany:'+':'.join(unrelated)]
        args+=['studies/ournew_comparison/generate.slurm']
        if retry:args+=['--retry']
        jobs['generation']=submit(args)
    deps=['--dependency=afterany:'+jobs['generation']] if missing else []
    jobs['evaluation']=submit(deps+['studies/ournew_comparison/evaluate.slurm'])
    jobs['report']=submit(['--dependency=afterany:'+jobs['evaluation'],'studies/ournew_comparison/report.slurm'])
    write(OUT/'jobs.json',dict(jobs=jobs,submitted_indices=missing,retry=retry,max_concurrent_generation_gpus=2,manifest_sha256=sha(OUT/'manifest.json')))
    print('SUBMITTED',json.dumps(jobs),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('status','launch'));p.add_argument('--dry-run',action='store_true');p.add_argument('--retry',action='store_true');a=p.parse_args()
    status() if a.action=='status' else launch(a.dry_run,a.retry)
