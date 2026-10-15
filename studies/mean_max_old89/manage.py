"""Login-safe status, first-row profile, and missing-only resumable submission."""
import argparse,collections,fcntl,time
from studies.mean_max_old89.common import *


def status():
    if not (OUT/'manifest.json').exists():print('No frozen manifest yet. Expected exactly 90 new trajectories.');return
    m=read(OUT/'manifest.json');counts=collections.Counter();failed=[]
    for row in m['rows']:
        p=folder(row)/'status.json';s=read(p) if p.exists() else dict(state='missing')
        state='complete' if complete(row,verify=False) else 'invalid' if s['state']=='complete' else s['state'];counts[state]+=1
        if state in ('failed','invalid'):failed.append(dict(index=row['index'],case_id=row['case_id'],error=s.get('error','')[-1600:]))
    print(json.dumps(dict(expected=90,new=90,reused=0,counts=dict(counts),failed=failed),indent=2))
    for name in ('cpu.json','weights.json','metrics-cpu.json','gpu.json'):
        p=OUT/'validation'/name;print('GATE',name,read(p).get('passed') if p.exists() else 'pending')
    for p in (OUT/'jobs.json',OUT/'evaluation/status.json'):
        if p.exists():print(str(p),json.dumps(read(p),indent=2))
    print(subprocess.check_output(['squeue','-u','shig','-o','%.20i %.28j %.10T %.10M %.20R'],universal_newlines=True))


def launch(dry_run=False,retry=False,profile=False):
    m=freeze_check();preserved()
    for n in ('cpu.json','weights.json','metrics-cpu.json','gpu.json'):gate(n)
    lock=(OUT/'submission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    queue=subprocess.check_output(['squeue','-u','shig','-h','-r','-o','%i|%j|%T|%b'],universal_newlines=True)
    active=[line.split('|') for line in queue.splitlines()]
    assert not any(x[1].startswith('meanmax-') for x in active),'Suite jobs already active; inspect/wait before another submission'
    rows=m['rows'][:1] if profile else m['rows'];missing=[]
    if not profile:
        assert complete(m['rows'][0]),'Run and validate the first matrix row via profile before the remaining launch'
        meta=read(folder(m['rows'][0])/'metadata.json')
        assert meta['generation_seconds']+meta['setup_seconds']<9000,'First profile approaches 3h walltime: inspect before launching'
        write(OUT/'validation/first-matrix-profile.json',dict(passed=True,index=0,generation_seconds=meta['generation_seconds'],setup_seconds=meta['setup_seconds'],
            selected_walltime_seconds=10800,metadata_sha256=sha(folder(m['rows'][0])/'metadata.json'),counts=meta['counts'],poisson_solves=meta['poisson_solves']))
    for row in rows:
        if complete(row):continue
        if (folder(row)/'status.json').exists() and not retry:raise RuntimeError('Use --retry for incomplete/failed case: '+row['case_id'])
        missing.append(row['index'])
    unrelated=[x[0] for x in active if 'gpu' in x[3]]
    print(json.dumps(dict(missing_indices=missing,generation_throttle=1 if profile else 2,wait_for_unrelated_gpu_jobs=unrelated,dry_run=dry_run),indent=2))
    if dry_run:return
    def submit(args):
        value=subprocess.check_output(['sbatch','--parsable']+args,cwd=str(ROOT),universal_newlines=True).strip().split(';')[0]
        assert value.isdigit(),value
        return value
    jobs={}
    if missing:
        args=['--array='+','.join(map(str,missing))+('%1' if profile else '%2')]
        if unrelated:args+=['--dependency=afterany:'+':'.join(unrelated)]
        args+=['studies/mean_max_old89/generate.slurm']
        if retry:args+=['--retry']
        jobs['generation']=submit(args)
    deps=['--dependency=afterany:'+jobs['generation']] if missing else []
    jobs['evaluation']=submit(deps+['studies/mean_max_old89/evaluate.slurm']+(['--allow-partial'] if profile else []))
    jobs['report']=submit(['--dependency=afterany:'+jobs['evaluation'],'studies/mean_max_old89/report.slurm'])
    record=dict(jobs=jobs,submitted_indices=missing,profile=profile,retry=retry,max_concurrent_generation_gpus=2,manifest_sha256=sha(OUT/'manifest.json'))
    write(OUT/'jobs.json',record);write(OUT/'history'/('submission-%d.json'%int(time.time()*1000)),record)
    print('SUBMITTED',json.dumps(jobs),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('status','launch','profile'));p.add_argument('--dry-run',action='store_true');p.add_argument('--retry',action='store_true');a=p.parse_args()
    status() if a.action=='status' else launch(a.dry_run,a.retry,a.action=='profile')
