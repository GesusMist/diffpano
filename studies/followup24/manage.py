"""One shared Slurm launch graph: never independent overlapping %2 generation arrays."""
import argparse
import collections
import fcntl
import json
import math
import os
import subprocess
import time
from pathlib import Path
from studies.followup24.common import ROOT,OUT_A,OUT_B
from studies.perspective24.common import read,write,sha,digest

CONTROL=OUT_B/'orchestration'


def shell(args):
    env=dict(os.environ)
    virtual=env.pop('VIRTUAL_ENV',None)
    if virtual:env['PATH']=':'.join(p for p in env.get('PATH','').split(':') if p!=str(Path(virtual)/'bin'))
    return subprocess.check_output(args,cwd=str(ROOT),env=env,universal_newlines=True).strip()

def queue():
    text=shell(['squeue','-u','shig','-h','-r','-o','%i|%j|%T|%b'])
    return [dict(zip(('job','name','state','resources'),line.split('|',3))) for line in text.splitlines() if line]

def state(suite):
    module=__import__('studies.'+suite+'.common',fromlist=['OUT'])
    path=module.OUT/'manifest.json'
    if not path.exists():return dict(prepared=False)
    manifest=read(path);counts=collections.Counter();variants={};unfinished=[]
    for row in manifest['rows']:
        p=module.folder(row)/'status.json';status=read(p) if p.exists() else {}
        if module.complete(row):value='reused' if status.get('reused') else 'new_complete'
        else:value='invalid' if status.get('state')=='complete' else status.get('state','missing')
        counts[value]+=1
        if row.get('variant'):variants.setdefault(row['variant'],collections.Counter())[value]+=1
        if value not in ('reused','new_complete'):unfinished.append(dict(index=row['index'],case_id=row['case_id'],state=value))
    return dict(prepared=True,total=len(manifest['rows']),counts=dict(counts),per_variant={k:dict(v) for k,v in variants.items()},unfinished=unfinished)

def status():
    jobs=queue();result=dict(A=state('perspective24'),B=state('layout24'),jobs=jobs)
    metric=OUT_B/'evaluation/status.json'
    result['metrics']=read(metric) if metric.exists() else None
    result['job_state_counts']=dict(collections.Counter(j['state'] for j in jobs))
    print(json.dumps(result,indent=2,sort_keys=True));return result

def check_missing(module,manifest,retry):
    missing=[]
    for row in manifest['rows']:
        if module.complete(row):continue
        p=module.folder(row)/'status.json'
        if p.exists() and not retry:raise RuntimeError('Explicit --retry required for interrupted/failed '+row['case_id'])
        missing.append(row['index'])
    return missing

def launch(dry_run=False,retry=False,perspectives_only=False):
    from studies.perspective24 import common as a
    from studies.layout24 import common as b
    CONTROL.mkdir(parents=True,exist_ok=True)
    lock=(CONTROL/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    am,bm=a.freeze_check(),b.freeze_check();profile=a.gate('profile.json');a.gate('cpu.json');a.gate('weights.json');b.gate()
    metric_gate=read(OUT_B/'validation/metrics-cpu.json')
    assert metric_gate['passed'] and metric_gate['test_source_sha256']==sha(ROOT/'studies/layout24/tests_metrics.py')
    existing=queue()
    active=[j for j in existing if j['name'].startswith('followup24-') and j['name'] not in ('followup24-eval','followup24-report','followup24-metric-test') and 'gpu' in j['resources']]
    if active:raise RuntimeError('An existing follow-up generation allocation/array is active: '+str(active[:5]))
    a_missing=check_missing(a,am,retry);b_missing=check_missing(b,bm,retry)
    approved=read(b.OUT/'generation_plan.json')
    assert all(next(r for r in approved['rows'] if r['index']==index)['state']=='generate' for index in b_missing)
    # Use measured single-image cost with a generous margin; reuse one model per group.
    seconds=max(float(profile['generation_seconds']),1.)
    setup=float(profile['shared_setup_seconds'])
    group_size=max(1,min(30,int((2*3600-2*setup-600)/(1.5*seconds))))
    a_groups=[a_missing[i:i+group_size] for i in range(0,len(a_missing),group_size)]
    # Four actual requested rows cover N89/N70/N50 RGB and N89 screened-select.
    # Already completed rows are valid profiles after content/counter verification.
    profile_keys=[(89,'rgb'),(70,'rgb'),(50,'rgb'),(89,'poisson_select')]
    b_profiles=[];profile_requirements=[]
    for key in profile_keys:
        candidates=[r for r in bm['rows'] if r['method']=='diffpano' and (r['camera_count'],r['gradient_mode'])==key]
        new_valid=[r for r in candidates if b.complete(r) and not read(b.folder(r)/'status.json').get('reused')]
        if new_valid:chosen=new_valid[0]
        else:
            absent=[r for r in candidates if r['index'] in b_missing]
            if not absent:continue
            chosen=absent[0];b_profiles.append(chosen['index'])
        profile_requirements.append(chosen['index'])
    b_bulk=[i for i in b_missing if i not in b_profiles]
    plan=dict(created=time.time(),retry=retry,perspectives_only=perspectives_only,A_manifest_sha256=sha(a.OUT/'manifest.json'),B_manifest_sha256=sha(b.OUT/'manifest.json'),
        A_groups=a_groups,B_profiles=b_profiles,B_bulk=b_bulk,B_profile_requirements=profile_requirements,
        A_profile=profile,max_concurrent_generation_gpu_jobs=2,
        dependency_policy='A array %2, then B actual-profile array %2, then B remaining array %2; no overlapping generation stages; wait for existing GPU allocations before first stage',
        scientific_profile_rows_are_counted_once=True,
        sources={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__).resolve(),ROOT/'studies/followup24/dispatch.py',ROOT/'studies/followup24/generate.slurm')})
    overview=dict(A_missing=len(a_missing),A_groups=len(a_groups),A_group_size=group_size,A_seconds_per_image=seconds,
                  B_missing=len(b_missing),B_actual_profile_slots=b_profiles,B_bulk=len(b_bulk),cap=2,perspectives_only=perspectives_only)
    print('LAUNCH_PLAN',json.dumps(overview,sort_keys=True),flush=True)
    if dry_run:return overview
    directory=CONTROL/('launch-'+str(int(time.time()*1e9)));directory.mkdir()
    plan_path=directory/'plan.json';write(plan_path,plan)
    jobs=[]
    def submit(args,stage):
        job=shell(['sbatch','--parsable']+args).split(';')[0]
        assert job.isdigit(),job
        jobs.append(dict(stage=stage,job=job,args=args));write(directory/'jobs.json',dict(jobs=jobs,plan=str(plan_path),overview=overview))
        print('SUBMITTED',stage,job,flush=True);return job
    # Serialize with any active or queued GPU allocation; never cancel unrelated work.
    other=sorted({j['job'].split('_')[0] for j in existing if 'gpu' in j['resources']})
    previous=None;first_dependency='afterany:'+':'.join(other) if other else None
    def gen(stage,entries,after,success=False):
        if not entries:return after
        dependency=('afterok:' if success else 'afterany:')+after if after else first_dependency
        args=['--job-name=followup24-'+stage,'--array=0-%d%%2'%(len(entries)-1)]
        if success:args+=['--kill-on-invalid-dep=yes']
        if stage=='A':
            minutes=int(math.ceil((1.5*seconds*max(map(len,a_groups))+2*setup+600)/900)*15)
            args+=['--time=%02d:%02d:00'%(minutes//60,minutes%60)]
        if dependency:args+=['--dependency='+dependency]
        args+=['studies/followup24/generate.slurm','--plan',str(plan_path),'--stage',stage]
        return submit(args,stage)
    previous=gen('A',a_groups,previous)
    if not perspectives_only:
        previous=gen('B-profile',b_profiles,previous)
        previous=gen('B',b_bulk,previous,success=bool(b_profiles))
    # Existing evaluator instances are included so the suite lock cannot collide.
    prior_evals=[j['job'].split('_')[0] for j in existing if j['name']=='followup24-eval']
    deps=sorted(set(([previous] if previous else [])+prior_evals))
    eval_args=['--dependency=afterany:'+':'.join(deps)] if deps else []
    evaluation=previous if perspectives_only else submit(eval_args+['studies/layout24/evaluate.slurm','--allow-partial'],'evaluation')
    submit((['--dependency=afterany:'+evaluation] if evaluation else [])+['studies/layout24/report.slurm'],'report')
    write(CONTROL/'latest-launch.json',dict(directory=str(directory),jobs=jobs,plan=str(plan_path),overview=overview))
    return overview

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('status','launch'))
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--retry',action='store_true');parser.add_argument('--perspectives-only',action='store_true');args=parser.parse_args()
    if args.action=='status':status()
    else:launch(args.dry_run,args.retry,args.perspectives_only)
