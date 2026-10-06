"""One CPU dispatcher, separate fresh Slurm workers, a global five-GPU limit."""
import argparse
import time
import uuid
from .common import *
from .report import events,layout_status,report

TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL'}
INFRA={'TIMEOUT','NODE_FAIL','PREEMPTED','BOOT_FAIL'}
def tasks():
    out=[]
    for j,n in enumerate(COUNTS):
        for k,strategy in enumerate(STRATEGIES):
            module='random_search' if strategy=='random' else 'coverage'
            args=['--camera-count',str(n)]+([] if strategy=='random' else ['--strategy',strategy])
            out.append(dict(key='geometry/'+layout_key(strategy,n),kind='geometry',strategy=strategy,n=n,
                            environment='diffpano',module=module,args=args,priority=j*100+k))
        groups={}
        for r in rows():
            if r['num_cameras']!=n:continue
            groups.setdefault(group_key(r),r)
        for k,r in enumerate(groups.values()):
            out.append(dict(key='preflight/'+group_key(r),kind='preflight',row=r,environment=
                'spherediff' if r['family']=='spherediff_camera_override' else 'diffpano',
                module='preflight',args=['--index',str(r['index'])],priority=j*100+10+k))
        for r in rows():
            if r['num_cameras']!=n:continue
            out.append(dict(key='run/'+str(r['index']),kind='run',row=r,environment=
                'spherediff' if r['family']=='spherediff_camera_override' else 'diffpano',
                module='run',args=['--index',str(r['index'])],priority=j*100+40+(0 if r['prompt']=='firework' else 1)))
    return sorted(out,key=lambda x:x['priority'])

_COMPLETE_CACHE={}
def done(row):
    p=folder(row);paths=[p/'final.png',p/'config.json',ROOT/'status'/(str(row['index'])+'.json')]
    signature=tuple((str(x),x.stat().st_mtime_ns,x.stat().st_size) if x.exists() else (str(x),None,None) for x in paths)
    if _COMPLETE_CACHE.get(row['index'])==signature:return True
    if complete(row):_COMPLETE_CACHE[row['index']]=signature;return True
    return False
def task_ready(task):
    kind=task['kind']
    if kind=='geometry':
        p=layout_path(task['strategy'],task['n'])
        if p.exists():return False,'done'
        progress=ROOT/'layouts'/('random_search_n'+str(task['n'])+'.json')
        return True,'layout pending'
    row=task['row'];status,reason=layout_status(row)
    if status!='ready':return False,reason
    if kind=='preflight':
        if preflight_path(row).exists():
            preflight_gate(row);return False,'done'
        return True,'preflight pending'
    if done(row):return False,'done'
    if not preflight_path(row).exists():return False,'waiting for preflight'
    preflight_gate(row)
    pilot=next(r for r in rows() if group_key(r)==group_key(row) and r['prompt']=='firework')
    if row['prompt']!='firework' and not done(pilot):return False,'waiting for first required numerical pilot'
    return True,'scientific output missing'

def accounting(job):
    output=shell_command(['sacct','-n','-X','-P','-j',job,'--format=JobIDRaw,State,ExitCode'])
    for line in output.splitlines():
        cells=line.split('|')
        if cells[0]==job:return cells[1].split()[0],cells[2]
    return 'UNKNOWN',''
def submissions():
    result={}
    for e in events():
        if e['event']=='SUBMITTED':result.setdefault(e['task'],[]).append(e)
    return result
def audit_failed(task,entry,state,exitcode):
    if any(e['event']=='TASK_EXIT' and e.get('job')==entry['job'] for e in events()):return
    tails={}
    for ext in ('out','err'):
        files=list((REPO/'logs').glob('ccv2-*.'+entry['job']+'.'+ext))
        tails[ext]=files[0].read_text()[-5000:] if files else ''
    ledger('TASK_EXIT',task=task['key'],job=entry['job'],state=state,exitcode=exitcode,log_tail=tails)

def eligible(task,history):
    ready,reason=task_ready(task)
    if not ready:return False,reason
    prior=history.get(task['key'],[])
    if not prior:return True,reason
    last=prior[-1];state,code=accounting(last['job'])
    if state not in TERMINAL:return False,'existing job '+last['job']+' '+state
    audit_failed(task,last,state,code)
    if task['kind']=='geometry' and task['strategy']=='random' and state=='COMPLETED' and code=='0:0':
        batches=sum(accounting(e['job'])[0]=='COMPLETED' for e in prior)
        return (batches<12,'resumable random search' if batches<12 else 'random search batch budget exhausted; incomplete')
    if state in INFRA and len(prior)<3:return True,'infrastructure retry, unchanged parameters'
    return False,'failed prerequisite/task '+last['job']+' '+state+' '+code

def reconcile_intents():
    records=events()
    acknowledged={e.get('token') for e in records if e['event']=='SUBMITTED' and e.get('token')}
    unresolved=[e for e in records if e['event']=='SUBMISSION_INTENT' and e.get('token') and e['token'] not in acknowledged]
    if not unresolved:return
    queue=shell_command(['squeue','-r','-u','shig','-h','-o','%i|%k'])
    account=shell_command(['sacct','-u','shig','-S','now-2days','-n','-X','-P','--format=JobIDRaw,Comment%200,State'])
    for intent in unresolved:
        matches=set()
        for line in (queue+'\n'+account).splitlines():
            cells=line.split('|')
            if len(cells)>1 and cells[1].strip()==intent['token']:matches.add(cells[0].strip())
        if len(matches)!=1:
            raise RuntimeError('Ambiguous sbatch result needs inspection; refusing a duplicate: '+repr(intent))
        job=matches.pop();assert job.isdigit()
        ledger('SUBMITTED',task=intent['task'],job=job,kind=intent['kind'],token=intent['token'],
               recovered_from_accounting=True,source_fingerprint=intent['source_fingerprint'])

def submit_task(task):
    token='ccv2:'+uuid.uuid4().hex
    cmd=['sbatch','--parsable','--comment='+token,'--job-name=ccv2-'+task['kind'],str(STUDY/'gpu.slurm'),
         task['environment'],'-m','studies.camera_count_v2.'+task['module']]+task['args']
    if task['kind']=='geometry' and task['strategy']=='random':
        cmd.insert(2,'--time=01:00:00')
    ledger('SUBMISSION_INTENT',task=task['key'],kind=task['kind'],command=cmd,token=token,source_fingerprint=fingerprint())
    # Ambiguous command failures are not retried: dispatcher exits for inspection.
    job=shell_command(cmd).split(';')[0];assert job.isdigit(),job
    ledger('SUBMITTED',task=task['key'],job=job,kind=task['kind'],token=token,source_fingerprint=fingerprint())
    emit('SUBMITTED',task=task['key'],job=job)

def dry_run():
    validation_gate()
    for row in rows():
        state,reason=layout_status(row)
        emit('PLANNED_ROW',**row,status='completed' if done(row) else state,reason=reason)
    audit=read(ROOT/'audit.json');gpu_hours=0.;disk=0
    for row in rows():
        if row['family']=='diffpano':
            refs=audit['diffpano'][row['backend']]['camera_n89_references']
            times=[r['configuration']['execution']['generation_seconds'] for r in refs]
            seconds=sum(times)/len(times)*row['num_cameras']/89
        else:
            ref=REPO/'outputs/flux-step-controls-seed0/spherediff/flux/steps20/ruins/config.json'
            if row['backend']=='flux':seconds=read(ref)['execution']['generation_seconds']*row['num_cameras']/89
            else:
                p=read(REPO/'outputs/original-spherediff/ruins-underwater-seed0-v1/manifest.json')
                r=next(x for x in p['rows'] if x['backend']=='sana' and x['prompt']=='ruins')
                m=read(r['metadata']);seconds=m.get('runtime_seconds',m.get('generation_seconds',600))*row['num_cameras']/89
        if not done(row):gpu_hours+=seconds/3600
        disk+=4096*2048*3
    emit('PLAN_TOTAL',expected=126,diffpano=108,spherediff=18,generation_gpu_hours_estimate=gpu_hours,
         estimate_scope='historical measured runtime scaled by N; preflight/search and fixed overhead extra',
         uncompressed_final_image_gib_upper_estimate=disk/2**30,max_global_gpu_jobs=5,
         random_seeds={str(n):read(layout_path('random',n))['accepted_seed'] if layout_path('random',n).exists() else None for n in COUNTS},
         sphere_flux_steps=20,scratch=str(ROOT.resolve()))

def dispatch(hours=23.5):
    assert os.environ.get('SLURM_JOB_ID'),'Dispatcher uses CPU Slurm allocation'
    validation_gate();deadline=time.monotonic()+hours*3600
    with lock('dispatcher'):
        reconcile_intents()
        while time.monotonic()<deadline:
            validation_gate()
            queue=shell_command(['squeue','-r','-u','shig','-p','gpu','-h','-o','%i|%j|%T'])
            active=queue.splitlines() if queue else []
            slots=max(0,MAX_GPUS-len(active))
            history=submissions();states=[];submitted=0
            for task in tasks():
                ok,reason=eligible(task,history);states.append(dict(task=task['key'],ready=ok,reason=reason))
                if ok and submitted<slots:
                    submit_task(task);submitted+=1
            atomic(ROOT/'dispatcher-status.json',dict(time=now(),job=os.environ['SLURM_JOB_ID'],
                global_gpu_jobs=len(active),submitted_now=submitted,tasks=states))
            if not active and not submitted and not any(x['ready'] for x in states):
                report();ledger('DISPATCH_FINISHED',reason='all runnable tasks finished; remaining failures/blocks explicit')
                return
            time.sleep(30)
        report();ledger('DISPATCH_INCOMPLETE',reason='bounded dispatcher allocation; resume launch --submit')

def launch():
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--dry-run',action='store_true');m.add_argument('--submit',action='store_true')
    m.add_argument('--status',action='store_true');m.add_argument('--dispatch',action='store_true')
    a=p.parse_args()
    if a.status:report();return
    if a.dispatch:dispatch();return
    dry_run()
    if a.submit:
        with lock('launch'):
            q=shell_command(['squeue','-u','shig','-h','-o','%i|%j|%T'])
            if '|ccv2-dispatch|' in q:emit('DISPATCH_ALREADY_ACTIVE',queue=q);return
            cmd=['sbatch','--parsable',str(STUDY/'dispatch.slurm')]
            ledger('DISPATCH_INTENT',command=cmd)
            job=shell_command(cmd).split(';')[0]
            ledger('DISPATCH_SUBMITTED',job=job)
            emit('DISPATCH_SUBMITTED',job=job)
if __name__=='__main__':launch()
