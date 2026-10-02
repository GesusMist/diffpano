"""Validate result pairs and print progress/provenance; no metric artifacts."""
import argparse
from collections import Counter,defaultdict
from .common import *

def attempts():
    # Explicit IDs expand pending array tasks that broad user accounting omits.
    ids=set()
    for p in (REPO/'logs').glob('campatch-launch.*.out'):
        for line in p.read_text().splitlines():
            if line.startswith('ARRAY_SUBMITTED '):ids.add(json.loads(line.split(' ',1)[1])['job'])
    result=defaultdict(list)
    if ids:
        text=shell_command(['sacct','--array','-j',','.join(sorted(ids)),'-X','-n','-P','--format=JobID%40,State,ExitCode,ElapsedRaw,AllocTRES%100'])
        for line in text.splitlines():
            v=line.split('|');job=v[0]
            if '_' in job and job.rsplit('_',1)[1].isdigit():
                result[int(job.rsplit('_',1)[1])].append(dict(job=job,state=v[1],exit_code=v[2],elapsed_seconds=int(v[3] or 0),allocation=v[4]))
    return result

def report(require_complete=False):
    validation=validation_gate();records=attempts();totals=defaultdict(Counter);initial=defaultdict(set);failed=[];pending=[]
    baseline=validation['old89_baselines']
    for old in baseline:assert image_valid(old['path']) and sha(old['path'])==old['sha256']
    print('strategy | N | ERP complete | CEA complete | total',flush=True)
    print('old89 | 89 | 12 | 12 | 24',flush=True)
    for row in rows():
        if complete(row,verbose=True):
            totals[row['strategy']][row['projection']]+=1
            r=read(Path(row['output'])/'config.json')
            initial[row['strategy'],row['backend']].add(r['initialization']['initial_local_sha256'])
            assert r['source']['fingerprint']==validation['fingerprint']
            pf=preflight_gate(row['backend'])
            assert r['initialization']['initial_local_sha256']==pf['initialization_hashes'][row['strategy']]
        elif row['strategy'] in validation['blocked_strategies']:
            emit('BLOCKED_ROW',row=row,reason='Fixed geometry failed coverage gate')
        else:
            a=records.get(row['index'],[])
            active=[v for v in a if v['state'] in ('RUNNING','PENDING','COMPLETING','CONFIGURING','REQUEUED')]
            errors=[v for v in a if v['state'].startswith(('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'))]
            if errors and not active:
                failed.append(row)
                for e in errors:
                    path=REPO/'logs'/('campatch-run.'+e['job']+'.err')
                    emit('FAILED_RUN',row=row,attempt=e,error=path.read_text()[-3000:] if path.exists() else 'stderr absent')
            else:pending.append(row)
            emit('MISSING_RESULT',row=row,attempts=a)
    for strategy in STRATEGIES:
        t=totals[strategy];print(f"{strategy_name(strategy,89)} | 89 | {t['erp']} | {t['cea']} | {sum(t.values())}",flush=True)
    assert all(len(v)==1 for v in initial.values()),'Initialization differs across prompt/projection'
    for name,c in validation['coverage'].items():emit('GEOMETRY_SUMMARY',strategy=name,**c)
    total=sum(sum(v.values()) for v in totals.values())
    emit('TOTAL',new_complete=total,required_nonblocked=24*len(validation['allowed_strategies']),old89_complete=24,
        failed=len(failed),pending=len(pending),blocked_strategies=validation['blocked_strategies'],
        allocated_gpu_seconds=sum(v['elapsed_seconds'] for a in records.values() for v in a),output_root=str(ROOT),old89_root=str(BASE))
    if require_complete:
        assert total==24*len(validation['allowed_strategies']) and not failed and not pending
        for strategy in validation['allowed_strategies']:assert totals[strategy]['erp']==totals[strategy]['cea']==12
        extras=[str(p) for p in ROOT.rglob('*') if p.is_file() and p.name not in ('README.md','final.png','config.json')]
        assert not extras,extras
        assert len(list(ROOT.rglob('README.md')))==3
        print('CAMERA_PATCHING_GENERATION_COMPLETE; visual review remains a separate read-only inspection.',flush=True)
    return total

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');a=p.parse_args();report(a.require_complete)
