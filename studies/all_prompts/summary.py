"""Text-only completion and failure accounting; no report artifacts."""
import argparse
from collections import defaultdict
from studies.all_prompts.common import *

def accounting():
    raw=shell_command(['sacct','--array','-u','shig','-S','2026-09-29','-X','-n','-P',
        '--format=JobID,JobName%80,State,ExitCode,ElapsedRaw'])
    result=defaultdict(list)
    for line in raw.splitlines():
        cols=line.split('|')
        if len(cols)<5 or cols[1]!=PREFIX+'-run':continue
        job=cols[0]
        if '_' not in job or not job.rsplit('_',1)[1].isdigit():continue
        index=int(job.rsplit('_',1)[1])
        result[index].append(dict(job=job,state=cols[2],exit_code=cols[3],elapsed_seconds=cols[4]))
    return result,raw

def logged_success(job,row):
    p=REPO/'logs'/('allp21-run.'+job+'.out')
    if not p.exists():return False
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-20000));text=f.read().decode('utf-8',errors='replace')
    for line in text.splitlines():
        if line.startswith('SUCCESS_CASE '):
            record=json.loads(line[len('SUCCESS_CASE '):])
            if record['row']==row and record['image_sha256']==sha(row['output']):return True
    return False

def reused_success(row):
    path=Path(row['output'])
    if not path.is_symlink():return False
    for p in sorted((REPO/'logs').glob('allp21-validate.*.out')):
        # Audit records prove the historical worker exit and final image hash.
        text=p.read_text()
        if 'SWEEP_VALIDATION_PASSED ' not in text:continue
        for line in text.splitlines():
            if line.startswith('REUSE_APPROVED '):
                c=json.loads(line[len('REUSE_APPROVED '):])
                if c['row']==row and Path(c['source']).resolve()==path.resolve() and sha(path)==c['image_sha256']:return True
    return False

def summary(require_complete=False,full=False):
    matrix=rows();records,raw=accounting();groups=defaultdict(lambda:dict(expected=0,valid=0,failed=0,pending=0,missing=0))
    failed=[];missing=[];existing=0
    for row in matrix:
        key=(row['method'],row['backend'],row['projection']);group=groups[key];group['expected']+=1
        valid=image_valid(row['output'],True);existing+=int(valid)
        attempts=records.get(row['index'],[])
        success=valid and (reused_success(row) or any(x['state']=='COMPLETED' and x['exit_code']=='0:0' and logged_success(x['job'],row) for x in attempts))
        if success:group['valid']+=1;continue
        active=[a for a in attempts if a['state'] in ('RUNNING','PENDING','COMPLETING','CONFIGURING','REQUEUED')]
        errors=[a for a in attempts if a['state'].startswith(('FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL','PREEMPTED','BOOT_FAIL'))]
        if active:group['pending']+=1
        elif errors:
            group['failed']+=1;failed.append(dict(row=row,attempts=errors))
        else:group['missing']+=1
        missing.append(row['output'])
    print('METHOD/BACKEND/PROJECTION | EXPECTED | VALID+SUCCESS | FAILED | PENDING | MISSING',flush=True)
    for key,v in groups.items():print('/'.join(key)+' | '+' | '.join(str(v[k]) for k in ('expected','valid','failed','pending','missing')),flush=True)
    for item in failed:
        emit('FAILED_PROMPT',**item)
        for attempt in item['attempts']:
            p=REPO/'logs'/('allp21-run.'+attempt['job']+'.err')
            if p.exists():
                with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3500));error=f.read().decode('utf-8',errors='replace')
                emit('FAILURE_ERROR',job=attempt['job'],stderr_tail=error)
    total=sum(v['valid'] for v in groups.values())
    emit('TOTAL',expected=210,valid_successful=total,total_existing_valid_final_images=existing,failed=len(failed),
        pending=sum(v['pending'] for v in groups.values()),missing=len(missing),output_root=str(ROOT),sd2_new_images=0)
    if full:
        for path in missing:print('MISSING_OR_NOT_SUCCESSFUL '+path,flush=True)
    if require_complete:
        assert total==210 and existing==210,('Sweep incomplete',total,missing)
        assert sum(v['valid'] for k,v in groups.items() if k[2]=='erp')==84
        assert sum(v['valid'] for k,v in groups.items() if k[2]=='cea')==84
        assert sum(v['valid'] for k,v in groups.items() if k[0]=='spherediff')==42
        extras=[str(p) for p in ROOT.rglob('*') if (p.is_file() or p.is_symlink()) and p.name!='final.png']
        assert not extras,extras
        assert len(list(ROOT.rglob('final.png')))==210
        print('SWEEP_COMPLETE: 210 validated 4096x2048 final PNGs; successful workers; no extra scientific artifacts.',flush=True)
    return total

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');p.add_argument('--full',action='store_true')
    a=p.parse_args();summary(a.require_complete,a.full)
