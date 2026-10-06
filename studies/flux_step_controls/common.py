import json
import os
from pathlib import Path
from datetime import datetime, timezone
from studies.all_prompts.common import (REPO, read, sha, digest, inventory, inventory_hash,
    fingerprint as historical_fingerprint, image_valid, effective_prompt, lock, shell_command, emit)
from studies.original_spherediff.common import safe, atomic, immutable

STUDY = REPO/'studies/flux_step_controls'
ROOT = REPO/'outputs/flux-step-controls-seed0'
DIFFPANO_28_PROJECTION = 'erp'
ACCOUNT = '132684267659'
HISTORICAL = '76a48b61173437c463aa8e3730a2eca73fad86acae01c48002e4836afc32bddd'

def now(): return datetime.now(timezone.utc).isoformat()
def rows():
    out=[]
    for method,steps,projection in [('spherediff',20,'spherical'),('diffpano',28,DIFFPANO_28_PROJECTION)]:
        for p in inventory():
            folder=ROOT/method/'flux'
            if method=='diffpano': folder=folder/projection
            folder=folder/('steps'+str(steps))/p['name']
            out.append(dict(index=len(out),method=method,backend='flux',steps=steps,projection=projection,
                seed=0,prompt=p['name'],prompt_sha256=p['original_sha256'],effective_prompt_sha256=p['effective_sha256'],
                output=str(folder/'final.png'),config=str(folder/'config.json')))
    assert len(out)==42 and len({r['output'] for r in out})==42
    return out

def source_hashes():
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(STUDY.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}
def protected_hashes():
    return dict(historical=historical_fingerprint(),camera_sources={str(p.relative_to(REPO)):sha(p)
        for p in sorted((REPO/'studies/camera_patching').rglob('*')) if p.is_file() and '__pycache__' not in p.parts})
def manifest(): return read(ROOT/'manifest.json')
def assert_preserved(m):
    assert protected_hashes()==m['protected_sources'],'Historical source changed'
    assert source_hashes()==m['study_sources'],'Step-control source changed after freeze'
    assert inventory_hash()==m['prompt_inventory_sha256'],'Prompt inventory changed'
    assert rows()==m['rows']

def schedule_valid(record, steps):
    t=record['timesteps'];s=record['sigmas']
    assert len(t)==steps and len(s)==steps+1
    assert all(float(t[i])>float(t[i+1]) for i in range(steps-1))
    assert all(float(s[i])>float(s[i+1]) for i in range(steps))
    assert float(s[-1])==0 and float(s[0])==1

def complete(row):
    p=Path(row['output']);c=Path(row['config'])
    if not p.exists() or not c.exists():return False
    try:
        m=read(c)
        assert m['row']==row and m['state']=='complete'
        used=next((read(p) for p in [ROOT/'manifest.json']+list((ROOT/'manifests').glob('*.json')) if sha(p)==m['manifest_sha256']),None)
        assert used is not None and row in used['rows']
        assert m['output']['sha256']==sha(p) and image_valid(p)
        assert m['prompt']['original_sha256']==row['prompt_sha256']
        assert m['prompt']['effective_sha256']==row['effective_prompt_sha256']
        schedule_valid(m['schedule'],row['steps'])
        assert m['execution']['counts']['denoiser']==89*row['steps']
        assert m['source']['study_sources']==used['study_sources']
        return True
    except (AssertionError,KeyError,ValueError,OSError):return False

def publish(row,image,config):
    target=Path(row['output']);cp=Path(row['config']);target.parent.mkdir(parents=True,exist_ok=True)
    if complete(row):return
    # Existing unmatched final artifacts require explicit inspection; never overwrite them.
    if target.exists() or cp.exists():raise FileExistsError('Unmatched final artifacts: '+str(target.parent))
    tmp=target.with_name('final.tmp.png');image.save(tmp,format='PNG')
    assert image_valid(tmp)
    config.update(state='complete',row=row,manifest_sha256=sha(ROOT/'manifest.json'),
                  output=dict(sha256=sha(tmp),width=4096,height=2048,projection='erp'))
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,target)
    atomic(cp,config)  # evaluator requires the matching pair
    assert complete(row)

def ledger(event,**values):
    p=ROOT/'submissions.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        f.write(json.dumps(dict(time=now(),event=event,**safe(values)),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

def preflight_gate(method):
    p=ROOT/'validation'/('preflight-'+method+'.json');v=read(p)
    assert v['passed'] and v['manifest_sha256']==sha(ROOT/'manifest.json')
    assert v['study_sources']==source_hashes()
    return v
