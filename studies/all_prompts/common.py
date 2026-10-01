"""Deterministic matrix, stdout provenance and atomic PNG-only output."""
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

REPO = Path('/home/shig/diffpano')
STUDY = REPO/'studies/all_prompts'
ROOT = REPO/'outputs/all-prompts-erp-cea-seed0'
BACKENDS = ('sana','flux','sd35','pixeldit')
STEPS = dict(sana=20,flux=20,sd35=40,pixeldit=50)
TT = REPO/'outputs/tt-cea/seed0-v1'
ORIGINAL = REPO/'outputs/original-spherediff/ruins-underwater-seed0-v1'
BASES = dict(ruins=REPO/'outputs/gwtf-erp4k/20260923', underwater=REPO/'outputs/gwtf-underwater/20260923')
PREFIX = 'allp21'
EXPECTED_PROMPTS = (
    'air_balloons','aurora','blossom','city','clouds_over_grass','desert','desert_sandstorm',
    'firefly','firework','forest','grand_canyon','lavender','moonlit_snowy_village','native_control',
    'ocean','rock_mountain','rock_mountain_with_star','ruins','snow_mountain','storm','underwater')

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def safe(v):
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(tuple,list,set)):return [safe(x) for x in (sorted(v) if isinstance(v,set) else v)]
    if isinstance(v,Path):return '-Infinity' if v<0 else 'Infinity' if v>0 else 'NaN'
    if isinstance(v,float) and not math.isfinite(v):return '-Infinity' if v<0 else 'Infinity' if v>0 else 'NaN'
    if hasattr(v,'detach'):return safe(v.detach().cpu().tolist())
    return v
def emit(event,**values):print(event+' '+json.dumps(safe(values),sort_keys=True),flush=True)

def prompt_record(path):
    p=Path(path);raw=p.read_bytes();lines=raw.decode('utf-8').splitlines()
    if len(lines) not in (1,5) or any(not x.strip() for x in lines):raise ValueError('Expected one or five nonempty lines: '+str(p))
    adapted=len(lines)==1
    effective=('\n'.join(lines*5)+'\n').encode('utf-8') if adapted else raw
    return dict(name=p.stem,path=str(p),filename=p.name,line_count=len(lines),adapted=adapted,
        original_sha256=hashlib.sha256(raw).hexdigest(),effective_sha256=hashlib.sha256(effective).hexdigest(),
        effective_lines=lines*5 if adapted else lines),effective

def inventory(verbose=False):
    paths=sorted((REPO/'prompts').glob('*.txt'))
    if len(paths)!=21:
        emit('PROMPT_COUNT_ERROR',count=len(paths),filenames=[p.name for p in paths])
        raise RuntimeError('Local intended prompt count is not 21; no production submission')
    if tuple(p.stem for p in paths)!=EXPECTED_PROMPTS:raise RuntimeError('Prompt filenames changed; review inventory before resuming')
    records=[prompt_record(p)[0] for p in paths]
    if verbose:
        for r in records:emit('PROMPT',**r)
    return records

def rows():
    prompts=inventory();result=[]
    for n in BACKENDS:
        for projection in ('erp','cea'):
            for p in prompts:
                result.append(dict(index=len(result),method='diffpano',backend=n,projection=projection,prompt=p['name'],
                    output=str(ROOT/'diffpano'/n/projection/p['name']/'final.png'),steps=STEPS[n]))
    for n in ('sana','flux'):
        for p in prompts:
            result.append(dict(index=len(result),method='spherediff',backend=n,projection='spherical',prompt=p['name'],
                output=str(ROOT/'spherediff'/n/p['name']/'final.png'),steps=20 if n=='sana' else 28))
    assert len(result)==210
    assert sum(r['projection']=='erp' for r in result)==84
    assert sum(r['projection']=='cea' for r in result)==84
    assert sum(r['method']=='spherediff' for r in result)==42
    assert not any(r['backend']=='sd2' for r in result)
    return result

def fingerprint():
    paths=[]
    for root in ('studies/all_prompts','studies/tt_cea','studies/original_spherediff',
                 'studies/gwtf_noise','studies/gwtf_erp4k','diffpano','scripts','configs/experiments/erp_later'):
        paths += [p for p in (REPO/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and p.suffix in ('.py','.yaml','.slurm','.sh','.md','.txt','.json')]
    return digest({str(p.relative_to(REPO)):sha(p) for p in sorted(paths)})
def inventory_hash():return digest(inventory())

def temporary_root():
    p=Path(os.environ.get('TMPDIR','/scratch/user/shig/diffpano/tmp'))/'diffpano-all-prompts'
    p.mkdir(parents=True,exist_ok=True);return p
@contextlib.contextmanager
def lock(key):
    path=temporary_root()/'locks'/key;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield
@contextlib.contextmanager
def effective_prompt(name):
    r,data=prompt_record(REPO/'prompts'/(name+'.txt'));emit('PROMPT',**r)
    if not r['adapted']:yield r,Path(r['path']);return
    with tempfile.TemporaryDirectory(prefix=name+'-',dir=str(temporary_root())) as d:
        p=Path(d)/'prompt.txt';p.write_bytes(data)
        yield r,p

def image_valid(path,verbose=False):
    from PIL import Image
    import numpy as np
    p=Path(path)
    if not p.exists():return False
    try:
        with Image.open(p) as im:
            if im.format!='PNG' or im.size!=(4096,2048) or im.mode!='RGB':raise ValueError('Expected RGB PNG 4096x2048; got '+repr((im.format,im.size,im.mode)))
            im.verify()
        with Image.open(p) as im:
            im.load()
            if not np.isfinite(np.asarray(im)).all():raise ValueError('Nonfinite image')
        return True
    except Exception as e:
        if verbose:emit('INVALID_IMAGE',path=str(p),error=repr(e))
        return False

def publish(image,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if image_valid(p):emit('SKIP_VALID',path=str(p));return
    if p.exists() or p.is_symlink():
        assert not image_valid(p,True)
        emit('REMOVE_CONFIRMED_INVALID',path=str(p));p.unlink()
    tmp=p.with_name('final.tmp.png')
    if tmp.exists() or tmp.is_symlink():tmp.unlink()
    try:
        image.save(tmp,format='PNG')
        assert image_valid(tmp,True)
        with tmp.open('rb') as f:os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        if tmp.exists():tmp.unlink()

def link_reuse(source,destination):
    source=Path(source).resolve();target=Path(destination)
    assert image_valid(source,True)
    with lock('case-'+hashlib.sha256(str(target).encode()).hexdigest()):
        if image_valid(target):return
        if target.exists() or target.is_symlink():
            assert not image_valid(target,True);emit('REMOVE_CONFIRMED_INVALID',path=str(target));target.unlink()
        target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name('final.tmp.png')
        if tmp.exists() or tmp.is_symlink():tmp.unlink()
        try:
            tmp.symlink_to(source);assert image_valid(tmp,True);os.replace(tmp,target)
        finally:
            if tmp.exists() or tmp.is_symlink():tmp.unlink()
    emit('REUSED',source=str(source),output=str(target),image_sha256=sha(target))

def shell_command(args):
    env=dict(os.environ);venv=env.pop('VIRTUAL_ENV',None);env.pop('VIRTUAL_ENV_PROMPT',None)
    if venv:env['PATH']=':'.join(x for x in env['PATH'].split(':') if x.rstrip('/')!=str(Path(venv)/'bin'))
    return subprocess.check_output(args,cwd=str(REPO),env=env,universal_newlines=True).strip()

def require_validation(job,code_hash,prompt_hash):
    assert fingerprint()==code_hash,'Sweep source changed since validation'
    assert inventory_hash()==prompt_hash,'Prompt bytes changed since validation'
    log=REPO/'logs'/('allp21-validate.'+str(job)+'.out')
    token='SWEEP_VALIDATION_PASSED '+json.dumps(dict(fingerprint=code_hash,inventory_sha256=prompt_hash),sort_keys=True)
    assert token in log.read_text(),'Successful validation token absent'
    state=shell_command(['sacct','-j',str(job),'-n','-P','--format=JobIDRaw,State,ExitCode'])
    assert any(line.split('|')[:3]==[str(job),'COMPLETED','0:0'] for line in state.splitlines()),state
