"""Exact 15-prompt, 90-case matrix; login-safe state and content identities."""
import contextlib,hashlib,json,os,subprocess,tempfile
from pathlib import Path
from studies.all_prompts.common import prompt_record,safe
ROOT=Path('/home/shig/diffpano')
OUT=ROOT/'outputs/mean-max-old89-flux90/seed0-v1'
ORIGINAL=('air_balloons','aurora','blossom','clouds_over_grass','firefly','firework','forest','rock_mountain','ruins','underwater','storm')
NEW=('bamboo_grove','lantern_market','sailboat_marina','tiled_courtyard')
PROMPTS=ORIGINAL+NEW
REFERENCES=('screened','coarse_color')
ALPHAS=(.25,.5,.75)
MODE='poisson_mean_max'
COUNTS=dict(denoiser=1780,encode=3560,decode=1869,initialize=0)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(safe(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    tmp.write_text(json.dumps(safe(value),indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(str(tmp),str(path))
def prompt_path(name):
    if name not in PROMPTS:raise ValueError('Prompt outside exact mean-max matrix: '+name)
    return ROOT/('prompts/ournew' if name in NEW else 'prompts')/(name+'.txt')
def prompt(name):
    path=prompt_path(name);record,effective=prompt_record(path)
    if name in NEW and (record['adapted'] or record['line_count']!=5):raise ValueError('Expected unchanged five-line new prompt: '+str(path))
    return record,effective
@contextlib.contextmanager
def effective_prompt(name):
    info,raw=prompt(name)
    if not info['adapted']:yield info,prompt_path(name);return
    with tempfile.TemporaryDirectory(prefix=name+'-',dir=str(OUT/'effective-prompts')) as d:
        path=Path(d)/'prompt.txt';path.write_bytes(raw);yield info,path

def definitions():
    rows=[]
    for name in PROMPTS:
        identity=str(prompt_path(name).relative_to(ROOT)).removesuffix('.txt') if hasattr(str,'removesuffix') else str(prompt_path(name).relative_to(ROOT))[:-4]
        for reference in REFERENCES:
            for alpha in ALPHAS:
                tag='alpha%03d'%round(alpha*100)
                key=identity+'/'+reference+'/'+MODE+'/'+tag+'/refine000'
                rows.append(dict(index=len(rows),case_id=key,prompt=name,prompt_path=str(prompt_path(name)),
                    prompt_identity=identity,subset='new4' if name in NEW else 'original11',reference_mode=reference,
                    gradient_mode=MODE,gradient_mix_alpha=alpha,alpha_label=tag,backend='flux',layout='old89',camera_count=89,
                    projection='erp',seed=0,steps=20,last_fraction=0.,time_travel=False,camera_rotation=False))
    return rows

def sources():
    from studies.ournew_comparison.common import sources as prior_sources
    values=prior_sources()
    for p in sorted((ROOT/'studies/mean_max_old89').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.slurm','.sh','.txt'):values[str(p.relative_to(ROOT))]=sha(p)
    return values

def folder(row):return OUT/'cases'/row['case_id']
def complete(row,verify=True):
    path=folder(row)
    if not (path/'status.json').exists():return False
    status=read(path/'status.json')
    if status['state']!='complete':return False
    try:
        meta=read(path/'metadata.json')
        if meta['semantic_sha256']!=row['semantic_sha256'] or meta['source_sha256']!=row['source_sha256']:return False
        if status['semantic_sha256']!=row['semantic_sha256']:return False
        required={'final.png','terminal-rgb-reference.png','final-unclamped-fp32.pt','terminal-rgb-reference-fp32.pt','config.json'}
        if set(meta['artifacts'])!=required:return False
        if meta['artifacts']['final.png']!=status['image_sha256']:return False
        if meta['counts']!=COUNTS or meta['poisson_solves']!=21 or len(meta['fusions'])!=21:return False
        for fusion in meta['fusions']:
            if fusion['mode']!=MODE or fusion['reference_mode']!=row['reference_mode'] or fusion['gradient_mix_alpha']!=row['gradient_mix_alpha'] or not fusion['converged']:return False
        if not meta['fusions'][-1]['terminal']:return False
        if verify:
            if sha(path/'metadata.json')!=status['metadata_sha256']:return False
            for name,value in meta['artifacts'].items():
                if sha(path/name)!=value:return False
        return True
    except (KeyError,ValueError,OSError):return False

def freeze_check():
    m=read(OUT/'manifest.json');assert m['source_hashes']==sources(),'Mean-max implementation changed after freeze'
    for name in PROMPTS:assert prompt(name)[0]==m['prompts'][name]
    assert len(m['rows'])==90
    return m

def gate(name):
    m=freeze_check();record=read(OUT/'validation'/name)
    assert record['passed'] and record['source_sha256']==digest(m['source_hashes']) and record['manifest_sha256']==sha(OUT/'manifest.json'),name
    return record

def preserved():
    startup=read(OUT/'history/startup.json')
    assert os.readlink(ROOT/'outputs')==startup['outputs_symlink']
    if (OUT/'history/protected.json').exists():
        for path,value in read(OUT/'history/protected.json').items():assert sha(path)==value,'Historical record changed: '+path
    if (OUT/'history/protected-symlinks.json').exists():
        for path,value in read(OUT/'history/protected-symlinks.json').items():assert os.readlink(path)==value,'Historical link changed: '+path
