"""Login-safe exact case identities and content-verified state."""
import hashlib,json,os,subprocess
from pathlib import Path

ROOT=Path('/home/shig/diffpano')
OUT=ROOT/'outputs/10.15perspective24-flux/seed0-v1'
LABELS=('01_up','02_slightly_up','03_eye_level','04_slightly_down','05_down')
COUNTS=dict(denoiser=20,encode=0,decode=1,initialize=1)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(str(tmp),str(path))

def prompt_names():
    from studies.followup24.common import PROMPTS20,NEW4
    return tuple(PROMPTS20)+tuple(NEW4)

def prompt_record(name):
    from studies.followup24.common import NEW4
    if name not in prompt_names():raise ValueError(name)
    path=ROOT/('prompts/ournew' if name in NEW4 else 'prompts')/(name+'.txt')
    raw=path.read_bytes();lines=raw.decode('utf-8').splitlines()
    if len(lines)!=5 or any(not line.strip() for line in lines):raise ValueError('Expected five unchanged nonempty lines: '+str(path))
    return dict(name=name,path=str(path),relative_path=str(path.relative_to(ROOT)),source_group='new4' if name in NEW4 else 'original20',
                original_sha256=sha(path),effective_sha256=sha(path),effective_lines=lines,line_count=5,adapted=False)

def definitions():
    rows=[]
    for name in prompt_names():
        p=prompt_record(name)
        for line_index,(label,line) in enumerate(zip(LABELS,p['effective_lines']),1):
            rows.append(dict(index=len(rows),case_id=p['relative_path'][:-4]+'/'+label,prompt=name,prompt_path=p['path'],
                prompt_relative_path=p['relative_path'],source_group=p['source_group'],artifact_kind='standalone_perspective',
                direction_label=label,line_index=line_index,line_text=line,line_sha256=hashlib.sha256(line.encode('utf-8')).hexdigest(),
                original_prompt_sha256=p['original_sha256'],effective_prompt_sha256=p['effective_sha256'],backend='flux',seed=0,steps=20,
                height=1024,width=1024,batch_size=1))
    return rows

def sources():
    from studies.ournew_comparison.common import sources as base_sources
    result=base_sources()
    for p in sorted((ROOT/'studies/perspective24').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.slurm','.sh','.txt'):
            result[str(p.relative_to(ROOT))]=sha(p)
    for name in ('common.py','__init__.py'):
        shared=ROOT/'studies/followup24'/name;result[str(shared.relative_to(ROOT))]=sha(shared)
    return result

def folder(row):return OUT/'cases'/row['case_id']

def complete(row,verify=True):
    d=folder(row)
    try:
        s=read(d/'status.json');m=read(d/'metadata.json')
        if s['state']!='complete' or s['semantic_sha256']!=row['semantic_sha256']:return False
        if m['semantic_sha256']!=row['semantic_sha256'] or m['source_sha256']!=row['source_sha256']:return False
        if m['counts']!=COUNTS or m['guided_steps']!=20 or m['native_intervals']!=20:return False
        if m['initialization']['seed']!=0 or m['line_sha256']!=row['line_sha256']:return False
        if m['initialization']['fresh_generator_per_image'] is not True:return False
        if set(m['artifacts'])!={'config.json','final.png'}:return False
        if verify:
            if sha(d/'metadata.json')!=s['metadata_sha256']:return False
            if m['artifacts']['final.png']!=s['image_sha256']:return False
            for name,value in m['artifacts'].items():
                if sha(d/name)!=value:return False
        return True
    except (OSError,KeyError,ValueError):return False

def freeze_check():
    m=read(OUT/'manifest.json')
    assert m['source_hashes']==sources(),'Standalone perspective source changed after freeze'
    assert len(m['rows'])==120 and m['prompts']=={p:prompt_record(p) for p in prompt_names()}
    assert os.readlink(ROOT/'outputs')==m['outputs_symlink']
    for path,value in m['input_artifacts'].items():assert sha(path)==value,path
    return m

def gate(name):
    m=freeze_check();g=read(OUT/'validation'/name)
    assert g['passed'] and g['source_sha256']==digest(m['source_hashes']) and g['manifest_sha256']==sha(OUT/'manifest.json'),name
    return g

def status():
    m=read(OUT/'manifest.json');counts={'complete':0,'failed':0,'running':0,'missing':0,'invalid':0};pending=[]
    for row in m['rows']:
        if complete(row):state='complete'
        else:
            p=folder(row)/'status.json';state=read(p).get('state','invalid') if p.exists() else 'missing'
            if state=='complete':state='invalid'
            pending.append(dict(index=row['index'],case_id=row['case_id'],state=state))
        counts[state]=counts.get(state,0)+1
    return dict(expected=120,counts=counts,unfinished=pending)
