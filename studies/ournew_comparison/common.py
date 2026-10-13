"""Exact four-prompt, seven-variant suite and immutable semantic identities."""
import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path('/home/shig/diffpano')
OUT=ROOT/'outputs/10.9ournew-flux28/seed0-v1'
PROMPTS=('bamboo_grove','lantern_market','sailboat_marina','tiled_courtyard')
VARIANTS=('spherediff_official','diffpano_old89_rgb','diffpano_uniform89_rgb',
 'diffpano_uniform89_select_coarse_color','diffpano_uniform89_select_global_mean',
 'diffpano_uniform89_select_screened','diffpano_uniform89_select_single_pixel_erp')
REFS={VARIANTS[3]:'coarse_color',VARIANTS[4]:'global_mean',VARIANTS[5]:'screened',VARIANTS[6]:'single_pixel'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(str(temp),str(path))
def sources():
    directories=('diffpano','scripts','studies/ournew_comparison','studies/gradient_blending','studies/gradient_refinement','studies/gradient_refinement_retry','studies/gradient_score_refresh','studies/all_prompts','studies/tt_cea','studies/gwtf_noise','studies/gwtf_erp4k','studies/camera_patching','studies/flux_step_controls','studies/original_spherediff','studies/panorama_metrics')
    return {str(p.relative_to(ROOT)):sha(p) for d in directories for p in sorted((ROOT/d).rglob('*')) if p.is_file() and p.suffix in ('.py','.slurm','.yaml','.sh') and '__pycache__' not in p.parts}
def prompt(name):
    from studies.all_prompts.common import prompt_record
    assert name in PROMPTS
    path=ROOT/'prompts/ournew'/(name+'.txt');record,effective=prompt_record(path)
    assert record['line_count']==5 and not record['adapted'] and effective==path.read_bytes()
    return record,path

def definitions():
    return [dict(index=i*7+j,case_id=p+'/'+v,prompt=p,variant=v,backend='flux',seed=0,steps=20,
      method='spherediff' if j==0 else 'diffpano',layout='official' if j==0 else 'old89' if j==1 else 'uniform89',
      gradient_mode=None if j==0 else 'rgb' if j<3 else 'poisson_select',reference_mode=REFS.get(v),last_fraction=0.)
      for i,p in enumerate(PROMPTS) for j,v in enumerate(VARIANTS)]
def folder(row):return OUT/'cases'/row['case_id']
def complete(row,verify=True):
    path=folder(row)
    if not (path/'status.json').exists():return False
    s=read(path/'status.json')
    if s['state']!='complete':return False
    m=read(path/'metadata.json')
    if m['semantic_sha256']!=row['semantic_sha256'] or m['source_sha256']!=row['source_sha256']:return False
    if verify:
        assert sha(path/'metadata.json')==s['metadata_sha256']
        for name,value in m['artifacts'].items():assert sha(path/name)==value
    return True

def freeze_check():
    m=read(OUT/'manifest.json');assert m['source_hashes']==sources(),'Source freeze changed'
    for p in PROMPTS:
        record,_=prompt(p);assert record==m['prompts'][p]
    return m

def gate(name):
    m=freeze_check();r=read(OUT/'validation'/name)
    assert r['passed'] and r['source_sha256']==digest(m['source_hashes']) and r['manifest_sha256']==sha(OUT/'manifest.json'),name
    return r

def preserved():
    if not (OUT/'history/protected.json').exists():return
    for path,value in read(OUT/'history/protected.json').items():assert sha(path)==value,'Historical record changed: '+path
