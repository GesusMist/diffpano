"""Exact factor-aware inventory, immutable historical evidence, atomic metadata."""
import hashlib,itertools,json,os,subprocess
from pathlib import Path
ROOT=Path('/home/shig/diffpano')
OUT=ROOT/'outputs/10.8gradient-refinement-flux/seed0-v1'
PROMPTS=('air_balloons','underwater','ruins','firework')
REFERENCES=('screened','global_mean','single_pixel','coarse_color')
GRADIENTS=('poisson_select','poisson_max')
FRACTIONS=(0.,.1)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    tmp.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(str(tmp),str(path))
def source_hashes():
    paths=[]
    for directory in ('diffpano','scripts','studies/gradient_refinement','studies/gradient_blending','studies/all_prompts','studies/tt_cea','studies/gwtf_noise','studies/gwtf_erp4k'):
        paths+=sorted((ROOT/directory).rglob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def rows():
    result=[]
    for i,(p,r,g,f) in enumerate(itertools.product(PROMPTS,REFERENCES,GRADIENTS,FRACTIONS)):
        key='flux/seed0/'+p+'/'+r+'/'+g+'/refine'+('010' if f else '000')
        result.append(dict(index=i,key=key,backend='flux',seed=0,prompt=p,reference_mode=r,gradient_mode=g,last_fraction=f,
                           expected_reuse=r=='screened' and g=='poisson_select' and f==0))
    return result
def folder(row):return OUT/'cases'/row['key']
def preserved():
    for path,value in read(OUT/'history/artifacts.json').items():
        assert sha(path)==value,'Historical artifact modified: '+path

def case_complete(row,verify=True):
    p=folder(row)
    if not (p/'status.json').exists():return False
    status=read(p/'status.json')
    if status['state'] not in ('complete','reused'):return False
    metadata=read(p/'metadata.json')
    if metadata.get('semantic_sha256')!=row.get('semantic_sha256'):return False
    if not metadata.get('reused',False) and digest(metadata.get('source_hashes',{}))!=row.get('implementation_sha256'):return False
    if verify:
        if sha(p/'metadata.json')!=status['metadata_sha256']:raise AssertionError('Metadata hash mismatch '+row['key'])
        for name,value in metadata['artifacts'].items():
            if sha(p/name)!=value:raise AssertionError('Artifact hash mismatch '+row['key']+'/'+name)
    return True
