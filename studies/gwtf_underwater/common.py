from dataclasses import replace
from pathlib import Path
from studies.gwtf_erp4k import common as baseline
from studies.gwtf_erp4k.common import BACKENDS, ERP_SIZE, differences, read, write, sha

ROOT=Path('outputs/gwtf-underwater/20260923')
PROMPT=Path('prompts/underwater.txt')
METHODS=('shared','gwtf')

def folder(name,method):
    return ROOT/name if method=='gwtf' else ROOT/name/method

def geometry(name):
    c,stub,cams,_,evidence=baseline.geometry(name)
    old=read(baseline.ROOT/name/'metadata.json')
    assert c.to_dict()==old['config']
    # The frozen factorial validator deliberately allows ruins.txt only.
    # Validate that baseline unchanged, then enforce this study's one-field delta.
    c.validate()
    c=replace(c,prompt=replace(c.prompt,path=str(PROMPT)))
    assert {x['path'] for x in differences(old['config'],c.to_dict())}=={'prompt.path'}
    return c,stub,cams,old,evidence

def expected_hash(name,method):
    stat=read(baseline.BASE/'statistical-preflight'/(name+'.json'))
    return stat['exact_seed0']['nearest' if method=='shared' else method]['record']['initial_local_sha256']

def source_hashes():
    result=baseline.source_hashes();result[str(PROMPT)]=sha(PROMPT)
    for p in sorted(Path('studies/gwtf_underwater').rglob('*')):
        if p.is_file() and p.suffix in ('.py','.slurm','.md'):result[str(p)]=sha(p)
    return result

def manifest():
    baseline.require_validation();models={};preserved={}
    for name in BACKENDS:
        c,_,cams,old,_=geometry(name)
        for f in ('metadata.json','gwtf-final.png','gpu-preflight.json'):
            p=baseline.ROOT/name/f;preserved[str(p)]=sha(p)
        models[name]=dict(config=c.to_dict(),expected_initial_sha256={m:expected_hash(name,m) for m in METHODS},
            camera_geometry_sha256=old['camera_geometry_sha256'],camera_count=len(cams),FOV_x=80,FOV_y=80,
            source_noise_grid=old['noise_grid'],local_native_resolution=old['local_native_resolution'],
            local_RGB_resolution=old['local_RGB_resolution'])
    data=dict(study='Underwater nearest shared ERP versus GWTFlow',methods=list(METHODS),models=models,
        ERP_resolution=list(ERP_SIZE),image_width=4096,image_height=2048,seed=0,
        prompt_path=str(PROMPT),prompt_sha256=sha(PROMPT),prompt_lines=PROMPT.read_text().splitlines(),
        preserved=preserved,design='Ten fresh processes; initializer is the only within-model treatment; B0C0D1 fixed.')
    write(ROOT/'manifest.json',data);return data

def require_validation(statistical=True):
    baseline.require_validation()
    v=read(ROOT/'validation.json')
    assert v['passed'] and v['source_hashes']==source_hashes()
    assert v['manifest_sha256']==sha(ROOT/'manifest.json')
    m=read(ROOT/'manifest.json')
    for p,h in m['preserved'].items():assert sha(p)==h,p
    return m
