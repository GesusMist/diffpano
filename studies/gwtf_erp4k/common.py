from dataclasses import replace
from pathlib import Path
from studies.gwtf_noise import common as original
from studies.gwtf_noise.common import BACKENDS,read,write,sha

ROOT=Path('outputs/gwtf-erp4k/20260923')
BASE=original.ROOT
ERP_SIZE=(2048,4096)  # H,W; requested image width x height is 4096 x 2048.


def differences(a,b,prefix=''):
    if isinstance(a,dict) and isinstance(b,dict):
        assert a.keys()==b.keys(),prefix
        return [r for k in a for r in differences(a[k],b[k],prefix+'.'+k if prefix else k)]
    return [] if a==b else [dict(path=prefix,before=a,after=b)]


def geometry(name):
    c,stub,cameras,_,evidence=original.backend_geometry(name)
    old=read(BASE/name/'metadata.json')
    assert c.to_dict()==old['config']
    c=replace(c,erp=replace(c.erp,height=ERP_SIZE[0],width=ERP_SIZE[1]))
    c.validate()
    assert {r['path'] for r in differences(old['config'],c.to_dict())}=={'erp.height','erp.width'}
    return c,stub,cameras,old,evidence


def source_hashes():
    result=original.source_hashes()
    for p in sorted(Path('studies/gwtf_erp4k').rglob('*')):
        if p.is_file() and p.suffix in ('.py','.slurm','.md'):result[str(p)]=sha(p)
    return result


def manifest():
    original.require_validation(statistical=True);original.verify_preservation()
    models={};preserved={}
    for name in BACKENDS:
        c,stub,cams,old,evidence=geometry(name)
        for filename in ('metadata.json','gwtf-final.png','gpu-preflight.json'):
            p=BASE/name/filename;preserved[str(p)]=sha(p)
        models[name]=dict(config=c.to_dict(),setting_changes=differences(old['config'],c.to_dict()),
            baseline=str(BASE/name),initial_state_sha256=old['initialization']['initial_local_sha256'],
            source_noise_grid=old['noise_grid'],native_resolution=old['local_native_resolution'],
            local_RGB_resolution=old['local_RGB_resolution'],camera_sha256=old['camera_sha256'],
            camera_geometry_sha256=old['camera_geometry_sha256'],camera_count=len(cams),FOV_x=80,FOV_y=80,
            expected_predictions=89*c.generation.num_inference_steps)
    data=dict(study='GWTFlow clean RGB ERP resolution-only follow-up',ERP_resolution=list(ERP_SIZE),
        image_width=4096,image_height=2048,seed=0,models=models,preserved=preserved,
        baseline_validation_sha256=sha(BASE/'validation.json'),baseline_statistical_gate_sha256=sha(BASE/'preflight.json'),
        invariant='89 saved 80-degree cameras, local/native rasters, primary noise grid, exact GWTFlow states, prompt, B0C0D1 and schedules unchanged')
    write(ROOT/'manifest.json',data);return data


def require_validation(statistical=True):
    original.require_validation(statistical=True)
    v=read(ROOT/'validation.json');assert v['passed'] and v['source_hashes']==source_hashes()
    assert v['manifest_sha256']==sha(ROOT/'manifest.json')
    m=read(ROOT/'manifest.json')
    for p,h in m['preserved'].items():assert sha(p)==h,p
    return m
