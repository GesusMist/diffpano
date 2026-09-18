"""Resolution-only follow-up; the completed 80-cell study stays immutable."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from diffpano.bridge_factorial import make_config as baseline_config, saved_cameras as baseline_cameras, common_settings, digest, angular_geometry
from scripts.bridge_factorial_common import read, sha, write_new, source_hashes as baseline_source_hashes, require_validation as require_baseline

ROOT=Path('outputs/bridge-erp4k-ruins/20260918')
BASE=Path('outputs/bridge-factorial-ruins/20260918')
ERP_SIZE=(2048,4096)
SELECTED={'flux':('0001','0101','1001','1011','1101','1111'),
          'pixeldit':('1001','1011','1111','0001'),
          'sana':('0001','0011','1001','1011','1111'),
          'sd2':('1001','1011','1111'),
          'sd35':('1000','1001','1011','1010')}


def cell_name(code):
    if len(code)!=4 or set(code)-{'0','1'}:raise ValueError('Expected four ABCD bits')
    return ''.join(k+v for k,v in zip('ABCD',code))


def config_diff(a,b,prefix=''):
    if isinstance(a,dict) and isinstance(b,dict):
        if set(a)!=set(b):raise AssertionError('Config keys differ at '+prefix)
        return [r for k in a for r in config_diff(a[k],b[k],prefix+'.'+k if prefix else k)]
    return [] if a==b else [dict(path=prefix,before=a,after=b)]


def make_config(name,cell):
    old=baseline_config(name,cell)
    new=replace(old,erp=replace(old.erp,height=ERP_SIZE[0],width=ERP_SIZE[1]))
    new.validate()
    changes=config_diff(old.to_dict(),new.to_dict())
    if {r['path'] for r in changes}!={'erp.height','erp.width'}:raise AssertionError('Only ERP dimensions may change')
    return new


def saved_cameras(view,erp_size):
    if tuple(erp_size)!=ERP_SIZE:raise AssertionError('Follow-up ERP must be 2048x4096')
    # The historical digest identifies cameras; its verification raster is not a frustum definition.
    if (view.height,view.width) not in ((512,512),(1024,1024)):raise AssertionError('Local raster changed')
    cams=baseline_cameras(view,(view.height,2*view.width))
    if digest(angular_geometry(cams))!=read(BASE/'manifest.json')['camera_geometry_sha256']:raise AssertionError('Angular cover changed')
    return cams


def source_hashes():
    result=baseline_source_hashes()
    for p in sorted(Path('studies/bridge_erp4k').rglob('*')):
        if p.suffix in ('.py','.slurm'):result[str(p)]=sha(p)
    return result


def make_manifest():
    old=require_baseline();rows=[];models={}
    for name,codes in SELECTED.items():
        models[name]=deepcopy(old['models'][name]);c=make_config(name,cell_name(codes[0]))
        models[name].update(common_settings=common_settings(c),common_settings_sha256=digest(common_settings(c)),clean_ERP_resolution=list(ERP_SIZE),requested_cells=len(codes))
        cams=saved_cameras(c.view,ERP_SIZE)
        if digest(angular_geometry(cams))!=old['camera_geometry_sha256']:raise AssertionError('Camera mismatch')
        for code in codes:
            cell=cell_name(code);row=deepcopy(next(r for r in old['cells'] if (r['backend'],r['cell'])==(name,cell)))
            cfg=make_config(name,cell);baseline=read(Path(row['output'])/'metadata.json')
            if baseline['config']!=row['resolved_config']:raise AssertionError('Baseline config mismatch')
            row.update(code=code,pilot=False,baseline_output=row['output'],baseline_config_sha256=row['resolved_config_sha256'],
                output=str(ROOT/'cells'/name/cell),resolved_config=cfg.to_dict(),resolved_config_sha256=digest(cfg.to_dict()),
                scientific_setting_diff=config_diff(baseline['config'],cfg.to_dict()))
            rows.append(row)
    if len(rows)!=22 or len({(r['backend'],r['cell']) for r in rows})!=22:raise AssertionError('Expected exactly 22 selected cells')
    value=dict(study='Ruins selected-cell ERP 2048x4096 follow-up',seed=0,expected_cells=22,expected_scientific_generations=22,
        prompt=old['prompt'],camera_geometry_sha256=old['camera_geometry_sha256'],angular_cameras=old['angular_cameras'],
        geometry_source=old['geometry_source'],models=models,cells=rows,baseline_root=str(BASE),baseline_manifest_sha256=sha(BASE/'manifest.json'),
        erp_size=list(ERP_SIZE),local_raster_policy='Unchanged from each corresponding completed baseline cell',
        initialization_policy='Native grid/FOV-derived noise source and all initial local states unchanged; clean RGB ERP size does not resize noise ERP',
        analysis='Selected-cell paired ERP-resolution comparison; not a complete factorial; no new main effects or interactions estimated',
        references='No new original SphereDiff generations; existing references already have 2048x4096 ERP')
    write_new(ROOT/'manifest.json',value)
    return value


def require_validation():
    require_baseline();v=read(ROOT/'validation.json')
    if not v['passed'] or v['source_hashes']!=source_hashes() or v['manifest_sha256']!=sha(ROOT/'manifest.json'):raise AssertionError('Follow-up validation/source/manifest mismatch')
    return read(ROOT/'manifest.json')


def cell_record(name,cell,allow_preflight=False):
    m=require_validation();rows=[r for r in m['cells'] if (r['backend'],r['cell'])==(name,cell)]
    c=make_config(name,cell)
    if rows:r=rows[0]
    elif allow_preflight and name in SELECTED and cell=='A0B0C0D0':
        r=dict(backend=name,cell=cell,A=0,B=0,C=0,D=0,resolved_config=c.to_dict())
    else:raise ValueError('Cell not requested: '+name+'/'+cell)
    if r['resolved_config']!=c.to_dict():raise AssertionError('Resolved config changed')
    return m,r,c
