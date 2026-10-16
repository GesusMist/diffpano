"""Login-safe identities for the exact 216-slot panorama matrix."""
import os
from pathlib import Path
from studies.mean_max_old89.common import sha,digest,read,write,safe
from studies.followup24.common import ROOT,PROMPTS,ORIGINAL,NEW,prompt,prompt_path
OUT=ROOT/'outputs/10.15layout24-flux216/seed0-v1'
VARIANTS=('spherediff_official_20','diffpano_old89_rgb','diffpano_old70_rgb','diffpano_old50_rgb',
 'diffpano_uniform89_rgb','diffpano_uniform70_rgb','diffpano_uniform50_rgb',
 'diffpano_old89_screened_select','diffpano_uniform89_screened_select')
LAYOUTS=('old89','old70','old50','uniform89','uniform70','uniform50')


def definitions():
    rows=[]
    for name in PROMPTS:
        identity=str(prompt_path(name).relative_to(ROOT))[:-4]
        for variant in VARIANTS:
            official=variant=='spherediff_official_20'
            layout='official' if official else variant.split('_')[1]
            n=89 if official else int(layout.replace('old','').replace('uniform',''))
            selected=variant.endswith('_screened_select')
            rows.append(dict(index=len(rows),case_id=identity+'/'+variant,prompt=name,prompt_path=str(prompt_path(name)),
                prompt_identity=identity,subset='new4' if name in NEW else 'original20',variant=variant,
                backend='flux',method='spherediff' if official else 'diffpano',layout=layout,camera_count=n,
                gradient_mode=None if official else 'poisson_select' if selected else 'rgb',
                reference_mode='screened' if selected else None,projection='erp',seed=0,steps=20,
                last_fraction=0.,time_travel=False,camera_rotation=False,artifact_kind='panorama'))
    assert len(rows)==216 and len({r['case_id'] for r in rows})==216
    return rows


def folder(row):return OUT/'cases'/row['case_id']
def expected_counts(row):
    n=row['camera_count'];return dict(denoiser=20*n,encode=40*n,decode=21*n,initialize=0)
def sources():
    from studies.ournew_comparison.common import sources as historical
    values=historical()
    for directory in ('studies/camera_count_v2','studies/layout24'):
        for p in sorted((ROOT/directory).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.slurm','.sh','.txt') and p.name not in ('evaluate.py','report.py','tests_metrics.py','metrics_cpu.slurm','evaluate.slurm','report.slurm'):
                values[str(p.relative_to(ROOT))]=sha(p)
    for name in ('studies/followup24/common.py','studies/followup24/__init__.py'):
        values[name]=sha(ROOT/name)
    return values


def complete(row,verify=True):
    path=folder(row)
    try:
        status=read(path/'status.json')
        if status['state']!='complete':return False
        if status.get('reused'):
            from studies.layout24.reuse import verify_reused
            return verify_reused(row,verify=verify)
        meta=read(path/'metadata.json')
        if meta['semantic_sha256']!=row['semantic_sha256'] or meta['source_sha256']!=row['source_sha256']:return False
        if status['semantic_sha256']!=row['semantic_sha256']:return False
        if meta['artifacts']['final.png']!=status['image_sha256']:return False
        if row['method']=='diffpano':
            if meta['counts']!=expected_counts(row) or len(meta['fusions'])!=21:return False
            solves=21 if row['gradient_mode']=='poisson_select' else 0
            if meta['poisson_solves']!=solves or not meta['fusions'][-1]['terminal']:return False
            for fusion in meta['fusions']:
                if fusion['mode']!=row['gradient_mode'] or fusion.get('gradient_mix_alpha') is not None:return False
                if solves and (not fusion['converged'] or fusion['reference_mode']!='screened'):return False
        if verify:
            if sha(path/'metadata.json')!=status['metadata_sha256']:return False
            for name,value in meta['artifacts'].items():
                if sha(path/name)!=value:return False
        return True
    except (AssertionError,KeyError,ValueError,OSError):return False


def freeze_check():
    m=read(OUT/'manifest.json')
    assert m['source_hashes']==sources(),'Task B scientific source changed after freeze'
    assert os.readlink(ROOT/'outputs')==m['outputs_symlink'],'Outputs symlink changed'
    for name in PROMPTS:assert prompt(name)[0]==m['prompts'][name]
    assert len(m['rows'])==216
    assert sha(OUT/'generation_plan.json')==m['generation_plan_sha256'],'Frozen generation plan changed'
    assert sha(OUT/'reuse_audit.json')==m['reuse_audit_sha256'],'Frozen reuse audit changed'
    return m


def gate(name='prepared.json'):
    m=freeze_check();r=read(OUT/'validation'/name)
    assert r['passed'] and r['source_sha256']==digest(m['source_hashes']) and r['manifest_sha256']==sha(OUT/'manifest.json')
    return r
