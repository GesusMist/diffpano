"""Isolated control identities; the running mean-max source tree stays unchanged."""
import os
from pathlib import Path
from studies.mean_max_old89 import common as base
from studies.mean_max_old89.common import ROOT, PROMPTS, NEW, COUNTS, sha, digest, read, write, prompt, prompt_path, safe
OUT=ROOT/'outputs/10.14mean-max-old89-controls/seed0-v1'
MODE='poisson_select'
REFERENCES=('coarse_color','screened')


def definitions(value):
    if isinstance(value,dict):
        value=value['missing'] if 'missing' in value else value['rows']
    if not isinstance(value,list):raise ValueError('Expected a list, or an object with missing/rows list')
    pairs=[]
    for item in value:
        name=item['prompt'];reference=item['reference_mode']
        if name not in PROMPTS or reference not in REFERENCES:raise ValueError('Unknown missing control: '+str(item))
        if (name,reference) in pairs:raise ValueError('Duplicate missing control: '+str(item))
        pairs.append((name,reference))
    pairs.sort(key=lambda pair:(PROMPTS.index(pair[0]),REFERENCES.index(pair[1])))
    rows=[]
    for name,reference in pairs:
        identity=str(prompt_path(name).relative_to(ROOT))[:-4]
        rows.append(dict(index=len(rows),case_id=identity+'/'+reference+'/'+MODE+'/refine000',
            prompt=name,prompt_path=str(prompt_path(name)),prompt_identity=identity,
            subset='new4' if name in NEW else 'original11',reference_mode=reference,
            gradient_mode=MODE,backend='flux',layout='old89',camera_count=89,projection='erp',
            seed=0,steps=20,last_fraction=0.,time_travel=False,camera_rotation=False))
    return rows


def sources():
    values=base.sources()
    for path in sorted((ROOT/'studies/mean_max_controls').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.slurm','.sh','.txt'):
            values[str(path.relative_to(ROOT))]=sha(path)
    return values


def folder(row):return OUT/'cases'/row['case_id']


def complete(row,verify=True):
    path=folder(row)
    try:
        status=read(path/'status.json');meta=read(path/'metadata.json')
        if status['state']!='complete':return False
        if status['semantic_sha256']!=row['semantic_sha256'] or meta['semantic_sha256']!=row['semantic_sha256']:return False
        if meta['source_sha256']!=row['source_sha256']:return False
        required={'final.png','terminal-rgb-reference.png','final-unclamped-fp32.pt','terminal-rgb-reference-fp32.pt','config.json'}
        if set(meta['artifacts'])!=required:return False
        if meta['artifacts']['final.png']!=status['image_sha256']:return False
        if meta['counts']!=COUNTS or meta['poisson_solves']!=21 or len(meta['fusions'])!=21:return False
        for fusion in meta['fusions']:
            if fusion['mode']!=MODE or fusion['reference_mode']!=row['reference_mode'] or not fusion['converged']:return False
            if fusion.get('gradient_mix_alpha') is not None:return False
        if not meta['fusions'][-1]['terminal']:return False
        if verify:
            if sha(path/'metadata.json')!=status['metadata_sha256']:return False
            for name,value in meta['artifacts'].items():
                if sha(path/name)!=value:return False
        return True
    except (KeyError,ValueError,OSError):return False


def freeze_check():
    manifest=read(OUT/'manifest.json')
    assert manifest['source_hashes']==sources(),'Control implementation changed after freeze'
    assert sha(base.OUT/'manifest.json')==manifest['mean_max_manifest_sha256'],'Mean-max manifest changed'
    for name,info in manifest['prompts'].items():assert prompt(name)[0]==info
    assert len(manifest['rows'])==manifest['logical_cases']
    return manifest


def validation():
    manifest=freeze_check()
    gate=read(OUT/'validation/prepared.json')
    assert gate['passed'] and gate['source_sha256']==digest(manifest['source_hashes'])
    assert gate['manifest_sha256']==sha(OUT/'manifest.json')
    # Reuse the already validated real FLUX conditioning, initial noise, schedule,
    # camera cover, checkpoint, and legacy-algorithm regression evidence.
    for name in ('cpu.json','weights.json','metrics-cpu.json'):base.gate(name)
    gpu=base.gate('gpu.json')
    assert sha(base.OUT/'validation/gpu.json')==manifest['mean_max_gpu_validation_sha256']
    return gpu
