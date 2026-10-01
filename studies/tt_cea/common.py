import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import yaml

REPO=Path('/home/shig/diffpano')
ROOT=REPO/'outputs/tt-cea/seed0-v1'
CONFIG=REPO/'studies/tt_cea/experiment.yaml'
BACKENDS=('sd2','sana','flux','sd35','pixeldit')
PROMPTS=('ruins','underwater')
STEPS=dict(sd2=30,sana=20,flux=20,sd35=40,pixeldit=50)
BASES={'ruins':REPO/'outputs/gwtf-erp4k/20260923','underwater':REPO/'outputs/gwtf-underwater/20260923'}
CASES={
 'R0':dict(projection='erp',cameras='old89',time_travel='off',budget='N'),
 'T1':dict(projection='erp',cameras='old89',time_travel='original_noise_one_replay',budget='N+K'),
 'TB':dict(projection='erp',cameras='old89',time_travel='off',budget='N+K'),
 'C0':dict(projection='cea',cameras='old89',time_travel='off',budget='N'),
 'P0':dict(projection='erp',cameras='ea89',time_travel='off',budget='N'),
 'C1':dict(projection='cea',cameras='ea89',time_travel='off',budget='N'),
}

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(8*1024*1024),b''):h.update(v)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def historical_scheduler_json(value):
    """Serialize SANA's intentional -inf scheduler bound without nonfinite diagnostics."""
    import math
    if isinstance(value,dict):
        return {k:('-Infinity' if k=='lambda_min_clipped' and isinstance(v,float) and v==-math.inf else historical_scheduler_json(v)) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [historical_scheduler_json(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite value outside historical scheduler sentinel')
    return value

def atomic(p,v,*,allow_nan=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('x') as f:json.dump(v,f,indent=2,allow_nan=allow_nan);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def immutable(p,v,*,allow_nan=False):
    p=Path(p)
    if p.exists():
        if read(p)!=v:raise RuntimeError('Immutable artifact mismatch: '+str(p))
    else:atomic(p,v,allow_nan=allow_nan)
def sample_dir(backend,prompt,case):return ROOT/'samples'/prompt/backend/case

def load_settings(path=CONFIG):
    settings=yaml.safe_load(Path(path).read_text())
    canonical=yaml.safe_load(CONFIG.read_text())
    if settings!=canonical:raise ValueError('Only the frozen study configuration is authorized')
    assert settings['seed']==0 and settings['backends']==list(BACKENDS) and settings['prompts']==list(PROMPTS)
    assert settings['canvas']==dict(height=2048,width=4096,geometry_dtype='float32',fusion_dtype='float32')
    assert settings['execution']['max_concurrent_gpu_jobs']==5
    assert settings['time_travel']['repeats_per_eligible_interval']==1
    return settings

def base_config(name,prompt):
    from studies.gwtf_erp4k.common import geometry
    c,stub,cams,_,_=geometry(name)  # historical ruins validator remains unchanged
    if prompt not in PROMPTS:raise ValueError(prompt)
    if prompt=='underwater':c=replace(c,prompt=replace(c.prompt,path='prompts/underwater.txt'))
    m=read(BASES[prompt]/name/'metadata.json')
    assert c.to_dict()==m['config'],(name,prompt,'baseline configuration differs')
    assert c.generation.num_inference_steps==STEPS[name]
    return c,stub,cams,m

def resolved(name,prompt,case):
    from studies.tt_cea.schedule import eligible_indices
    c,stub,cams,m=base_config(name,prompt)
    n=STEPS[name];k=len(eligible_indices(n))
    if case=='TB':c=replace(c,generation=replace(c.generation,num_inference_steps=n+k))
    row=dict(CASES[case],case=case,backend=name,prompt=prompt,seed=0,batch_size=1,
        canvas_height=2048,canvas_width=4096,original_steps=n,replay_intervals=k if case=='T1' else 0,
        ordinary_steps=n+k if case=='TB' else n,downhill_passes=n+k if case in ('T1','TB') else n,
        camera_count=89,FOV_x=80.,FOV_y=80.,noise_projection='erp',noise_rule=m['initialization']['config'],
        generation_configuration=c.to_dict())
    row['expected_guided_predictions']=89*row['downhill_passes']
    row['allowed_differences']={
      'R0':[], 'T1':['replay of eligible original intervals with original local noise'],
      'TB':['generation.num_inference_steps','prepared_schedule'],
      'C0':['active clean RGB projection','CEA pole reconstruction','one terminal CEA-to-ERP export'],
      'P0':['camera poses/order/IDs','world-direction prompt binding','transported local initial noise'],
      'C1':['camera poses/order/IDs','world-direction prompt binding','transported local initial noise',
            'active clean RGB projection','CEA pole reconstruction','one terminal CEA-to-ERP export']}[case]
    return c,stub,cams,m,row

def study_hashes():
    return {str(p.relative_to(REPO)):sha(p) for p in sorted((REPO/'studies/tt_cea').rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.yaml','.slurm','.md','.txt','.json')}

def verify_preservation():
    saved=read(ROOT/'preservation.json')
    for p,h in saved['hashes'].items():
        if sha(REPO/p)!=h:raise AssertionError('Historical bytes changed: '+p)
    return True

def require_gate(backend=None,case=None):
    from studies.tt_cea.metadata import compatible_source_hashes
    v=read(ROOT/'validation.json')
    assert v['passed'] and v['source_hashes']==study_hashes(),'New study sources changed after validation'
    assert v['manifest_sha256']==sha(ROOT/'manifest.json')
    verify_preservation()
    if backend:
        p=read(ROOT/'preflight'/(backend+'.json'))
        assert p['interval_passed'] and p['source_hashes'] in compatible_source_hashes(v)
        if case in ('C0','P0','C1'):
            g=read(ROOT/'geometry/coverage.json')
            key=CASES[case]['cameras'];assert g['source_hashes'] in compatible_source_hashes(v) and g['covers'][key]['passed']
            operator=read(ROOT/'geometry/operator_tests.json');assert operator['passed']
            if case in ('P0','C1'):assert p['ea_noise_passed'],p.get('ea_noise_error')
            if case in ('C0','C1'):assert p['cea_passed_by_cover'][key],p.get('cea_errors')
    return read(ROOT/'manifest.json')

def parser(description):
    p=argparse.ArgumentParser(description=description);p.add_argument('--config',default=str(CONFIG));return p
