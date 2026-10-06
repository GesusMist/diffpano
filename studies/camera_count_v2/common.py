"""Authorized matrix, immutable provenance, and transactional result checks."""
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from studies.all_prompts.common import REPO, read, sha, digest, shell_command, effective_prompt, prompt_record
from studies.original_spherediff.common import safe, atomic, immutable

STUDY = REPO/'studies/camera_count_v2'
ROOT = REPO/'outputs/camera-count-erp-seed0-v2'
PROMPTS = ('firework','underwater','ruins')
COUNTS = (70,50,30)
BACKENDS = ('flux','sana','pixeldit','sd35')
STRATEGIES = ('old','fibonacci','random')
STEPS = dict(flux=20,sana=20,pixeldit=50,sd35=40)
ACCOUNT = '132684267659'
MAX_GPUS = 5
VERSION = 'camera-count-erp-seed0-v2'
def now(): return datetime.now(timezone.utc).isoformat()
def emit(event, **kw): print(event+' '+json.dumps(safe(kw),sort_keys=True),flush=True)
def rows():
    result = []
    # Count-major progression; SphereDiff has old/rings rows ONLY.
    for n in COUNTS:
        for family,backends,strategies in (
            ('diffpano',BACKENDS,STRATEGIES),
            ('spherediff_camera_override',('sana','flux'),('old',))):
            for strategy in strategies:
                for backend in backends:
                    for prompt in PROMPTS:
                        result.append(dict(index=len(result),family=family,backend=backend,strategy=strategy,
                            num_cameras=n,prompt=prompt,steps=STEPS[backend],seed=0,projection='erp'))
    assert len(result)==126 and sum(r['family']=='diffpano' for r in result)==108
    assert sum(r['family']=='spherediff_camera_override' for r in result)==18
    assert all(r['strategy']=='old' for r in result if r['family']=='spherediff_camera_override')
    assert len({tuple(r.values()) for r in result})==126
    return result
def validate_row(row):
    assert row in rows(), 'Unauthorized scientific row'
    if row['family']=='spherediff_camera_override' and row['strategy']!='old':
        raise ValueError('SphereDiff accepts old/rings only')
def layout_key(strategy,n): return f'{strategy}_n{n}'
def layout_path(strategy,n): return ROOT/'layouts'/(layout_key(strategy,n)+'.json')
def load_layout(row):
    validate_row(row)
    v=read(layout_path(row['strategy'],row['num_cameras']))
    assert v['strategy']==row['strategy'] and v['num_cameras']==row['num_cameras']
    assert v['version']==VERSION and v['coverage']['diffpano']['passed']
    if row['family']=='spherediff_camera_override':
        assert v['coverage'].get('spherediff',{}).get('passed'), 'SphereDiff coverage gate not passed'
    return v
def folder(row):
    validate_row(row)
    return ROOT/row['family']/layout_key(row['strategy'],row['num_cameras'])/row['backend']/row['prompt']
def group_key(row): return f"{row['family']}-{row['strategy']}-n{row['num_cameras']}-{row['backend']}"
def source_hashes():
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(STUDY.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}
def fingerprint(): return digest(source_hashes())
def geometry_identity():
    return digest({n:sha(STUDY/n) for n in ('cameras.py','coverage.py','spherediff_adapter.py')})
def protected_hashes():
    from studies.flux_step_controls.common import protected_hashes as old
    from studies.original_spherediff.common import official_hashes
    return dict(historical=old(),official=official_hashes())
def assert_preserved():
    a=read(ROOT/'audit.json')
    assert a['protected_sources']==protected_hashes()
    assert (REPO/'outputs').is_symlink() and str((REPO/'outputs').resolve())==a['scratch_destination']
def validation_gate():
    v=read(ROOT/'validation.json')
    assert v['passed'] and v['source_hashes']==source_hashes(), 'Current validation gate required'
    assert_preserved()
    return v
def preflight_path(row): return ROOT/'preflight'/(group_key(row)+'.json')
def preflight_gate(row):
    v=read(preflight_path(row));layout=load_layout(row)
    assert v['passed'] and v['source_hashes']==source_hashes()
    assert v['layout_hash']==layout['angular_geometry_sha256']
    assert v['geometry_identity']==geometry_identity()
    return v
@contextlib.contextmanager
def lock(key):
    p=ROOT/'locks'/key;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
def ledger(event, **kw):
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'execution.jsonl').open('a') as f:
        f.write(json.dumps(safe(dict(time=now(),event=event,**kw)),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def expected_counts(row, steps=None, terminal=True):
    n=row['num_cameras'];s=row['steps'] if steps is None else steps
    pixel=row['backend']=='pixeldit';sphere=row['family']=='spherediff_camera_override'
    return dict(denoiser=n*s,encode=0 if pixel or sphere else 2*n*s,
                decode=(n if terminal else 0) if sphere else (0 if pixel else n*s+(n if terminal else 0)),initialize=0)
def record_valid(record,row):
    validate_row(row);layout=load_layout(row)
    assert record['version']==VERSION and record['row']==row
    assert record['camera']['angular_geometry_sha256']==layout['angular_geometry_sha256']
    assert record['camera']['layout_sha256']==sha(layout_path(row['strategy'],row['num_cameras']))
    assert record['camera']['geometry_identity']==geometry_identity()
    assert record['source_hashes']==source_hashes()
    assert record['prompt']['original_sha256']==sha(REPO/'prompts'/(row['prompt']+'.txt'))
    assert record['execution']['counts']==expected_counts(row)
    assert record['trajectory']==dict(time_travel=False,replay_count=0,backward_count=0,original_noise_reinjection_count=0)
    assert record['generation']['steps']==row['steps'] and record['generation']['seed']==0
    assert record['output']['width']==4096 and record['output']['height']==2048
    assert record['coverage']['passed']
    pf=preflight_gate(row)
    assert record['initialization']['initial_state_sha256']==pf['record']['initialization']['initial_state_sha256']
    if row['family']=='diffpano':
        assert record['initialization']['method']=='gwtflow'
        assert record['aggregation']['mode']=='weighted_average'
        assert record['aggregation']['weight_mode']=='spherediff_center'
        assert record['aggregation']['temperature']==.1
        assert record['transition']=='preserve_current_state'
    else:
        assert record['initialization']['method']=='original spherical Gaussian'
        assert record['initialization']['spherical_points']==(2600 if row['backend']=='sana' else 26500)
        assert record['aggregation']['persistent_representation']=='spherical native latents'
def complete(row, verbose=False):
    p=folder(row);done=ROOT/'status'/(str(row['index'])+'.json')
    try:
        if not all(x.is_file() for x in (p/'final.png',p/'config.json',done)): return False
        status=read(done)
        if status['state']!='complete':return False
        r=read(p/'config.json');record_valid(r,row)
        assert status['config_sha256']==sha(p/'config.json')
        assert status['image_sha256']==r['output']['sha256']==sha(p/'final.png')
        from studies.all_prompts.common import image_valid
        assert image_valid(p/'final.png')
        return True
    except Exception as e:
        if verbose:emit('INVALID_RESULT',row=row,error=repr(e))
        return False
def publish(row,image,record):
    p=folder(row);p.mkdir(parents=True,exist_ok=True)
    assert not (p/'final.png').exists() and not (p/'config.json').exists(), 'Never overwrite final artifacts'
    image.save(p/'final.tmp.png',format='PNG')
    from studies.all_prompts.common import image_valid
    assert image_valid(p/'final.tmp.png')
    record['output']=dict(filename='final.png',width=4096,height=2048,sha256=sha(p/'final.tmp.png'))
    record_valid(record,row)
    atomic(p/'config.tmp.json',record)
    with (p/'final.tmp.png').open('rb') as f:os.fsync(f.fileno())
    os.replace(p/'final.tmp.png',p/'final.png')
    os.replace(p/'config.tmp.json',p/'config.json')
    atomic(ROOT/'status'/(str(row['index'])+'.json'),dict(state='complete',time=now(),row=row,
        config_sha256=sha(p/'config.json'),image_sha256=sha(p/'final.png'),job=os.environ['SLURM_JOB_ID']))
    assert complete(row)
