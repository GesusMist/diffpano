"""Study matrix, log-based gates and compact atomic provenance."""
import os
import json
import subprocess
from pathlib import Path
from studies.all_prompts.common import (REPO,sha,digest,read,emit,safe,shell_command,image_valid,
    effective_prompt,prompt_record,lock,BACKENDS,STEPS,fingerprint as historical_fingerprint)
from .cameras import checked_count,cover,angular_hash

STUDY=REPO/'studies/camera_patching'
ROOT=REPO/'outputs/camera-patching-seed0'
BASE=REPO/'outputs/all-prompts-erp-cea-seed0'
PROMPTS=('ruins','underwater','firework')
STRATEGIES=('fibonacci','random')
HISTORICAL_FINGERPRINT='76a48b61173437c463aa8e3730a2eca73fad86acae01c48002e4836afc32bddd'

def strategy_name(strategy,n,layout_seed=0):
    checked_count(n)
    if strategy=='fibonacci':return f'fibonacci_n{n}'
    if strategy=='random':return f'random_n{n}_seed{layout_seed}'
    raise ValueError(strategy)

def output_path(strategy,n,backend,projection,prompt,layout_seed=0):
    return ROOT/strategy_name(strategy,n,layout_seed)/backend/projection/prompt

def rows(camera_count=89):
    camera_count=checked_count(camera_count)
    result=[]
    for strategy in STRATEGIES:
        for backend in BACKENDS:
            for projection in ('erp','cea'):
                for prompt in PROMPTS:
                    result.append(dict(index=len(result),strategy=strategy,num_cameras=camera_count,layout_seed=0 if strategy=='random' else None,
                        fibonacci_phase=0. if strategy=='fibonacci' else None,backend=backend,projection=projection,prompt=prompt,
                        steps=STEPS[backend],output=str(output_path(strategy,camera_count,backend,projection,prompt))))
    assert len(result)==48 and len({r['output'] for r in result})==48
    return result

def fingerprint():
    paths=sorted(p for p in STUDY.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return digest(dict(historical=historical_fingerprint(),study={str(p.relative_to(REPO)):sha(p) for p in paths},
        prompts={n:sha(REPO/'prompts'/(n+'.txt')) for n in PROMPTS}))

def source_record():
    status=shell_command(['git','status','--short'])
    return dict(git_commit=shell_command(['git','rev-parse','HEAD']),git_dirty=bool(status),dirty_status=status,
        fingerprint=fingerprint())

def job_success(job):
    out=shell_command(['sacct','-j',str(job),'-X','-n','-P','--format=JobID%40,JobIDRaw%40,State,ExitCode'])
    return any(str(job) in x.split('|')[:2] and x.split('|')[2:4]==['COMPLETED','0:0'] for x in out.splitlines())

def gate(event,pattern,*,backend=None):
    current=fingerprint()
    for p in sorted((REPO/'logs').glob(pattern),key=lambda p:p.stat().st_mtime,reverse=True):
        for line in reversed(p.read_text().splitlines()):
            if line.startswith(event+' '):
                v=json.loads(line[len(event)+1:])
                if v['fingerprint']==current and (backend is None or v.get('backend')==backend) and job_success(v['job']):return v
    raise RuntimeError('No successful current gate: '+event+(' '+backend if backend else ''))

def validation_gate():return gate('CAMERA_VALIDATION_PASSED','campatch-validate.*.out')
def preflight_gate(backend):return gate('CAMERA_PREFLIGHT_PASSED','campatch-preflight.*.out',backend=backend)

def expected_counts(m,steps,pixel=False,terminal=True):
    predictions=m*steps
    return dict(denoiser=predictions,encode=0 if pixel else 2*predictions,
        decode=0 if pixel else predictions+(m if terminal else 0),initialize=0)

def requested_camera(row,view):
    return cover(row['strategy'],view,row['num_cameras'],layout_seed=row['layout_seed'] or 0,phase=row['fibonacci_phase'] or 0.)

def validate_config(record,row):
    from diffpano.config import ViewConfig
    c=record['camera'];p=record['projection'];b=record['backend'];g=record['generation'];t=record['trajectory'];f=record['fusion']
    assert record['study']=='camera_patching' and record['method']=='diffpano'
    assert c['strategy']==row['strategy'] and c['num_cameras']==checked_count(row['num_cameras'])
    assert c['layout_seed']==row['layout_seed'] and c['fibonacci_phase']==row['fibonacci_phase']
    assert c['fov_x_deg']==c['fov_y_deg']==80 and c['fixed_during_trajectory']
    cameras=requested_camera(row,ViewConfig(height=b['local_rgb_height'],width=b['local_rgb_width'],fov_x=80,fov_y=80))
    assert angular_hash(cameras)==c['angular_geometry_sha256']
    assert b['name']==row['backend'] and p['consensus_canvas']==row['projection']
    assert (p['height'],p['width'],p['final_height'],p['final_width'],p['final_export'])==(2048,4096,2048,4096,'erp')
    assert record['prompt']['name']==row['prompt'] and record['prompt']['sha256']==sha(REPO/'prompts'/(row['prompt']+'.txt'))
    assert g['seed']==0 and g['batch_size']==1 and g['num_inference_steps']==row['steps']
    assert (f['mode'],f['weight_mode'],f['temperature'],f['warp_mode'])==('weighted_average','spherediff_center',.1,'standard')
    assert (f['perspective_to_canvas_interpolation'],f['canvas_to_perspective_interpolation'])==('bilinear','nearest')
    assert t['transition']=='preserve_current_state' and t['time_travel'] is False
    assert t['replay_count']==t['backward_count']==t['original_noise_reinjection_count']==0
    assert t['counts']==expected_counts(len(cameras),row['steps'],row['backend']=='pixeldit')
    assert record['initialization']['method']=='gwtflow' and record['initialization']['source_seed']==0
    assert record['bridge']['mode']==('not_applicable_pixeldit' if row['backend']=='pixeldit' else 'local_identity_preserving')
    from studies.all_prompts.common import BASES
    old=read(BASES['ruins']/row['backend']/'metadata.json')
    assert b['model_id']==old['model_checkpoint'] and b['model_revision']==old['model_revision']
    assert b['precision']=='bfloat16'
    assert (b['local_rgb_height'],b['local_rgb_width'])==tuple(old['local_RGB_resolution'])
    assert b['native_channels']==old['native_channels'] and [b['native_height'],b['native_width']]==old['local_native_resolution']
    cfg=old['config'];expected_guidance=cfg['pixeldit']['cfg_scale'] if row['backend']=='pixeldit' else cfg['generation']['guidance_scale']
    assert g['guidance_scale']==expected_guidance and g['true_cfg_scale']==cfg['generation']['true_cfg_scale']
    assert g['prepared_schedule_sha256']==digest(safe(old['prepared_schedule']))
    assert record['initialization']['source_shape']==old['initialization']['source_shape']
    assert record['initialization']['native_scaling']==old['native_scale'] and record['initialization']['scaling_applications']==1
    assert record['initialization']['denoiser_calls']==record['initialization']['vae_calls']==0
    if row['backend']=='pixeldit':
        assert g['interval_guidance']==cfg['pixeldit']['interval_guidance'] and g['negative_prompt']==cfg['pixeldit']['negative_prompt']
        assert b['config_sha256']==sha(REPO/cfg['pixeldit']['config_path'])
    assert t['final_cea_export_count']==(1 if row['projection']=='cea' else 0) and t['original_noise_bank_retained'] is False
    assert record['output']['filename']=='final.png' and (record['output']['width'],record['output']['height'])==(4096,2048)

def complete(row,*,check_exit=True,verbose=False):
    folder=Path(row['output']);png=folder/'final.png';config=folder/'config.json'
    if not png.exists() or not config.exists():return False
    try:
        if not image_valid(png):raise ValueError('Invalid PNG')
        r=read(config);validate_config(r,row)
        assert r['output']['sha256']==sha(png)
        if check_exit:assert job_success(r['execution']['job_id'])
        return True
    except Exception as e:
        if verbose:emit('INCOMPLETE_RESULT',row=row,error=repr(e))
        return False

def publish(image,record,row):
    folder=Path(row['output']);folder.mkdir(parents=True,exist_ok=True)
    if complete(row):raise FileExistsError('Valid completed output must not be overwritten')
    image.save(folder/'final.tmp.png',format='PNG')
    record['output']=dict(filename='final.png',width=4096,height=2048,sha256=sha(folder/'final.tmp.png'))
    validate_config(record,row);assert image_valid(folder/'final.tmp.png')
    (folder/'config.tmp.json').write_text(json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+'\n')
    validate_config(read(folder/'config.tmp.json'),row)
    for name in ('final.tmp.png','config.tmp.json'):
        with (folder/name).open('rb') as f:os.fsync(f.fileno())
    os.replace(folder/'final.tmp.png',folder/'final.png')
    os.replace(folder/'config.tmp.json',folder/'config.json')
    assert complete(row,check_exit=False)
