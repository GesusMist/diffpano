"""Study paths and strict preservation of historical files."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from studies.all_prompts.common import safe, prompt_record, EXPECTED_PROMPTS
ROOT=Path('/home/shig/diffpano')
BASE_OUT=ROOT/'outputs/10.6gradient-blending/seed0-v1'
BACKEND=os.environ.get('DIFFPANO_GRADIENT_BACKEND','sana')
BACKENDS=('sana','flux','pixeldit','sd35')
if BACKEND not in BACKENDS:raise ValueError('Unsupported gradient-study backend: '+BACKEND)
GATES=BASE_OUT
PILOT_PROMPTS=('ruins','underwater','firework')
FLUX_SCENE_PROMPTS=tuple(p for p in EXPECTED_PROMPTS if p!='native_control')
NEW_FLUX_PROMPTS=tuple(p for p in FLUX_SCENE_PROMPTS if p not in PILOT_PROMPTS)
SUITE=os.environ.get('DIFFPANO_GRADIENT_SUITE','pilot')

def suite_context(backend,suite):
    if backend not in BACKENDS:raise ValueError('Unsupported backend: '+backend)
    if suite=='pilot':return (BASE_OUT if backend=='sana' else BASE_OUT/backend),PILOT_PROMPTS
    if suite=='flux-scenes20':
        if backend!='flux':raise ValueError('The expanded scene study is authorized for FLUX only')
        return ROOT/'outputs/10.7gradient-blending-flux/seed0-v1',FLUX_SCENE_PROMPTS
    raise ValueError('Unknown gradient study: '+suite)

OUT,PROMPTS=suite_context(BACKEND,SUITE)
MODES=('rgb','poisson_mean','poisson_select')

def expected_counts(backend,steps,camera_count=89):
    if backend not in BACKENDS:raise ValueError('Unsupported backend: '+backend)
    if not isinstance(steps,int) or steps<1:raise ValueError('Positive integer step count required')
    predictions=camera_count*steps
    return dict(denoiser=predictions,encode=0 if backend=='pixeldit' else 2*predictions,
                decode=0 if backend=='pixeldit' else predictions+camera_count,initialize=0)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(safe(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temp.write_text(json.dumps(safe(data),indent=2,sort_keys=True)+'\n');os.replace(temp,path)
def core_hashes():
    paths=[ROOT/'diffpano/gradient_fusion.py',ROOT/'studies/gradient_blending/operator.py']
    paths+=sorted((ROOT/'studies/gradient_blending/tests').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def source_hashes():
    paths=[ROOT/'diffpano/gradient_fusion.py']+sorted((ROOT/'studies/gradient_blending').rglob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def preserved():
    record=read(BASE_OUT/'preservation.json')
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip()==record['branch']=='no_sphere'
    for p,h in record['hashes'].items():assert sha(ROOT/p)==h,'Historical source changed: '+p
    return record

def scene_rows():
    return [dict(index=i,prompt=prompt,mode=mode) for i,(prompt,mode) in
            enumerate((p,m) for p in NEW_FLUX_PROMPTS for m in MODES[1:])]

def preserve_prior_results():
    for path,expected in read(OUT/'prior-results-preservation.json').items():
        assert sha(path)==expected,'Pilot artifact changed: '+path

def baseline_audit():
    from studies.all_prompts.audit import configuration,audit_cache
    c,stub,cameras,ids,old=configuration(BACKEND)
    audit_cache(BACKEND,c)
    historical=read(ROOT/'outputs/metrics-evaluation/seed0-v1/inventory.json')
    rows=[r for r in historical['rows'] if r['backend']==BACKEND and r['method']=='diffpano' and
          r['consensus_projection']=='erp' and r['camera_strategy']=='old89' and r['prompt_id'] in PROMPTS and r['steps']==c.generation.num_inference_steps]
    unique={r['prompt_id']:r for r in rows}
    assert len(rows)==len(unique),'Ambiguous cached RGB controls'
    assert set(unique)==set(PROMPTS)
    for name,r in unique.items():
        assert r['completion_status']=='complete' and r['camera_geometry_sha256']==old['camera_geometry_sha256']
        assert sha(r['output_path'])==r['image_sha256']
        assert prompt_record(ROOT/'prompts'/(name+'.txt'))[0]['original_sha256']==r['original_prompt_sha256']
    result=dict(backend=BACKEND,config=c.to_dict(),cameras=[dict(yaw=x.yaw,pitch=x.pitch,roll=x.roll,fov_x=x.fov_x,fov_y=x.fov_y) for x in cameras],
        ids=ids,camera_sha256=old['camera_sha256'],camera_geometry_sha256=old['camera_geometry_sha256'],
        initialization=old['initialization'],prepared_schedule=safe(old['prepared_schedule']),
        schedule_sha256=digest(old['prepared_schedule']),conditioning_sha256=old['conditioning_sha256'],cached_cases=unique,
        path=['ordinary_run','ExperimentalPipeline.advance_interval','predict_clean_and_endpoint','decode_clean',
              'BridgeFactorialPipeline._local_clean_residual','CanvasOperator.accumulate','CanvasOperator.finalize',
              'CanvasOperator.sample_view','BridgeFactorialPipeline._consensus_native_clean','interpolate_from_current_state',
              'ExperimentalPipeline.terminal'] if BACKEND!='pixeldit' else
             ['ordinary_run','ExperimentalPipeline.advance_interval','predict_clean_and_endpoint',
              'native predicted-clean RGB (no VAE)','CanvasOperator.accumulate','CanvasOperator.finalize',
              'CanvasOperator.sample_view','interpolate_from_current_state','ExperimentalPipeline.terminal'],
        fusion_input=dict(meaning=('native predicted-clean RGB (terminal: native RGB; no VAE)' if BACKEND=='pixeldit' else 'predicted-clean decoded RGB (terminal: decoded terminal native state)'),
                         shape=[1,3,1024,1024],inverse_projected_shape=[1,3,2048,4096],dtype='torch.float32',
                         range='nominal [-1,1], unconstrained; actual min/max/out-of-range recorded at selected intervals'),
        reducer='exact RGBFusionAccumulator weighted_average',weight='bilinearly projected exp(-sqrt(x_norm^2+y_norm^2)/0.1)',
        expected_counts=expected_counts(BACKEND,c.generation.num_inference_steps,len(cameras)),
        consensus='every %d existing denoising intervals and terminal assembly'%c.generation.num_inference_steps,
        geometry='saved Old89, 80x80 degrees',source_revision=preserved()['git_head'],
        diagnostic_disk_estimate_gib=2.5,
        source_set_note='Additive study/module; all historical bytes checked individually. Historical equality-of-file-set gates and caches are not modified.')
    write(OUT/'baseline.json',result);return result
