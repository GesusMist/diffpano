"""Read-only historical adapters: integrity plus semantics, never current-source equality.

The inventory is a locator, not proof. Every accepted candidate is checked against
its original PNG and completion record/log, with source gates compared to their
own archived records. The adopter writes only the new suite.
"""
import argparse
import collections
import csv
import functools
import hashlib
import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/shig/diffpano')
OUT = ROOT / 'outputs/10.15layout24-flux216/seed0-v1'
INVENTORY = ROOT / 'outputs/metrics-evaluation/seed0-v1/inventory.json'
REVISION = 'fa45a9eb6808ba8fdfc7cc2756f7f1a16e0921f4'
OFFICIAL_REVISION = '3de623fc3c33e44ffbe2bad470d0f45bccf2eb21'
OLD = 'b55b4eb5051cdfb8c579e81d4f51ece1f15aeb0e8d7e1fde6b7b74a49154435c'
UNIFORM = 'aabecc0b2d29255a67ffcc29b0e80168358b00c34671423cfe1c488bfbb74499'
NOISE = 'f9d9e74986551c2d700e56166430df0ab30ee38c8c36b128720ad22b9219a551'
SCHEDULE = 'b2caee458b8279600e77e5dfd78048b0544ce178981c397a3189edc2d430b960'


def read(p): return json.loads(Path(p).read_text())
def digest(v): return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(str(tmp), str(path))
def require(condition, reason):
    if not condition: raise ValueError(reason)
@functools.lru_cache(None)
def hashed(p): return dict(path=str(p), resolved_path=str(Path(p).resolve()), sha256=sha(p))
def png(path):
    with Path(path).open('rb') as f: header = f.read(33)
    require(header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR', 'Not an original PNG')
    width, height, bits, color = struct.unpack('>IIBB', header[16:26])
    require((width,height,bits,color) == (4096,2048,8,2), 'Wrong ERP size, depth, or color format')
    return sha(path)


def events(path):
    result = {}
    for line in Path(path).read_text(errors='replace').splitlines():
        name, sep, value = line.partition(' ')
        if sep:
            try: result[name] = json.loads(value)
            except ValueError: pass
    return result


@functools.lru_cache(None)
def references():
    baseline = read(ROOT / 'outputs/10.7gradient-blending-flux/seed0-v1/baseline.json')
    result = {'old89': dict(geometry=OLD, initial='0841cf95c51858d6641f8b4780a4c0ef205f6d1f068570feef529675a6df78a7',
                           slots=list(range(89))),
              'uniform89': dict(geometry=UNIFORM, initial='71de281d7ca14c0fb27abbf1249ad22d5a06a8674d3afcd7d95e41fbae251e18',
                               slots=['fibonacci:n089:%03d' % n for n in range(89)])}
    for layout in ('old70', 'old50', 'uniform70', 'uniform50'):
        strategy = 'old' if layout.startswith('old') else 'fibonacci'; n = int(layout[-2:])
        source = ROOT / ('outputs/camera-count-erp-seed0-v2/layouts/%s_n%d.json' % (strategy,n))
        saved = read(source)
        pf = read(ROOT / ('outputs/camera-count-erp-seed0-v2/preflight/diffpano-%s-n%d-flux.json' % (strategy,n)))
        require(saved['coverage']['diffpano']['passed'] and pf['passed'], 'Historical coverage/preflight failed')
        require(pf['layout_hash'] == saved['angular_geometry_sha256'], 'Preflight geometry mismatch')
        result[layout] = dict(geometry=saved['angular_geometry_sha256'], initial=pf['record']['initialization']['initial_state_sha256'],
                             slots=saved['camera_ids'], source=hashed(source))
    result['schedule'] = baseline['prepared_schedule']
    manifest = ROOT / 'outputs/10.9ournew-flux28/seed0-v1/manifest.json'
    weights = read(manifest)['weight_audit']
    require(weights['all_shared_files_identical'], 'Official/DiffPano checkpoint equivalence was not verified')
    require(all(v['identical'] and v['methods']['diffpano']['sha256'] == v['methods']['spherediff']['sha256']
                for v in weights['files']), 'Unequal audited checkpoint bytes')
    result['weight_evidence'] = dict(hashed(manifest), weight_audit_sha256=digest(weights),
                                   files=len(weights['files']), all_shared_files_identical=True)
    return result


def layout_name(strategy):
    if strategy in ('old89','official','official_spherediff'): return 'official' if 'official' in strategy else strategy
    if strategy == 'uniform89': return strategy
    return strategy.replace('fibonacci_n', 'uniform').replace('old_n', 'old')


def prompt_check(record, row):
    path = Path(row['prompt_path']); raw = path.read_bytes(); lines = raw.decode('utf-8').splitlines()
    require(len(lines)==5 and all(x.strip() for x in lines), 'Target prompt format invalid')
    original = record.get('original_sha256',record.get('sha256'))
    effective = record.get('effective_sha256',original)
    require(original==sha(path) and effective==sha(path), 'Prompt bytes/effective hash mismatch')
    if 'effective_lines' in record: require(record['effective_lines']==lines, 'Prompt line order/text mismatch')
    recorded_path = record.get('path')
    if recorded_path:
        recorded_path=Path(recorded_path)
        if not recorded_path.is_absolute(): recorded_path=ROOT/recorded_path
        require(recorded_path.resolve()==path.resolve(), 'Wrong prompt folder/path')
    return dict(original_sha256=original,effective_sha256=effective,path=str(path),lines=lines)


def config_check(c):
    require(c['model']['id']=='ModelsLab/flux.1-dev' and c['model']['revision']==REVISION, 'Wrong model/revision')
    require(c['model']['precision']=='bf16', 'Wrong precision')
    require(c['generation']==dict(batch_size=1,guidance_scale=3.5,num_inference_steps=20,true_cfg_scale=1.0), 'Wrong steps/guidance/batch')
    require(c['experiment']['seed']==0, 'Wrong seed')
    require(c['erp']==dict(height=2048,width=4096), 'Wrong generation ERP resolution')
    require(c['view']==dict(height=1024,width=1024,fov_x=80.0,fov_y=80.0), 'Wrong local geometry')
    require(c['canvas']['mode']=='erp' and c['warp']['mode']=='standard', 'Wrong projection/warp')
    require(c['warp']['erp_to_perspective']['interpolation']=='nearest' and c['warp']['perspective_to_erp']['interpolation']=='bilinear', 'Wrong interpolation')
    require(c['consensus_transition']==dict(mode='preserve_current_state',vae_residual_correction=True), 'Wrong bridge')
    require(not c['sampling']['rotation']['enabled'], 'Camera rotation enabled')
    require(c['fusion']['mode']=='weighted_average' and c['fusion']['weight_mode']=='spherediff_center' and c['fusion']['spherediff_temperature']==.1, 'Wrong RGB reference weights')
    require(c['prompt']['negative_path'] is None, 'Different negative conditioning')
    ref=c['global_pipeline'].get('refinement',{})
    if ref: require(ref['last_fraction']==0, 'Refinement tail enabled')


def normalize_inventory(r):
    """Verify a locator against actual original logs/configs/status before using it."""
    require(r['completion_status']=='complete', 'Historical completion is '+r['completion_status'])
    image=Path(r['output_path']); h=png(image)
    require(h==r['image_sha256'], 'Image bytes differ from archived inventory')
    evidence=[hashed(INVENTORY)]; sources=[Path(p) for p in r['source_provenance']]
    evidence.extend(hashed(p) for p in sources)
    require(r.get('accounting')==dict(exit_code='0:0',state='COMPLETED'), 'Historical Slurm completion absent')
    family=r['family']; m=r['configuration']; provenance={}
    if family=='benchmark':
        if sources[0].suffix=='.out':
            log=events(sources[0]); success=log['SUCCESS_CASE']; runtime=log['RUNTIME_PROVENANCE']
            require(success['image_sha256']==h, 'Original log image hash mismatch')
            require(runtime['configuration']==m and runtime['row']['steps']==20, 'Inventory differs from original run configuration')
            require(runtime['row']==success['row'] and runtime['row']['projection']==r['consensus_projection'],'Locator projection differs from original runtime row')
            validation_log=ROOT/'logs/allp21-validate.19898944.out'
            gate=events(validation_log)['SWEEP_VALIDATION_PASSED']
            require(log['WORKER_START']['source_fingerprint']==gate['fingerprint'], 'Archived original source fingerprint mismatch')
            require(log['WORKER_START']['prompt_inventory_sha256']==gate['inventory_sha256'],'Archived original prompt inventory differs')
            evidence.append(hashed(validation_log))
            require(log['PROMPT']['original_sha256']==r['original_prompt_sha256'], 'Log prompt mismatch')
            require(log['INITIALIZATION']==r['initialization'], 'Log initialization mismatch')
            require(log['GENERATION_VALIDATED']==r['efficiency'], 'Log execution counters mismatch')
            provenance=runtime['provenance']; original_metadata=sources[0]
        else:
            original=read(sources[0]); log=Path(sources[1]).read_text(errors='replace')
            proofs=[]
            for line in log.splitlines():
                if line.startswith('REUSE_APPROVED '):
                    q=json.loads(line.split(' ',1)[1])
                    if q['row']['output']==str(image): proofs.append(q)
            require(len(proofs)==1, 'Original symlink reuse proof absent')
            proof=proofs[0]
            require(proof['image_sha256']==h and proof['metadata_sha256']==sha(sources[0]), 'Original reuse proof hashes mismatch')
            require(Path(proof['source']).resolve()==image.resolve(), 'Original reuse alias target changed')
            require(proof['row']['projection']==r['consensus_projection'],'Locator projection differs from original reuse proof')
            require(original.get('config',original.get('generation_configuration'))==m, 'Original configuration changed')
            provenance=original; original_metadata=sources[0]
        init=r['initialization'].get('record',r['initialization'])
        normalized=dict(method='diffpano',layout='old89',mode='rgb',config=m,initialization=init,
                        geometry=r['camera_geometry_sha256'],counts=r['efficiency'].get('counts'),
                        schedule=r.get('schedule'),schedule_sha256=r.get('schedule_sha256'),
                        refinement=0.,details=r['efficiency'].get('details',{}),fusion={},
                        conditioning_sha256=provenance.get('conditioning_sha256'),fusions=[])
    else:
        original_metadata=sources[0]; actual=read(original_metadata)
        require(actual==m, 'Archived inventory configuration differs from original metadata')
        require(m['output']['sha256']==h, 'Original config image hash mismatch')
        if family=='step_control':
            state=ROOT/('outputs/flux-step-controls-seed0/status/%d.json'%m['row']['index'])
            status=read(state);require(status['state']=='complete','Original control incomplete')
            require(status['config_sha256']==sha(original_metadata) and status.get('image_sha256',m['output']['sha256'])==h,'Original control status hash mismatch')
            evidence.append(hashed(state))
            manifest=ROOT/'outputs/flux-step-controls-seed0/manifest.json'
            require(sha(manifest)==m['manifest_sha256'],'Original control manifest changed')
            evidence.append(hashed(manifest))
            normalized=official(m)
        elif family=='camera':
            require(m['source']['fingerprint']=='d4f9de4d97b29ba1fa72c8a8e33a63bd12ac74e5adb4c155dd816fdef43d5cf5','Unknown archived camera source scope')
            normalized=dict(method='diffpano',layout=layout_name(r['camera_strategy']),mode='rgb',
                            config=None,initialization=m['initialization'],geometry=m['camera']['angular_geometry_sha256'],
                            counts=m['trajectory']['counts'],schedule_sha256=m['generation']['prepared_schedule_sha256'],
                            refinement=0.,details=m['trajectory'],fusion={},fusions=[],camera_schema=m)
        elif family=='camera_count':
            state=read(sources[1]);require(state['state']=='complete','Original camera count incomplete')
            require(state['config_sha256']==sha(sources[0]) and state['image_sha256']==h,'Camera count status hash mismatch')
            validation=ROOT/'outputs/camera-count-erp-seed0-v2/validation.json'; gate=read(validation)
            require(gate['passed'] and gate['source_hashes']==m['source_hashes'],'Own archived source validation differs')
            require(sha(m['camera']['layout_reference'])==m['camera']['layout_sha256'],'Saved layout changed')
            evidence.append(hashed(validation))
            normalized=dict(method='diffpano',layout=layout_name(r['camera_strategy']),mode='rgb',
                            config=m['configuration'],initialization=m['initialization'],geometry=m['camera']['angular_geometry_sha256'],
                            counts=m['execution']['counts'],schedule=m['schedule'],schedule_sha256=m['schedule_sha256'],
                            refinement=0.,details=m['trajectory'],fusion={},fusions=[],conditioning_sha256=m['runtime_audit'].get('conditioning_sha256'))
        else: raise ValueError('Unrecognized historical inventory family')
    normalized.update(image_path=str(image),image_sha256=h,original_metadata_path=str(original_metadata),
                      original_metadata_sha256=sha(original_metadata),evidence=evidence,
                      prompt_record=r['prompt'],backend=r['backend'],steps=r['steps'],seed=r['seed'],consensus_projection=r['consensus_projection'],
                      generation_seconds=(r.get('efficiency') or {}).get('generation_seconds',(r.get('efficiency') or {}).get('denoising_and_fusion_seconds')),
                      original_provenance=dict(family=family,id=r['id'],job=r.get('job'),efficiency=r.get('efficiency'),
                                               source_revision=m.get('source',{}).get('git_commit') if isinstance(m.get('source'),dict) else None))
    return normalized


def official(m):
    c=m['configuration']; require(c['pipeline']=='SphericalFluxPipeline', 'Not original SphereDiff')
    require(c['model_source']=='black-forest-labs/FLUX.1-dev' and c['revision']==OFFICIAL_REVISION,'Wrong original checkpoint')
    call=c['call'];require(call==dict(erp_height=2048,erp_width=4096,guidance_scale=3.5,n_spherical_points=26500,num_inference_steps=20,true_cfg_scale=1.0,weighted_average_temperature=.1), 'Wrong original SphereDiff invocation')
    require(m['official_source_commit']=='2c8c68ba088f2803b3dce4b52b7b0d68bc996139','Wrong original algorithm revision')
    require(m['official_source_hashes']['pipelines_ours/pipeline_spherical_flux.py']=='641e5c839188887939e8e2346e971a001f1444bfe1dce760b04487ae4e03fa7e','Official pipeline source mismatch')
    require(m['initialization']['initial_state_sha256']=='eb663bb2a580fbce2822fb4ba2b154c437bfe0b4b6f904102fc90998287cca4b','Wrong original spherical seed/noise')
    require(m['execution']['counts']==dict(decode=89,denoiser=1780,encode=0),'Wrong measured original counts')
    require(len(m['schedule']['timesteps'])==20 and len(m['schedule']['sigmas'])==21,'Wrong original schedule length')
    reference=read(ROOT/'outputs/flux-step-controls-seed0/spherediff/flux/steps20/air_balloons/config.json')['schedule']
    require(m['schedule']==reference,'Original SphereDiff schedule differs')
    return dict(method='spherediff',layout='official',mode=None,config=c,initialization=m['initialization'],
                geometry='official_source_2c8c68b',counts=m['execution']['counts'],schedule=m['schedule'],
                schedule_sha256=m['schedule_sha256'],refinement=0.,fusion={},fusions=[],details={},
                official_source_commit=m['official_source_commit'],official_source_hashes=m['official_source_hashes'])


def normalize_metadata(path):
    path=Path(path);m=read(path);folder=path.parent;image=folder/'final.png';h=png(image)
    statuspath=folder/'status.json';status=read(statuspath)
    require(status['state']=='complete','Historical status not complete')
    require(status['metadata_sha256']==sha(path) and status['image_sha256']==h,'Historical completion artifact hashes mismatch')
    require(m['artifacts']['final.png']==h,'Historical final artifact mismatch')
    evidence=[hashed(path),hashed(statuspath)]
    if 'config.json' in m['artifacts']:
        require(sha(folder/'config.json')==m['artifacts']['config.json'],'Original config integrity failure');evidence.append(hashed(folder/'config.json'))
    suite=next(p for p in path.parents if p.name=='seed0-v1')
    manifest=suite/'manifest.json'
    if 'source_sha256' in m:
        saved=read(manifest)
        require(saved['source_hashes']==m['source_hashes'] and digest(m['source_hashes'])==m['source_sha256'],'Own archived source hash mismatch')
        require(any(r['semantic_sha256']==m['semantic_sha256'] for r in saved['rows']),'Own archived semantic case missing')
        evidence.append(hashed(manifest))
    elif m.get('reused_from'):
        origin=Path(m['reused_from'])/'metadata.json'
        proof=m['reuse_audit']
        require(proof['passed'] and proof['metadata_sha256']==sha(origin) and proof['image_sha256']==h,'Historical alias proof mismatch')
        require(read(origin).get('source_hashes')==m.get('source_hashes'),'Alias historical source hash changed')
        evidence.append(hashed(origin))
    elif 'source_hashes' in m:
        gates=[suite/'validation.json',suite/'extension-validation.json',suite/'manifest.json',suite/'pilot-submission.json',suite/'extension-submission.json']
        matching=[p for p in gates if p.exists() and read(p).get('source_hashes')==m['source_hashes']]
        require(matching,'No matching own archived source gate');evidence.append(hashed(matching[0]))
    if 'official' in m:
        normalized=official(m['official'])
    else:
        geo=m.get('camera_geometry_sha256');layout='old89' if geo==OLD else 'uniform89' if geo==UNIFORM else 'unknown'
        fusion=m['fusion'];mode=fusion['mode']
        normalized=dict(method='diffpano',layout=layout,mode=mode,config=m['configuration'],initialization=m['initialization'],
                        geometry=geo,counts=m['counts'],schedule=m.get('provenance',{}).get('prepared_schedule'),
                        schedule_sha256=m['schedule_sha256'],refinement=m.get('refinement',{}).get('last_fraction',0.),
                        details=m.get('details',{}),fusion=fusion,fusions=m.get('fusions',[]),
                        conditioning_sha256=m.get('provenance',{}).get('conditioning_sha256'),
                        geometry_record=m.get('geometry'),audit=m.get('audit'))
        if m.get('reused'):
            # Wrapper records must continue pointing to the original audited source.
            cached=m.get('cached_case')
            require(cached is not None or m.get('reused_from'),'Missing original source for historical wrapper')
            verified=normalize_inventory(cached) if cached is not None else normalize_metadata(Path(m['reused_from'])/'metadata.json')
            require(verified['image_sha256']==h,'Wrapped image differs from original')
            evidence.extend(verified['evidence'])
            normalized['conditioning_sha256']=verified.get('conditioning_sha256')
            normalized['verified_source_projection']=verified['consensus_projection']
    normalized.update(image_path=str(image),image_sha256=h,original_metadata_path=str(path),original_metadata_sha256=sha(path),
                      evidence=evidence,prompt_record=m['prompt'],consensus_projection=m.get('details',{}).get('active_consensus_projection', 'spherical' if 'official'in m else m['configuration'].get('canvas',{}).get('mode')),backend=m.get('backend',m.get('case',{}).get('backend','flux')),
                      steps=m.get('case',{}).get('steps',m['configuration'].get('generation',{}).get('num_inference_steps',20)),
                      seed=m.get('case',{}).get('seed',m['configuration'].get('experiment',{}).get('seed',0)),
                      generation_seconds=m.get('generation_seconds',m.get('historical_runtime_seconds')),
                      original_provenance=dict(source_revision=m.get('implementation_revision',m.get('source_revision')),
                                               source_hashes_sha256=digest(m['source_hashes']) if 'source_hashes'in m else None,
                                               semantic_sha256=m.get('semantic_sha256'),job=m.get('job'),reused=m.get('reused',False)))
    return normalized


def compatibility(n,row):
    """Pure semantic comparison, deliberately independent of checkout source hash."""
    require(n['backend']=='flux' and n['steps']==20 and n['seed']==0,'Wrong backend/steps/seed')
    require(n['method']==row['method'] and n['layout']==row['layout'],'Different method/layout/count')
    require(n['mode']==row['gradient_mode'],'Different fusion guidance')
    prompt_record=prompt_check(n['prompt_record'],row)
    require(n['refinement']==0.,'Independent refinement tail is not allowed')
    if n['method']=='spherediff': return prompt_record
    require(n['consensus_projection']=='erp','Actual consensus projection is not ERP')
    require(n.get('verified_source_projection','erp')=='erp','Wrapped source actual projection is not ERP')
    count=row['camera_count'];expected=references()[row['layout']]
    require(n['geometry']==expected['geometry'],'Ordered camera geometry/FOV differs')
    require(n['counts']==dict(denoiser=20*count,encode=40*count,decode=21*count,initialize=0),'Measured counts do not describe no-tail RGB bridge')
    require(n['schedule_sha256']==SCHEDULE,'Prepared schedule differs')
    if n.get('schedule') is not None:require(n['schedule']==references()['schedule'],'Actual timesteps/sigmas differ')
    init=n['initialization'];method=init.get('method','')
    require('gwt' in method.lower(),'Initialization is not GWTF')
    require(init.get('source_sha256')==NOISE,'GWTF source noise differs')
    local=init.get('initial_local_sha256',init.get('initial_state_sha256'))
    require(local==expected['initial'],'GWTF local states/noise slots/native scaling differ')
    if 'camera_slots' in init:require(init['camera_slots']==expected['slots'],'Canonical noise slots differ')
    require(init.get('scaling',init.get('native_scaling',1.))==1.,'Native scaling differs')
    require(init.get('scaling_applications',1)==1,'Native scaling count differs')
    if n['config'] is not None:config_check(n['config'])
    else:
        c=n['camera_schema'];g=c['generation'];b=c['backend'];f=c['fusion'];cam=c['camera']
        require(b['name']=='flux' and b['model_id']=='ModelsLab/flux.1-dev' and b['model_revision']==REVISION,'Camera schema checkpoint differs')
        require(b['precision']=='bfloat16' and b['local_rgb_height']==b['local_rgb_width']==1024 and b['native_height']==b['native_width']==128,'Camera precision/local size differs')
        require(g['guidance_scale']==3.5 and g['true_cfg_scale']==1. and g['num_inference_steps']==20 and g['seed']==0 and g['batch_size']==1,'Camera generation settings differ')
        require(cam['num_cameras']==count and cam['fov_x_deg']==cam['fov_y_deg']==80 and cam['roll']==0 and cam['fixed_during_trajectory'],'Camera geometry differs')
        require(cam['fibonacci_phase']==0.,'Uniform phase differs')
        require(c['projection']['consensus_canvas']=='erp' and c['projection']['final_export']=='erp','Wrong canvas projection')
        require(c['bridge']['mode']=='local_identity_preserving','Bridge differs')
        require(f==dict(canvas_to_perspective_interpolation='nearest',mode='weighted_average',perspective_to_canvas_interpolation='bilinear',temperature=.1,warp_mode='standard',weight_mode='spherediff_center'),'Camera RGB fusion differs')
    details=n.get('details',{})
    require(details.get('active_consensus_projection','erp')=='erp','Runtime consensus projection is not ERP')
    require(not details.get('time_travel',details.get('time_travel_enabled',False)),'Time travel enabled')
    for key in ('replay_count','backward_count','backward_calls','original_noise_reinjection_count','original_noise_reinjections'):
        require(details.get(key,0)==0,'Extra trajectory operations: '+key)
    if n['mode']=='poisson_select':
        f=n['fusion'];require(f.get('reference_mode','screened')=='screened','Reference penalty differs')
        require(f.get('gradient_mix_alpha') is None,'Mean/max alpha is not applicable')
        require((f['lambda_color'],f['relative_tolerance'],f['absolute_tolerance'],f['max_iterations'])==(.1,1e-5,1e-7,200),'Screened solver policy differs')
        require(len(n['fusions'])==21 and n['fusions'][-1]['terminal'],'Missing coupled/terminal Poisson solve')
        for solve in n['fusions']:
            require(solve['mode']=='poisson_select' and solve.get('reference_mode','screened')=='screened' and solve['converged'],'Wrong or nonconverged solve')
            require(solve.get('gradient_mix_alpha') is None,'Mixed guidance observed')
            require(solve.get('final_true_residual',0)<=solve.get('stopping_threshold',float('inf')),'Residual exceeds recorded threshold')
    return prompt_record


def catalog():
    result=[]
    for r in read(INVENTORY)['rows']:
        if r.get('backend')!='flux' or r.get('prompt_id')=='native_control':continue
        # Keep incompatible 28-step/CEA/random/count variants visible as rejected candidates.
        result.append(dict(kind='inventory',locator=r,prompt=r['prompt_id'],path=r['output_path'],
                           family=r['family'],method=r['method']))
    roots=['10.6gradient-blending/seed0-v1/flux','10.7gradient-blending-flux/seed0-v1',
           '10.8gradient-refinement-flux/seed0-v1','10.9ournew-flux28/seed0-v1',
           '10.14mean-max-old89-controls/seed0-v1']
    for root in roots:
        for path in sorted((ROOT/'outputs'/root/'cases').rglob('metadata.json')):
            m=read(path);name=m['prompt'].get('name',m.get('case',{}).get('prompt'))
            result.append(dict(kind='metadata',locator=str(path),prompt=name,path=str(path.parent/'final.png'),
                               family=root,method='spherediff' if 'official'in m else 'diffpano'))
    return result


def audit(rows=None,publish=True,adopt_reused=False):
    from studies.layout24 import common
    rows=common.definitions() if rows is None else rows
    require(len(rows)==216 and len({r['case_id']for r in rows})==216,'Expected exact 216 logical rows')
    candidates=catalog();normalized={};errors={};audited=[];plans=[]
    for candidate in candidates:
        key=(candidate['kind'],candidate['path'])
        try:
            normalized[key]=normalize_inventory(candidate['locator']) if candidate['kind']=='inventory' else normalize_metadata(candidate['locator'])
        except (ValueError,KeyError,OSError,AssertionError,TypeError,AttributeError) as e:errors[key]=type(e).__name__+': '+str(e)
    for row in rows:
        choices=[]
        for candidate in candidates:
            if candidate['prompt']!=row['prompt']:continue
            key=(candidate['kind'],candidate['path']);error=errors.get(key);n=normalized.get(key)
            record=dict(index=row['index'],case_id=row['case_id'],prompt=row['prompt'],variant=row['variant'],
                        candidate=candidate['path'],candidate_family=candidate['family'],status='rejected',
                        artifact_sha256=n.get('image_sha256') if n else None,semantic_differences=[],reasons=[])
            if error:record['reasons']=[error]
            else:
                try:
                    prompt_check(n['prompt_record'],row);compatibility(n,row)
                    record['status']='accepted';record['reasons']=['Actual original artifact and completion provenance verified; target semantics match']
                    choices.append(n)
                except (ValueError,KeyError,TypeError) as e:
                    record['reasons']=[str(e)];record['semantic_differences']=[str(e)]
            audited.append(record)
        # Prefer original sources over wrapper aliases; all valid aliases are retained.
        choices.sort(key=lambda c:(bool(c['original_provenance'].get('reused')),c['image_path']))
        plan=dict(index=row['index'],case_id=row['case_id'],prompt=row['prompt'],variant=row['variant'],state='reused' if choices else 'generate',
                  reason='Verified compatible historical result' if choices else 'No unambiguous compatible local original result; generate requested target',
                  aliases=[dict(path=c['image_path'],sha256=c['image_sha256'],metadata=c['original_metadata_path'])for c in choices])
        if choices:
            plan['source']=choices[0]
            if adopt_reused:adopt(row,choices[0])
        plans.append(plan)
    paired=[]
    for name in common.PROMPTS:
        for layout in ('old89','uniform89'):
            pair=[p for p in plans if p['prompt']==name and p['state']=='reused' and p['source']['layout']==layout]
            if len(pair)!=2:continue
            hashes=[p['source'].get('conditioning_sha256') for p in pair]
            require(hashes[0] is not None and hashes[0]==hashes[1],'Reused RGB/select conditioning differs for '+name+'/'+layout)
            paired.append(dict(prompt=name,layout=layout,case_ids=[p['case_id']for p in pair],conditioning_sha256=hashes[0],
                               initial_local_sha256=references()[layout]['initial'],geometry_sha256=references()[layout]['geometry'],schedule_sha256=SCHEDULE))
    per_variant={v:{state:sum(r['variant']==v and r['state']==state for r in plans)for state in ('reused','generate','blocked')}for v in common.VARIANTS}
    require(all(sum(c.values())==24 for c in per_variant.values()),'Variant total mismatch')
    counts={s:sum(p['state']==s for p in plans)for s in ('reused','generate','blocked')}
    report=dict(created=datetime.now(timezone.utc).isoformat(),adapter_version='layout24-reuse-v1',
                historical_source_policy='Verify original completion hashes/logs and own archived gates, then semantic equality; never compare whole-repository hashes to current source',
                inventory=hashed(INVENTORY),weight_equivalence_evidence=references()['weight_evidence'],
                excluded_source_families={'mean-max-old89-flux90':'Wrong gradient guidance; preserved and never accepted',
                                         'mean-max-firefly-retry':'Wrong gradient guidance/reference; preserved and never accepted'},
                target_count=216,candidate_records=len(audited),counts=counts,per_variant=per_variant,paired_RGB_select_identity_checks=paired,rows=audited)
    plan=dict(created=report['created'],artifact_kind='panorama',logical_slots=216,counts=counts,per_variant=per_variant,rows=plans)
    if publish:
        write(OUT/'reuse_audit.json',report);write(OUT/'generation_plan.json',plan)
        with (OUT/'reuse_audit.csv').open('w',newline='')as stream:
            fields=['index','case_id','prompt','variant','candidate','candidate_family','status','artifact_sha256','reasons','semantic_differences']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for r in audited:writer.writerow({k:json.dumps(r[k])if isinstance(r[k],(list,dict))else r[k]for k in fields})
    print('REUSE_AUDIT',json.dumps(dict(counts=counts,per_variant=per_variant),sort_keys=True),flush=True)
    return report,plan


def adopt(row,candidate):
    from studies.layout24.common import folder
    compatibility(candidate,row)
    source=Path(candidate['image_path']);require(sha(source)==candidate['image_sha256'],'Source changed before adoption')
    path=folder(row)
    if (path/'status.json').exists():
        require(verify_reused(row),'Existing suite-local case is not a valid identical reuse');return
    path.mkdir(parents=True,exist_ok=True)
    require(not (path/'final.png').exists(),'Never overwrite existing suite-local image')
    os.symlink(str(source.resolve()),str(path/'final.png'))
    m=dict(reused=True,artifact_kind='panorama',case_id=row['case_id'],variant=row['variant'],prompt=row['prompt'],
           source=candidate,original_metadata_path=candidate['original_metadata_path'],
           original_metadata_sha256=candidate['original_metadata_sha256'],artifacts={'final.png':candidate['image_sha256']},
           counts=candidate['counts'],generation_seconds=candidate.get('generation_seconds'),
           historical_source_revision=candidate['original_provenance'].get('source_revision'),
           provenance_note='Immutable historical image; not generated by this suite or its current source revision')
    write(path/'metadata.json',m)
    write(path/'status.json',dict(state='complete',reused=True,image_sha256=candidate['image_sha256'],metadata_sha256=sha(path/'metadata.json'),
                                   case_id=row['case_id'],variant=row['variant']))
    require(verify_reused(row),'Adopted reuse failed content verification')


def verify_reused(row,verify=True):
    from studies.layout24.common import folder
    try:
        path=folder(row);s=read(path/'status.json');m=read(path/'metadata.json');c=m['source']
        require(s['state']=='complete' and s['reused'] and m['reused'],'Not completed reuse')
        require(s['case_id']==m['case_id']==row['case_id'] and s['variant']==row['variant'],'Reuse target identity mismatch')
        require(s['image_sha256']==m['artifacts']['final.png']==c['image_sha256'],'Reuse hash identity mismatch')
        compatibility(c,row)
        if verify:
            require(sha(path/'metadata.json')==s['metadata_sha256'],'Suite-local reuse metadata changed')
            require(sha(path/'final.png')==s['image_sha256'],'Reused image changed')
            require(sha(c['original_metadata_path'])==c['original_metadata_sha256'],'Original metadata changed')
            for evidence in c['evidence']:require(sha(evidence['path'])==evidence['sha256'],'Original provenance evidence changed')
        return True
    except (ValueError,KeyError,OSError,TypeError):return False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--adopt',action='store_true');args=parser.parse_args()
    audit(adopt_reused=args.adopt)
if __name__=='__main__':main()
