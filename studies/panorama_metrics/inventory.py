"""Image completion is established from bytes, configuration, stdout and accounting."""
import collections
import concurrent.futures
import subprocess
from PIL import Image
from .common import *
from studies.all_prompts.common import rows as benchmark_rows, inventory as prompts, fingerprint


def accounting():
    result={}
    out=subprocess.check_output(['sacct','--array','-u','shig','-S','2026-09-01','-X','-n','-P',
        '--format=JobID%50,JobIDRaw%50,State,ExitCode'],text=True)
    for line in out.splitlines():
        c=line.split('|')
        if len(c)>=4:
            for job in c[:2]:result[job.strip()]=dict(state=c[2],exit_code=c[3])
    return result

def log_records():
    by_output={};reuse={}
    wanted={'WORKER_START','PROMPT','RUNTIME_PROVENANCE','ORIGINAL_ARGUMENTS','INITIALIZATION','GENERATION_VALIDATED','SUCCESS_CASE'}
    for path in sorted((REPO/'logs').glob('allp21-run.*.out')):
        events={}
        for line in path.read_text(errors='replace').splitlines():
            name,sep,payload=line.partition(' ')
            if sep and name in wanted:
                try:events[name]=json.loads(payload)
                except ValueError:pass
        if 'SUCCESS_CASE' in events:
            row=events['SUCCESS_CASE']['row'];events['log_path']=str(path)
            events['job']=path.name[len('allp21-run.'):-len('.out')]
            by_output[row['output']]=events
    for path in sorted((REPO/'logs').glob('allp21-validate.*.out')):
        lines=path.read_text(errors='replace').splitlines()
        if not any(l.startswith('SWEEP_VALIDATION_PASSED ') for l in lines):continue
        for line in lines:
            if line.startswith('REUSE_APPROVED '):
                r=json.loads(line.split(' ',1)[1]);r['validation_log']=str(path);reuse[r['row']['output']]=r
    return by_output,reuse

def camera_rows():
    result=[]
    for strategy in ('fibonacci_n89','random_n89_seed0'):
        for backend,steps in [('sana',20),('flux',20),('sd35',40),('pixeldit',50)]:
            for projection in ('erp','cea'):
                for prompt in ('ruins','underwater','firework'):
                    folder=REPO/'outputs/camera-patching-seed0'/strategy/backend/projection/prompt
                    result.append(dict(method='diffpano',backend=backend,steps=steps,projection=projection,prompt=prompt,
                        camera_strategy=strategy,output=str(folder/'final.png'),config=str(folder/'config.json')))
    return result

def scan():
    from studies.flux_step_controls.common import rows as control_rows,complete as control_complete
    accounting_records=accounting();events,reuse=log_records();prompt_map={p['name']:p for p in prompts()}
    protected=fingerprint();cases=[]
    cancellation_path=REPO/'outputs/flux-step-controls-seed0/cancellation-report.json'
    cancellations=read(cancellation_path).get('groups',{}) if cancellation_path.exists() else {}
    for r in benchmark_rows():cases.append(('benchmark',r))
    for r in camera_rows():cases.append(('camera',r))
    for r in control_rows():cases.append(('step_control',r))
    assert len(cases)==300
    def inspect(case):
        family,r=case;p=Path(r['output']);prompt=prompt_map[r['prompt']]
        strategy=r.get('camera_strategy','old89' if r['method']=='diffpano' else 'official_spherediff')
        memberships=[family]
        if family=='benchmark' and r['method']=='diffpano' and r['prompt'] in ('ruins','underwater','firework'):memberships.append('camera_old89')
        rec=dict(id=family+'/'+r['method']+'/'+r['backend']+'/'+r['projection']+'/'+str(r['steps'])+'/'+strategy+'/'+r['prompt'],
            family=family,memberships=memberships,method=r['method'],backend=r['backend'],steps=r['steps'],seed=0,
            consensus_projection=r['projection'],final_projection='erp',prompt_id=r['prompt'],prompt=prompt,
            original_prompt_sha256=prompt['original_sha256'],effective_prompt_sha256=prompt['effective_sha256'],
            directional_texts=prompt['effective_lines'],camera_strategy=strategy,camera_count=89,
            layout_seed=0 if strategy=='random_n89_seed0' else None,fibonacci_phase=0. if strategy=='fibonacci_n89' else None,
            output_path=str(p),resolved_path=str(p.resolve()),completion_status='missing',evaluation_status='pending',
            model_verification_status='unresolved',provenance_status='unresolved')
        if not p.exists():
            if family=='step_control':
                status=REPO/'outputs/flux-step-controls-seed0/status'/(str(r['index'])+'.json')
                if status.exists():rec['completion_status']=read(status)['state']
                if r['prompt'] in cancellations.get(r['method'],{}).get('cancelled',[]):
                    rec.update(completion_status='cancelled',evaluation_status='cancelled',exclusion_reason='Generation cancelled at user request; not resubmitted by evaluation.')
            return rec
        try:
            with Image.open(p) as im:
                assert im.format=='PNG' and im.size==(4096,2048) and im.mode=='RGB';im.verify()
            with Image.open(p) as im:im.load()
            h=sha(p);rec.update(image_sha256=h,width=4096,height=2048)
            if family=='benchmark':
                if p.is_symlink():
                    proof=reuse[str(p)];assert proof['row']==r and proof['image_sha256']==h
                    assert Path(proof['source']).resolve()==p.resolve()
                    assert proof['prompt_sha256']==prompt['original_sha256']
                    metadata=Path(proof['metadata']);assert sha(metadata)==proof['metadata_sha256']
                    m=read(metadata);job=str(proof['job']);rec['source_provenance']=[str(metadata),proof['validation_log']]
                    model=m.get('model_checkpoint',m.get('checkpoint',m.get('model_source')))
                    revision=m.get('model_revision',m.get('revision'))
                    if r['method']=='diffpano':
                        c=m.get('config',m.get('generation_configuration'));model=c['model']['id'] if r['backend']!='pixeldit' else model
                        revision=revision or c['model'].get('revision')
                        schedule=m['prepared_schedule'];camera=m.get('camera_geometry_sha256')
                    else:
                        from studies.original_spherediff.common import specs
                        spec=specs()[r['backend']];model=spec['model_source'];revision=spec['revision']
                        schedule=m.get('scheduler',m.get('prepared_schedule',{}));camera=None
                    rec['efficiency']={k:m.get(k) for k in ('runtime_seconds','generation_seconds','peak_allocated_gib','actual_transformer_forward_invocations','vae_calls','actual_model_input_shapes','model_input_shapes')}
                    rec['efficiency']['counts']=m.get('counts')
                    rec['configuration']=m.get('config',m.get('generation_configuration',m.get('explicit_call_arguments')))
                    rec['initialization']=m.get('initialization',{})
                    rec['local_native_resolution']=m.get('local_native_resolution')
                else:
                    e=events[str(p)];assert e['SUCCESS_CASE']['row']==r and e['SUCCESS_CASE']['image_sha256']==h
                    assert e['WORKER_START']['source_fingerprint']==protected
                    assert e['PROMPT']['original_sha256']==prompt['original_sha256'] and e['PROMPT']['effective_sha256']==prompt['effective_sha256']
                    job=e['job'];rec['source_provenance']=[e['log_path']];done=e['GENERATION_VALIDATED'];rec['efficiency']=done
                    if r['method']=='diffpano':
                        runtime=e['RUNTIME_PROVENANCE'];assert runtime['row']==r and not runtime['time_travel_enabled']
                        provenance=runtime['provenance'];model=provenance['model_checkpoint'];revision=provenance['model_revision']
                        schedule=provenance['prepared_schedule'];camera=provenance['camera_geometry_sha256']
                        assert done['active_projection_each_interval']==r['projection']
                        rec['configuration']=runtime['configuration'];rec['initialization']=e.get('INITIALIZATION',{})
                    else:
                        from studies.original_spherediff.common import specs
                        spec=specs()[r['backend']];model=spec['model_source'];revision=spec['revision']
                        assert e['ORIGINAL_ARGUMENTS']['explicit']==spec['call']
                        schedule=done['scheduler'];camera=None;rec['configuration']=e['ORIGINAL_ARGUMENTS']
            else:
                cp=Path(r['config']);assert cp.is_file(),'Final PNG without matching config'
                m=read(cp);assert m['output']['sha256']==h
                rec['source_provenance']=[str(cp)];rec['configuration']=m
                if family=='camera':
                    c=m['camera'];assert c['num_cameras']==89 and c['fov_x_deg']==c['fov_y_deg']==80
                    assert m['generation']['num_inference_steps']==r['steps'] and m['generation']['seed']==0
                    assert m['projection']['consensus_canvas']==r['projection'] and m['projection']['final_export']=='erp'
                    assert m['prompt']['sha256']==prompt['original_sha256']
                    assert not m['trajectory']['time_travel'] and m['trajectory']['counts']['denoiser']==89*r['steps']
                    expected='random' if strategy.startswith('random') else 'fibonacci';assert c['strategy']==expected
                    assert c['layout_seed']==rec['layout_seed'] and c['fibonacci_phase']==rec['fibonacci_phase']
                    assert m['source']['fingerprint']=='d4f9de4d97b29ba1fa72c8a8e33a63bd12ac74e5adb4c155dd816fdef43d5cf5'
                    model=m['backend']['model_id'];revision=m['backend']['model_revision'];job=m['execution']['job_id']
                    schedule=None;rec['schedule_sha256']=m['generation']['prepared_schedule_sha256'];camera=c['angular_geometry_sha256']
                    rec['efficiency']=dict(m['execution'],counts=m['trajectory']['counts']);rec['initialization']=m['initialization']
                else:
                    assert control_complete(r),'New final pair has not passed completion validation'
                    model=m['model']['id'];revision=m['model']['revision'];job=m['execution']['job'];schedule=m['schedule']
                    camera=m.get('camera',{}).get('angular_geometry_sha256');rec['efficiency']=m['execution'];rec['initialization']=m['initialization']
            if schedule:
                assert len(schedule['timesteps'])==r['steps']
                rec['schedule_sha256']=digest(schedule);rec['schedule']=schedule
            rec.update(model_id=model,model_revision=revision,model_verification_status='pinned_provenance; cross-method weight equivalence separately audited',
                camera_geometry_sha256=camera,job=str(job))
            account=accounting_records.get(str(job));rec['accounting']=account
            assert account and account['state']=='COMPLETED' and account['exit_code']=='0:0',('Scheduler success absent',job,account)
            rec.update(completion_status='complete',provenance_status='verified',evaluation_status='eligible')
        except Exception as e:
            rec.update(completion_status='excluded',provenance_status='unresolved',evaluation_status='excluded',exclusion_reason=str(e))
        return rec
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(inspect,cases))
    from .camera_counts import scan as camera_count_scan
    result.extend(camera_count_scan(accounting_records,prompt_map))
    assert len(result)==426 and len({r['id'] for r in result})==426
    hashes=collections.defaultdict(list)
    for r in result:
        if r.get('image_sha256'):hashes[r['image_sha256']].append(r['id'])
    for r in result:r['identical_image_memberships']=hashes.get(r.get('image_sha256'),[])
    atomic(ROOT/'inventory.json',dict(created=now(),expected=len(result),counts=dict(collections.Counter(r['completion_status'] for r in result)),
        source_fingerprint=protected,rows=result))
    print('INVENTORY',dict(collections.Counter((r['family'],r['completion_status']) for r in result)),flush=True)
    for r in result:
        if r['completion_status']=='excluded':print('EXCLUDED',r['id'],r.get('exclusion_reason'),flush=True)
    return result
if __name__=='__main__':scan()
