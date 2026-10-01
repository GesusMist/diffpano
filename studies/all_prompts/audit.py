"""Read-only scientific audit and narrowly proven final-image reuse."""
import argparse
from dataclasses import replace
from studies.all_prompts.common import *

def verify_sources():
    from studies.gwtf_noise.common import require_validation
    from studies.tt_cea.common import study_hashes
    from studies.tt_cea.metadata import compatible_source_hashes
    from studies.original_spherediff.common import require_gate as original_gate
    require_validation(statistical=True)
    gate=read(TT/'validation.json');assert gate['passed'] and gate['source_hashes']==study_hashes()
    assert gate['manifest_sha256']==sha(TT/'manifest.json')
    compatible_source_hashes(gate)
    exceptions=[]
    for name,value in read(TT/'preservation.json')['hashes'].items():
        path=REPO/name
        if not path.exists() and name in ('prompts/xfjord.txt','prompts/xvalley.txt'):
            exceptions.append(name);continue
        assert sha(path)==value,('Historical source/artifact changed',name)
    assert sorted(exceptions)==['prompts/xfjord.txt','prompts/xvalley.txt']
    plan=original_gate()
    assert read(ORIGINAL/'audit.json')['passed']
    assert read(ORIGINAL/'final_audit.json')['artifact_hashes']['completion.json']==sha(ORIGINAL/'completion.json')
    emit('HISTORICAL_PROVENANCE',generation_sources_unchanged=True,
        obsolete_prompt_deletions_preserved=exceptions,
        note='Only these pre-existing prompt inventory deletions are outside the new scoped preservation check; historical gates are unmodified.',
        original_source_commit=plan['source_commit'])
    return plan

def configuration(name,prompt_path='prompts/ruins.txt'):
    from studies.gwtf_erp4k.common import geometry,differences
    from studies.tt_cea.cameras import load_cover
    from diffpano.bridge_factorial import angular_geometry,digest as geometry_digest
    from diffpano.erp_local_consensus import camera_digest
    c,stub,old_cams,_,_=geometry(name)
    old=read(BASES['ruins']/name/'metadata.json')
    assert c.to_dict()==old['config']
    cams,ids=load_cover('old89',c.view.height,c.view.width)
    assert cams==old_cams and ids==list(range(89))
    assert geometry_digest(angular_geometry(cams))==old['camera_geometry_sha256']
    assert camera_digest(cams)==old['camera_sha256']
    assert len(cams)==89 and all(cam.fov_x==cam.fov_y==80 for cam in cams)
    assert c.generation.num_inference_steps==STEPS[name] and c.generation.batch_size==1 and c.experiment.seed==0
    assert c.view.height==c.view.width==1024 and (c.erp.height,c.erp.width)==(2048,4096)
    assert (c.fusion.mode,c.fusion.weight_mode,c.fusion.spherediff_temperature)==('weighted_average','spherediff_center',.1)
    assert c.warp.mode=='standard'
    assert (c.warp.perspective_to_erp.interpolation,c.warp.erp_to_perspective.interpolation)==('bilinear','nearest')
    assert not c.sampling.rotation.enabled and c.sampling.strategy=='spherediff_fixed'
    assert c.consensus_transition.mode=='preserve_current_state'
    assert c.consensus_transition.vae_residual_correction==(name!='pixeldit')
    expected={
        'sana':('Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers','e2b3c0cbffebcd09d83805e88b9f5f106afc74ac',20,4.5),
        'flux':('ModelsLab/flux.1-dev','fa45a9eb6808ba8fdfc7cc2756f7f1a16e0921f4',20,3.5),
        'sd35':('stabilityai/stable-diffusion-3.5-medium','b940f670f0eda2d07fbb75229e779da1ad11eb80',40,4.5)}
    if name in expected:
        actual=(c.model.id,c.model.revision,c.generation.num_inference_steps,c.generation.guidance_scale)
        if actual!=expected[name]:emit('VALIDATED_CONFIG_DISCREPANCY',backend=name,expected=expected[name],actual=actual,action='preserve proven configuration')
    if name=='flux':assert c.generation.true_cfg_scale==1.
    c=replace(c,prompt=replace(c.prompt,path=str(prompt_path)))
    delta=differences(old['config'],c.to_dict())
    assert {d['path'] for d in delta}<= {'prompt.path'}
    return c,stub,cams,ids,old

def audit_cache(name,c):
    if name=='pixeldit':
        p=Path(read(BASES['ruins']/name/'metadata.json')['model_checkpoint'])
        assert p.is_file();assert Path(c.pixeldit.config_path).is_file()
        commit=shell_command(['git','-C',str(REPO/c.pixeldit.repo_path),'rev-parse','HEAD'])
        assert commit==c.pixeldit.expected_commit
        emit('CHECKPOINT',backend=name,path=str(p),bytes=p.stat().st_size,commit=commit,cfg_scale=c.pixeldit.cfg_scale,
            interval_guidance=c.pixeldit.interval_guidance,flow_shift=c.pixeldit.flow_shift)
    else:
        p=Path('/scratch/user/shig/diffpano/hf_cache/hub')/('models--'+c.model.id.replace('/','--'))/'snapshots'/c.model.revision
        assert (p/'model_index.json').is_file()
        assert any(x.is_file() for x in p.rglob('*.safetensors'))
        assert not any(x.is_symlink() and not x.exists() for x in p.rglob('*'))
        emit('CHECKPOINT',backend=name,model=c.model.id,revision=c.model.revision,path=str(p))

def reusable_diffpano(row):
    from studies.tt_cea.metadata import compatible_source_hashes
    n,p,projection=row['backend'],row['prompt'],row['projection']
    if p not in BASES:return None
    c,_,cams,ids,ruins=configuration(n,'prompts/'+p+'.txt')
    baseline=read(BASES[p]/n/'metadata.json')
    assert baseline['config']==c.to_dict()
    prompt=prompt_record(REPO/c.prompt.path)[0]
    manifest=read(BASES[p]/'manifest.json')
    if p=='underwater':
        assert manifest['prompt_sha256']==baseline['prompt_sha256']==prompt['original_sha256']
        assert manifest['prompt_lines']==prompt['effective_lines']
    else:
        evidence=read(REPO/'outputs/bridge-factorial-ruins/20260918/manifest.json')['prompt']
        assert evidence['sha256']==prompt['original_sha256'] and evidence['lines']==prompt['effective_lines']
    expected=89*STEPS[n]
    assert baseline['initialization']['initial_local_sha256']==ruins['initialization']['initial_local_sha256']
    assert baseline['initialization']['scaling_applications']==1
    assert baseline['initialization']['denoiser_calls']==baseline['initialization']['vae_calls']==0
    assert baseline['method']=='gwtf'
    assert baseline['camera_geometry_sha256']==ruins['camera_geometry_sha256']
    assert baseline['prepared_schedule']==ruins['prepared_schedule']
    assert baseline['model_checkpoint']==ruins['model_checkpoint'] and baseline['model_revision']==ruins['model_revision']
    assert baseline['bridge_mode']==('not_applicable_identity' if n=='pixeldit' else 'local_identity_preserving')
    assert baseline['actual_transformer_forward_invocations']==expected
    assert baseline['vae_calls']==dict(encode=0 if n=='pixeldit' else 2*expected,decode=0 if n=='pixeldit' else expected+89)
    a=baseline['audit']
    assert a['initialization']=='gwtf' and a['warp']=='standard' and a['fusion']=='weighted_average'
    assert a['transition']=='preserve_current_state' and not a['fixed_noise_renoising'] and a['extra_denoiser_calls']==0
    for path,value in baseline['source_hashes'].items():assert sha(REPO/path)==value,path
    if projection=='erp':
        path=BASES[p]/n/'gwtf-final.png';metadata=BASES[p]/n/'metadata.json'
        job=str(baseline['environment']['job'])
    else:
        folder=TT/'samples'/p/n/'C0';metadata=folder/'metadata.json';m=read(metadata);status=read(folder/'status.json')
        assert status['state']=='complete' and status['metadata_sha256']==sha(metadata)
        assert m['source_hashes'] in compatible_source_hashes(read(TT/'validation.json'))
        assert m['generation_configuration']==c.to_dict() and m['active_clean_projection']=='cea'
        assert m['study_overlay']['cameras']=='old89' and m['study_overlay']['time_travel']=='off'
        assert m['study_overlay']['replay_intervals']==0 and m['study_overlay']['ordinary_steps']==STEPS[n]
        assert m['noise_projection']=='erp' and m['camera_geometry_sha256']==baseline['camera_geometry_sha256']
        assert m['initialization']['initial_local_sha256']==baseline['initialization']['initial_local_sha256']
        assert m['model_checkpoint']==baseline['model_checkpoint'] and m['model_revision']==baseline['model_revision']
        assert safe(m['prepared_schedule'])==safe(baseline['prepared_schedule'])
        assert m['bridge_mode']==baseline['bridge_mode'] and m['prompt_indices']==baseline['prompt_indices']
        assert m['conditioning_sha256']==baseline['conditioning_sha256']
        assert m['counts']==dict(denoiser=expected,encode=0 if n=='pixeldit' else 2*expected,decode=0 if n=='pixeldit' else expected+89,initialize=0)
        d=m['diagnostics'];assert d['backward_seconds']==0 and d['original_noise_sha256_before'] is None
        assert d['original_noise_sha256_after'] is None and d['replay_boundary_hashes']==[]
        assert all(x['pass_kind']=='initial' for x in d['diagnostics'])
        assert len(d['diagnostics'])==STEPS[n] and d['projection']=='cea' and d['residuals_local']
        assert 'CEA exports once' in d['terminal_semantics']
        path=folder/'final.png';assert sha(path)==m['artifacts']['final.png']==status['artifacts']['final.png']
        job=str(m['job'])
    assert image_valid(path,True)
    return dict(row=row,source=str(path),metadata=str(metadata),metadata_sha256=sha(metadata),
        image_sha256=sha(path),job=job,initial_state_sha256=baseline['initialization']['initial_local_sha256'],
        prompt_sha256=prompt['original_sha256'],camera_geometry_sha256=baseline['camera_geometry_sha256'],
        evidence='Exact configuration/source/schedule/checkpoint/prompt/old89/bridge/call-count audit; no replay; projection-specific terminal assembly')

def reusable_original(row,plan):
    n,p=row['backend'],row['prompt']
    if p not in ('ruins','underwater'):return None
    from studies.original_spherediff.common import specs
    reference=next(r for r in plan['rows'] if r['backend']==n and r['prompt']==p)
    finished=next(r for r in read(ORIGINAL/'completion.json')['rows'] if r['backend']==n and r['prompt']==p)
    assert finished['status']=='complete' and finished['seed']==0 and finished['steps']==row['steps']
    assert reference['spec']==specs()[n]
    prompt=prompt_record(REPO/'prompts'/(p+'.txt'))[0]
    assert prompt['original_sha256']==plan['prompts'][p]['sha256'] and prompt['effective_lines']==plan['prompts'][p]['lines']
    m=read(reference['metadata'])
    if reference['reuse']:
        assert sha(reference['metadata'])==reference['reused_metadata_sha256']
        assert reference['reuse_evidence']['matched'] and all(reference['reuse_evidence']['checks'].values())
    else:
        status=read(Path(reference['metadata']).with_name('status.json'))
        assert status['state']=='complete' and sha(reference['metadata'])==status['metadata_sha256']
        assert m['actual_import_paths']==plan['import_paths'] and m['official_source_hashes']==plan['official_source_hashes']
        assert m['explicit_call_arguments']==reference['spec']['call']
        assert m['checkpoint']==reference['spec']['model_source'] and m['revision']==reference['spec']['revision']
        assert m['seed']==0 and m['generator_device'].split(':')[0]=='cuda' and m['precision']=='torch.bfloat16'
        assert not m['model_cpu_offload'] and not m['vae_tiling']
        assert m['scheduler']['effective_numerical_order']==1
    assert image_valid(reference['output'],True) and sha(reference['output'])==finished['image_sha256']
    return dict(row=row,source=reference['output'],metadata=reference['metadata'],metadata_sha256=sha(reference['metadata']),
        image_sha256=finished['image_sha256'],job=str(finished['job']),prompt_sha256=prompt['original_sha256'],
        evidence='Previously audited exact original source/revision/settings/RNG/effective-scheduler reference; final image/hash reverified')

def audit():
    inventory(verbose=True);matrix=rows();plan=verify_sources()
    for n in BACKENDS:
        c,_,_,_,_=configuration(n);audit_cache(n,c)
        emit('DIFFPANO_CONFIG',backend=n,configuration=c.to_dict(),time_travel_enabled=False,projection_methods=['erp','cea'])
    candidates=[]
    for row in matrix:
        if row['prompt'] not in ('ruins','underwater'):continue
        try:
            candidate=reusable_diffpano(row) if row['method']=='diffpano' else reusable_original(row,plan)
            if candidate:candidates.append(candidate)
        except (AssertionError,KeyError,ValueError,OSError) as error:
            emit('REUSE_REJECTED',row=row,error=repr(error))
    jobs=sorted(set(x['job'] for x in candidates))
    accounting=shell_command(['sacct','-j',','.join(jobs),'-n','-P','--format=JobIDRaw,State,ExitCode']) if jobs else ''
    successful={line.split('|')[0] for line in accounting.splitlines() if line.split('|')[1:3]==['COMPLETED','0:0']}
    approved=[]
    for c in candidates:
        if c['job'] not in successful:emit('REUSE_REJECTED',row=c['row'],reason='Successful historical worker exit unavailable',job=c['job']);continue
        approved.append(c);emit('REUSE_APPROVED',**c)
    emit('REUSE_AUDIT',approved=len(approved),required=210,old_worker_accounting=accounting)
    return approved

def original_import_check():
    from studies.original_spherediff.common import import_official,official_hashes,COMMIT,GIT_SOURCE,SPHERE,snapshot
    from studies.original_spherediff.audit import prepared_scheduler
    plan=verify_sources();classes,paths=import_official()
    assert paths==plan['import_paths']
    for path,value in official_hashes().items():
        content=subprocess.check_output(['git','-C',str(GIT_SOURCE),'show',COMMIT+':'+path])
        assert hashlib.sha256(content).hexdigest()==value
    for n in ('sana','flux'):
        assert snapshot(n).is_dir()
        expected=next(r['expected_scheduler'] for r in plan['rows'] if r['backend']==n)
        now=prepared_scheduler(n)
        for k in ('class_name','config','timesteps','sigmas','effective_numerical_order'):assert now[k]==expected[k],k
        emit('ORIGINAL_SCHEDULER',backend=n,**now)
    emit('ORIGINAL_IMPORT_VERIFIED',source=str(SPHERE),commit=COMMIT,paths=paths)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original-import-check',action='store_true');args=p.parse_args()
    original_import_check() if args.original_import_check else audit()
