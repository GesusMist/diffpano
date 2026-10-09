"""Resolve the proven FLUX baseline; audit four original screened/select slots."""
from dataclasses import asdict,replace
import copy,shutil,subprocess
from studies.gradient_refinement.common import *


def conditioning_evidence(cached):
    for evidence in cached['source_provenance']:
        path=Path(evidence)
        if path.suffix=='.json':
            m=read(path)
            if 'conditioning_sha256' in m:return m['conditioning_sha256'],str(path),sha(path)
        elif path.suffix=='.out':
            for line in path.read_text().splitlines():
                if line.startswith('RUNTIME_PROVENANCE '):
                    m=json.loads(line.split(' ',1)[1]);row=m.get('row',{})
                    if row.get('prompt')==cached['prompt_id'] and row.get('backend')=='flux':
                        return m['provenance']['conditioning_sha256'],str(path),sha(path)
    raise ValueError('Missing matched conditioning evidence')


def audit(row,c,old,ids,prompt,expected_conditioning):
    path=ROOT/'outputs/10.7gradient-blending-flux/seed0-v1/cases'/row['prompt']/'poisson_select'
    status=read(path/'status.json');m=read(path/'metadata.json')
    checks={}
    def check(name,ok):
        checks[name]=bool(ok)
        if not ok:raise AssertionError(name)
    check('complete',status['state']=='complete')
    check('metadata_hash',sha(path/'metadata.json')==status['metadata_sha256'])
    check('image_hash',sha(path/'final.png')==status['image_sha256']==m['artifacts']['final.png'])
    for name,h in m['artifacts'].items():check('artifact_'+name,sha(path/name)==h)
    check('configuration',m['configuration']==c.to_dict())
    check('prompt',m['prompt']==prompt)
    check('conditioning',m['provenance']['conditioning_sha256']==expected_conditioning)
    for key in ('model_checkpoint','model_revision','camera_sha256','camera_geometry_sha256','prepared_schedule','bridge_mode','local_RGB_resolution','local_native_resolution','native_channels','native_arithmetic_dtype'):
        if key in old:check(key,m['provenance'][key]==old[key])
    check('camera_order',m['camera_ids']==ids==list(range(89)))
    check('steps',c.generation.num_inference_steps==20 and len(m['intervals'])==20 and len(m['fusions'])==21)
    check('screened_select',m['fusion']==dict(mode='poisson_select',lambda_color=.1,max_iterations=200,relative_tolerance=1e-5,absolute_tolerance=1e-7))
    check('converged',all(r['mode']=='poisson_select' and r['converged'] and r['lambda_color']==.1 and r['final_true_residual']<=r['stopping_threshold'] for r in m['fusions']))
    check('terminal',m['terminal_fusion_mode']=='poisson_select' and m['fusions'][-1]['terminal'] and not m['terminal_reference_is_independent_baseline'])
    check('counts',m['counts']==dict(denoiser=1780,encode=3560,decode=1869,initialize=0))
    check('refinement_disabled',m['configuration']['global_pipeline'].get('refinement',{}).get('last_fraction',0)==0 and all(r.get('phase')!='independent_refinement' for r in m['intervals']))
    check('initial_native',m['initialization']['initial_local_sha256']==old['initialization']['initial_local_sha256']==m['details']['initial_state_sha256'])
    check('initialization_calls',m['initialization_audit']['calls']==dict(denoiser=0,encode=0,decode=0,initialize=89))
    check('ordinary_schedule',not m['details']['time_travel_enabled'] and m['details']['replay_count']==0 and m['details']['additional_denoising_cycles']==0 and digest(m['provenance']['prepared_schedule'])==m['schedule_sha256'])
    return m,dict(passed=True,reused_from=str(path),resolved_from=str(path.resolve()),checks=checks,metadata_sha256=sha(path/'metadata.json'),image_sha256=sha(path/'final.png'))


def main():
    from studies.all_prompts.audit import configuration,audit_cache
    from studies.all_prompts.common import prompt_record
    from diffpano.gradient_fusion import GradientSettings
    from diffpano.refinement import RefinementConfig
    preserved();historical=read(ROOT/'outputs/10.7gradient-blending-flux/seed0-v1/baseline.json')
    sources=source_hashes();revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip()
    c,_,cams,ids,old=configuration('flux');audit_cache('flux',c)
    assert len(cams)==89 and all(cam.fov_x==cam.fov_y==80 for cam in cams)
    assert c.to_dict()==historical['config']
    all_rows=rows();audits=[]
    for row in all_rows:
        c,_,cams,ids,old=configuration('flux',str(ROOT/'prompts'/(row['prompt']+'.txt')))
        info,_=prompt_record(ROOT/'prompts'/(row['prompt']+'.txt'))
        expected,evidence,h=conditioning_evidence(historical['cached_cases'][row['prompt']])
        c.global_pipeline.refinement=RefinementConfig(row['last_fraction'])
        settings=GradientSettings(row['gradient_mode'],reference_mode=row['reference_mode'])
        semantic=dict(configuration=asdict(c),fusion=asdict(settings),prompt=info,conditioning_sha256=expected,
            camera_ids=ids,camera_geometry_sha256=old['camera_geometry_sha256'],camera_sha256=old['camera_sha256'],
            schedule=old['prepared_schedule'],initial_native_sha256=old['initialization']['initial_local_sha256'],
            initializer='GWTF',steps=20,backend='flux',seed=0)
        row.update(semantic=semantic,semantic_sha256=digest(semantic),implementation_sha256=digest(sources),conditioning_evidence=dict(path=evidence,sha256=h))
        destination=folder(row);destination.mkdir(parents=True,exist_ok=True)
        if row['expected_reuse']:
            try:
                m,record=audit(row,c,old,ids,info,expected)
            except (OSError,KeyError,AssertionError,ValueError) as error:
                record=dict(passed=False,prompt=row['prompt'],reason=str(error));row['reuse_rejected']=record
            else:
                row['reused_from']=record['reused_from'];m=copy.deepcopy(m)
                m.update(semantic_sha256=row['semantic_sha256'],case=row,reused=True,reused_from=record['reused_from'],reuse_audit=record,
                    historical_raw_artifact='unavailable: historical run retained PNG and raw statistics, no unclamped FP32 ERP',
                    reference_mode='screened',refinement=dict(last_fraction=0.,total_intervals=20,coupled_intervals=20,independent_intervals=0),
                    historical_source_revision=m['source_revision'])
                for name,value in m['artifacts'].items():
                    target=destination/name
                    if not target.exists():shutil.copyfile(Path(record['reused_from'])/name,target)
                    assert sha(target)==value
                write(destination/'metadata.json',m);write(destination/'config.json',semantic)
                write(destination/'status.json',dict(state='reused',metadata_sha256=sha(destination/'metadata.json'),image_sha256=m['artifacts']['final.png'],reused_from=record['reused_from']))
            audits.append(dict(prompt=row['prompt'],**record))
    assert len(all_rows)==len({r['key'] for r in all_rows})==len({r['semantic_sha256'] for r in all_rows})==64
    reused=sum('reused_from' in r for r in all_rows)
    manifest=dict(schema=1,rows=all_rows,logical_cases=64,audited_reuse=reused,new_generation=64-reused,source_hashes=sources,implementation_revision=revision,
        historical_baseline_sha256=sha(ROOT/'outputs/10.7gradient-blending-flux/seed0-v1/baseline.json'),
        camera_count=89,FOV_x=80,FOV_y=80,camera_geometry_sha256=old['camera_geometry_sha256'],
        local_RGB_resolution=[1024,1024],local_native_resolution=[128,128],ERP_resolution=[2048,4096],
        constrained_solver='projected PCG, reflected FFT preconditioner, 400 iteration bound; no objective ridge',
        coarse_setting_note='16x32 nonoverlapping blocks, eta=0.1: fixed initial ablation choice, not tuned optimum',max_concurrent_generation_gpus=2)
    write(OUT/'reuse-audit.json',dict(candidates=audits,accepted=reused,rejected=4-reused))
    write(OUT/'manifest.json',manifest)
    print('MANIFEST',64,'REUSED',reused,'MISSING',64-reused,flush=True)
if __name__=='__main__':main()
