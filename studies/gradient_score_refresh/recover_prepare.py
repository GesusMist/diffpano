"""Prepare exact missing cases using existing validated solver implementations."""
import copy,shutil
from dataclasses import asdict
from studies.gradient_score_refresh.recovery_common import *

def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Compute environment required'
    from diffpano.gradient_fusion import GradientSettings
    from diffpano.refinement import RefinementConfig
    from studies.all_prompts.audit import configuration
    from studies.all_prompts.common import prompt_record
    from studies.gradient_refinement.prepare import conditioning_evidence
    assert not (RECOVERY/'plan.json').exists(), 'Recovery already prepared'
    m=read(REF_OUT/'manifest.json');sources=recovery_sources()
    assert m['source_hashes']==original_source_hashes()
    for name in ('validation.json','gpu-smoke.json'):
        gate=read(REF_OUT/name);assert gate['passed'] and gate['source_hashes']==m['source_hashes']
    prior=read(REF_OUT/'retries/rtol-1e-4/validation.json')
    assert prior['passed']
    for path,value in prior['source_hashes'].items():assert sha(ROOT/path)==value
    firework=copy.deepcopy(m['rows'][60]);assert firework['prompt']=='firework' and firework['reference_mode']=='coarse_color'
    firework['semantic']['fusion']['relative_tolerance']=1e-4
    firework['semantic_sha256']=digest(firework['semantic']);firework['implementation_sha256']=digest(sources)
    firework['recovery_output']=str(REF_OUT/'cases'/firework['key'])
    firework['recovery_kind']='refinement'
    b=read(BLEND_OUT/'baseline.json');prompt='clouds_over_grass'
    c,_,cameras,ids,old=configuration('flux',str(ROOT/'prompts'/(prompt+'.txt')))
    c.global_pipeline.refinement=RefinementConfig(0.)
    info,_=prompt_record(ROOT/'prompts'/(prompt+'.txt'))
    expected,evidence,h=conditioning_evidence(b['cached_cases'][prompt])
    peer=read(BLEND_OUT/'cases'/prompt/'poisson_select/metadata.json')
    assert peer['provenance']['conditioning_sha256']==expected
    assert peer['prompt']==info
    assert sha(BLEND_OUT/'cases'/prompt/'poisson_select/final.png')==peer['artifacts']['final.png']
    settings=GradientSettings('poisson_mean',reference_mode='screened')
    semantic=dict(configuration=asdict(c),fusion=asdict(settings),prompt=info,conditioning_sha256=expected,
        camera_ids=ids,camera_geometry_sha256=old['camera_geometry_sha256'],camera_sha256=old['camera_sha256'],
        schedule=old['prepared_schedule'],initial_native_sha256=old['initialization']['initial_local_sha256'],
        initializer='GWTF',steps=20,backend='flux',seed=0)
    clouds=dict(index=8,key=prompt+'/poisson_mean',backend='flux',seed=0,prompt=prompt,reference_mode='screened',gradient_mode='poisson_mean',
        last_fraction=0.,semantic=semantic,semantic_sha256=digest(semantic),implementation_sha256=digest(sources),
        conditioning_evidence=dict(path=evidence,sha256=h),recovery_output=str(BLEND_OUT/'cases'/prompt/'poisson_mean'),recovery_kind='blending')
    rows=[firework,clouds];RECOVERY.mkdir(parents=True,exist_ok=True)
    shutil.copy2(str(REF_OUT/'manifest.json'),str(RECOVERY/'original-refinement-manifest.json'))
    attempts=[]
    for row in rows:
        path=folder(row);status=read(path/'status.json');assert status['state']=='failed'
        archive=RECOVERY/'failed-attempts'/row['recovery_kind']/row['key'];archive.mkdir(parents=True,exist_ok=True)
        files={}
        for p in path.iterdir():
            if p.is_file() and p.name!='lock':
                q=archive/p.name;shutil.copy2(str(p),str(q));files[str(q)]=sha(q);assert sha(p)==files[str(q)]
        attempts.append(dict(kind=row['recovery_kind'],key=row['key'],previous_job=status['job'],archive_hashes=files,previous_failure=status.get('solver'),previous_error=status['error']))
    check=copy.deepcopy(firework)
    for k in ('recovery_output','recovery_kind'):check.pop(k)
    check['semantic']['fusion']['relative_tolerance']=1e-5
    check['semantic_sha256']=m['rows'][60]['semantic_sha256'];check['implementation_sha256']=m['rows'][60]['implementation_sha256']
    assert check==m['rows'][60], 'Firework retry changes more than tolerance'
    diag=attempts[0]['previous_failure'];assert diag['output_normal_residual']<=1e-7+1e-4*diag['initial_true_residual']
    plan=dict(rows=rows,attempts=attempts,source_hashes=sources,implementation_revision=m['implementation_revision'],
        passed=True,job=os.environ['SLURM_JOB_ID'],prior_tolerance_validation=str(REF_OUT/'retries/rtol-1e-4/validation.json'),
        note='Recover exactly two missing outputs for requested complete evaluation. Firework uses the already-validated rtol1e-4. Clouds retains original screened tolerance and exact matched-conditioning check.')
    write(RECOVERY/'plan.json',plan)
    for path,value in sources.items():assert sha(ROOT/path)==value
    updated=copy.deepcopy(m)
    replacement=copy.deepcopy(firework)
    for k in ('recovery_output','recovery_kind'):replacement.pop(k)
    updated['rows'][60]=replacement
    updated['additional_tolerance_retries']=dict(indices=[60],relative_tolerance=1e-4,absolute_tolerance=1e-7,plan=str(RECOVERY/'plan.json'))
    assert all(a==b for i,(a,b) in enumerate(zip(m['rows'],updated['rows'])) if i!=60)
    write(REF_OUT/'manifest.json',updated)
    print('RECOVERY_READY',[(r['recovery_kind'],r['key']) for r in rows],flush=True)

if __name__=='__main__':main()
