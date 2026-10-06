import subprocess
import sys
from .common import *

def main():
    m=manifest();assert_preserved(m);assert m['protected_sources']['historical']==HISTORICAL
    assert os.environ.get('SLURM_JOB_ID'),'CPU Slurm required'
    from studies.all_prompts.audit import verify_sources,reusable_diffpano,reusable_original
    from studies.all_prompts.common import rows as old_rows
    from studies.all_prompts.summary import accounting,logged_success,reused_success
    plan=verify_sources();attempts,_=accounting();baselines=[]
    for row in [r for r in old_rows() if r['backend']=='flux']:
        assert image_valid(row['output'],True)
        proof=None
        if row['prompt'] in ('ruins','underwater'):
            proof=(reusable_diffpano(row) if row['method']=='diffpano' else reusable_original(row,plan))
            assert reused_success(row)
        else:
            valid=[a for a in attempts.get(row['index'],[]) if a['state']=='COMPLETED' and a['exit_code']=='0:0' and logged_success(a['job'],row)]
            assert valid,('No successful matching provenance',row)
            proof=dict(job=valid[-1]['job'],log=str(REPO/'logs'/('allp21-run.'+valid[-1]['job']+'.out')))
        baselines.append(dict(row=row,sha256=sha(row['output']),proof=proof))
    assert len(baselines)==63
    env=dict(os.environ);env['PYTHONPATH']=str(REPO/'tests')+':'+str(REPO)+':'+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,'-m','unittest','-v','studies.flux_step_controls.tests.test_controls',
        'studies.tt_cea.tests.test_pipeline.PipelineTests.test_full_noop_against_actual_frozen_loop',
        'test_current_state_transition'],check=True,env=env)
    # Compare available pinned checkpoint/config provenance without changing either source.
    from studies.original_spherediff.common import snapshot
    from studies.flux_step_controls.runtime import diff_configuration
    c,*_=diff_configuration('prompts/ruins.txt')
    dp=Path('/scratch/user/shig/diffpano/hf_cache/hub')/('models--'+c.model.id.replace('/','--'))/'snapshots'/c.model.revision
    sources={}
    for name,folder in [('diffpano',dp),('spherediff',snapshot('flux'))]:
        records={}
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.safetensors','.bin'):
                records[str(p.relative_to(folder))]=dict(bytes=p.stat().st_size,content_address=p.resolve().name,
                    sha256=sha(p) if p.suffix=='.json' else None)
        sources[name]=dict(snapshot=str(folder),files=records)
    shared=set(sources['diffpano']['files'])&set(sources['spherediff']['files'])
    matching=[p for p in sorted(shared) if sources['diffpano']['files'][p]==sources['spherediff']['files'][p]]
    provenance=dict(sources=sources,matching_file_records=matching,
        status='pinned_sources_preserved; blob addresses compared; tensor equivalence not yet independently verified')
    atomic(ROOT/'checkpoint_comparison.json',provenance)
    assert_preserved(m)
    atomic(ROOT/'validation'/'cpu.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],
        manifest_sha256=sha(ROOT/'manifest.json'),study_sources=source_hashes(),baselines=baselines,
        tests='step controls, fresh schedules, narrow overrides, partial outputs, existing interval/bridge regressions'))
    emit('FLUX_CONTROL_VALIDATION_PASSED',job=os.environ['SLURM_JOB_ID'],baselines=len(baselines))

if __name__=='__main__':main()
