"""CPU validation with stdout-only evidence and optional proven image links."""
import argparse
import gc
import os
import subprocess
import sys
from studies.all_prompts.common import *

def main(reuse):
    before=fingerprint();prompts=inventory_hash()
    emit('VALIDATION_START',fingerprint=before,inventory_sha256=prompts,job=os.environ.get('SLURM_JOB_ID'))
    subprocess.run([sys.executable,'-m','compileall','-q',str(STUDY)],check=True)
    subprocess.run(['git','diff','--check'],check=True)
    from studies.all_prompts.audit import audit,configuration
    candidates=audit()
    tests=[
        'studies.all_prompts.tests.test_sweep',
        'studies.tt_cea.tests.test_canvas',
        'studies.tt_cea.tests.test_pipeline.PipelineTests.test_full_noop_against_actual_frozen_loop',
        'test_gwtf_noise_initialization',
        'test_fusion_lpw.FusionTests.test_weighted_average_arithmetic',
        'test_fusion_lpw.FusionTests.test_spherediff_center_confidence_decreases_to_edge',
        'test_bridge_factorial.FactorialTests.test_bridge_identity_and_locality',
        'test_current_state_transition']
    test_env=dict(os.environ);test_env['PYTHONPATH']=str(REPO/'tests')+':'+str(REPO)+':'+test_env.get('PYTHONPATH','')
    subprocess.run([sys.executable,'-m','unittest','-v']+tests,check=True,env=test_env)
    from diffpano.gwtf_noise_initialization import initialize_gwtf_shared_noise,GWTFlowNoiseConfig
    from diffpano.erp_noise_initialization import native_cameras,primary_noise_size,states_digest
    import torch
    coverage=read(TT/'geometry/coverage.json')
    assert coverage['passed'] and read(TT/'geometry/operator_tests.json')['passed']
    for n in BACKENDS:
        c,stub,cams,ids,old=configuration(n)
        h,w=primary_noise_size(native_cameras(stub,cams));seen={}
        for projection in ('erp','cea'):
            states,record=initialize_gwtf_shared_noise(stub,cams,GWTFlowNoiseConfig(h,w,0),1,camera_slots=ids)
            current=states_digest(states)
            assert current==record['initial_local_sha256']==old['initialization']['initial_local_sha256']
            assert record['denoiser_calls']==record['vae_calls']==0 and record['scaling_applications']==1
            assert all(s.dtype==torch.float32 and s.device.type=='cpu' and bool(torch.isfinite(s).all()) for s in states)
            seen[projection]=current
            emit('INITIALIZATION_CPU_VERIFIED',backend=n,projection=projection,initial_state_sha256=current,
                denoiser_calls=0,vae_calls=0,noise_grid=[h,w],scaling_applications=1,camera_ids=ids)
            del states,record;gc.collect()
        assert seen['erp']==seen['cea']
        for projection in ('erp','cea'):
            proof=coverage['covers']['old89']['rasters']['1024'][projection]
            assert proof['samples']==2048*4096 and proof['uncovered_fraction']==0 and proof['finite_positive_denominators']
        emit('FULL_RESOLUTION_COVERAGE_REUSED',backend=n,old89=True,coverage_artifact=str(TT/'geometry/coverage.json'),
            artifact_sha256=sha(TT/'geometry/coverage.json'),samples_per_projection=2048*4096)
    # Verify imports/schedulers in the proven original environment without weights.
    output=shell_command(['bash',str(STUDY/'environment.sh'),'spherediff','-m','studies.all_prompts.audit','--original-import-check'])
    print(output,flush=True)
    pilots={(x['row']['method'],x['row']['backend'],x['row']['projection']) for x in candidates}
    required={(x['method'],x['backend'],x['projection']) for x in rows()}
    assert pilots==required,('Missing exact prior pilot',required-pilots)
    assert fingerprint()==before and inventory_hash()==prompts
    if reuse:
        for candidate in candidates:link_reuse(candidate['source'],candidate['row']['output'])
    emit('PILOTS_VERIFIED',methods=len(pilots),reused_images=len(candidates),extra_pilot_images=0)
    emit('SWEEP_VALIDATION_PASSED',fingerprint=before,inventory_sha256=prompts)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reuse',action='store_true');a=p.parse_args();main(a.reuse)
