"""Audit all 20 FLUX scene controls, reuse the pilot, and freeze the 17-prompt extension."""
import os,shutil,subprocess
from studies.gradient_blending.common import *

def main():
    assert os.environ.get('SLURM_JOB_ID') and BACKEND=='flux' and SUITE=='flux-scenes20'
    preserved();preserve_prior_results();sources=source_hashes()
    for name in ('validation.json','gpu-smoke.json','offline/replay.json'):
        gate=read(GATES/name);assert gate['passed'] and gate['core_hashes']==core_hashes()
    assert read(GATES/'offline/replay.json')['pilot_lambda_color']==.1
    assert read(BASE_OUT/'extension-completion.json')['passed']
    from studies.all_prompts.audit import configuration
    from studies.all_prompts.common import inventory
    from studies.gwtf_erp4k.common import differences
    inventory();baseline=baseline_audit();c,_,cameras,_,old=configuration('flux')
    assert c.generation.num_inference_steps==20 and len(cameras)==89
    assert len(PROMPTS)==20 and len(NEW_FLUX_PROMPTS)==17 and len(scene_rows())==34
    records=[]
    for prompt in PROMPTS:
        r=baseline['cached_cases'][prompt]
        assert r['accounting']['state']=='COMPLETED' and r['accounting']['exit_code']=='0:0'
        expected=configuration('flux','prompts/'+prompt+'.txt')[0].to_dict()
        delta=differences(safe(expected),r['configuration'])
        assert {v['path'] for v in delta}<={'prompt.path'},(prompt,delta)
        assert r['schedule_sha256']==baseline['schedule_sha256']
        assert r['seed']==0 and r['camera_count']==89
        initial=r['initialization'].get('record',r['initialization'])
        assert initial['initial_local_sha256']==baseline['initialization']['initial_local_sha256']
        if prompt in NEW_FLUX_PROMPTS:
            assert r['efficiency']['counts']==expected_counts('flux',20)
            evidence=[Path(p) for p in r['source_provenance'] if p.endswith('.out')]
            events=[]
            for path in evidence:
                for line in path.read_text().splitlines():
                    if line.startswith('RUNTIME_PROVENANCE '):
                        event=json.loads(line.split(' ',1)[1])
                        if event['row']['backend']=='flux' and event['row']['prompt']==prompt and event['row']['projection']=='erp':events.append(event)
            assert len(events)==1,(prompt,'missing/ambiguous runtime provenance')
            event=events[0];provenance=event['provenance']
            for key in ('model_checkpoint','model_revision','bridge_mode','camera_sha256','camera_geometry_sha256','prepared_schedule','prompt_indices'):
                assert safe(provenance[key])==safe(old[key]),(prompt,key)
            assert not event['time_travel_enabled']
            records.append(dict(prompt=prompt,image_sha256=r['image_sha256'],conditioning_sha256=provenance['conditioning_sha256'],
                original_sha256=r['original_prompt_sha256'],effective_sha256=r['effective_prompt_sha256'],
                evidence_sha256={str(p):sha(p) for p in evidence}))
    # The pilot cases are immutable directory links; only the 17 additional prompts are generated.
    for prompt in PILOT_PROMPTS:
        source=BASE_OUT/'flux/cases'/prompt;target=OUT/'cases'/prompt
        for mode in MODES:
            folder=source/mode;status=read(folder/'status.json');m=read(folder/'metadata.json')
            assert status['state']=='complete' and sha(folder/'final.png')==m['artifacts']['final.png']
            assert m['counts']==expected_counts('flux',20)
            assert m['camera_geometry_sha256']==baseline['camera_geometry_sha256']
            assert m['schedule_sha256']==baseline['schedule_sha256']
            assert m['initialization']['initial_local_sha256']==baseline['initialization']['initial_local_sha256']
            if mode!='rgb':
                assert len(m['fusions'])==21 and all(v['converged'] for v in m['fusions'])
                for key,value in core_hashes().items():
                    if key in m['source_hashes']:assert m['source_hashes'][key]==value
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.resolve()==source.resolve()
        else:target.symlink_to(source.resolve(),target_is_directory=True)
    from studies.gradient_blending.reuse import main as reuse
    reuse()
    # Copy validated feature caches, never write through links into historical evaluator caches.
    target=OUT/'evaluation';target.mkdir(exist_ok=True)
    for prompt in PILOT_PROMPTS:
        for mode in MODES:
            for suffix in ('npz','json'):
                source=BASE_OUT/'flux/evaluation'/(prompt+'-'+mode+'-features.'+suffix)
                destination=target/source.name
                if not destination.exists():shutil.copyfile(source,destination)
                assert sha(destination)==sha(source)
    # Collection-score regressions use the same frozen environment as production evaluation.
    # Strip an inherited active venv before the established evaluator wrapper purges modules.
    environment=dict(os.environ);venv=environment.pop('VIRTUAL_ENV',None)
    environment.pop('VIRTUAL_ENV_PROMPT',None)
    if venv:environment['PATH']=':'.join(v for v in environment['PATH'].split(':') if v.rstrip('/')!=str(Path(venv)/'bin'))
    result=subprocess.run(['bash','studies/panorama_metrics/environment.sh','-m','unittest','-v',
        'studies.gradient_blending.scene_checks.SceneChecks'],env=environment,universal_newlines=True,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (OUT/'validation-tests.log').write_text(result.stdout);print(result.stdout,flush=True)
    assert result.returncode==0 and 'Ran 3 tests' in result.stdout,'Scene checks failed in frozen evaluator environment'
    preserved();preserve_prior_results();assert sources==source_hashes()
    write(OUT/'manifest.json',dict(backend='flux',suite=SUITE,prompts=PROMPTS,new_prompts=NEW_FLUX_PROMPTS,
        pilot_prompts=PILOT_PROMPTS,excluded_prompt='native_control',exclusion_reason='17 remaining scene prompts requested; native_control is the additional control file',
        rows=scene_rows(),logical_outputs=60,new_gradient_runs=34,new_reused_rgb_controls=17,
        reused_pilot_outputs=9,lambda_color=.1,seed=0,steps=20,camera_count=89,fov_x=80.,fov_y=80.,
        local_RGB_resolution=[1024,1024],ERP_resolution=[2048,4096],camera_geometry_sha256=baseline['camera_geometry_sha256']))
    write(OUT/'validation.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],tests=3,
        prompts=PROMPTS,new_prompts=NEW_FLUX_PROMPTS,core_hashes=core_hashes(),source_hashes=sources,
        baseline_sha256=sha(OUT/'baseline.json'),manifest_sha256=sha(OUT/'manifest.json'),cached_sources=records,
        previous_results_preserved=True,git_head=subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip(),
        checks='Scope and array uniqueness; unchanged original pilot metric scores; dynamic matched collection sizes; exact original benchmark config/checkpoint/camera/schedule/initialization/prompt/call-count/image hashes; shared unchanged numerical gates'))
    print('FLUX_SCENE_VALIDATION_PASSED',len(NEW_FLUX_PROMPTS),len(scene_rows()),flush=True)
if __name__=='__main__':main()
