"""Audit a requested backend against original benchmark evidence and cached RGB controls."""
import os
from studies.gradient_blending.common import *

def main():
    assert BACKEND in ('pixeldit','sd35') and os.environ.get('SLURM_JOB_ID')
    preserved();gate=read(GATES/'validation.json');smoke=read(GATES/'gpu-smoke.json');replay=read(GATES/'offline/replay.json')
    for item in (gate,smoke,replay):assert item['passed'] and item['core_hashes']==core_hashes()
    assert replay['pilot_lambda_color']==.1
    baseline=baseline_audit();steps=baseline['config']['generation']['num_inference_steps']
    assert steps=={'pixeldit':50,'sd35':40}[BACKEND]
    from studies.all_prompts.audit import reusable_diffpano,configuration
    from studies.gwtf_erp4k.common import differences
    checks=[]
    for prompt in PROMPTS:
        cached=baseline['cached_cases'][prompt]
        config=configuration(BACKEND,'prompts/'+prompt+'.txt')[0].to_dict()
        delta=differences(safe(config),cached['configuration'])
        assert {d['path'] for d in delta}<= {'prompt.path'},delta
        assert cached['schedule_sha256']==baseline['schedule_sha256']
        assert cached['seed']==0 and cached['camera_count']==89
        if prompt in ('ruins','underwater'):
            proof=reusable_diffpano(dict(backend=BACKEND,prompt=prompt,projection='erp'))
            assert proof['image_sha256']==cached['image_sha256']
        else:
            evidence=Path(cached['source_provenance'][0]);records={}
            for line in evidence.read_text().splitlines():
                for label in ('RUNTIME_PROVENANCE','RESULT'):
                    if line.startswith(label+' '):records[label]=json.loads(line.split(' ',1)[1])
            provenance=records['RUNTIME_PROVENANCE']
            assert provenance['row']['backend']==BACKEND and provenance['row']['prompt']==prompt
            assert provenance['provenance']['camera_geometry_sha256']==baseline['camera_geometry_sha256']
            assert digest(provenance['provenance']['prepared_schedule'])==baseline['schedule_sha256']
            proof=dict(log=str(evidence),log_sha256=sha(evidence))
        checks.append(dict(prompt=prompt,proof=proof,configuration_differences=delta))
    from studies.gradient_blending.reuse import main as reuse
    reuse()
    write(OUT/'backend-validation.json',dict(passed=True,backend=BACKEND,job=os.environ['SLURM_JOB_ID'],
        steps=steps,expected_counts=expected_counts(BACKEND,steps),core_hashes=core_hashes(),
        source_hashes=source_hashes(),baseline_sha256=sha(OUT/'baseline.json'),cached_sources=checks,
        operator_validation_job=gate['job'],model_independent_gpu_smoke_job=smoke['job'],
        scope='Explicit user extension: original three prompts, Old89, ERP, seed0, unchanged backend schedule, three fusion modes',
        offline_replay='Shared validated SANA replay of the unchanged model-independent RGB operator.'))
    print('BACKEND_AUDIT_PASSED',BACKEND,steps,flush=True)
if __name__=='__main__':main()
