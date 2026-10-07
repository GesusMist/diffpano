"""Audit the explicitly requested FLUX extension; reuse model-neutral numerical gates."""
import os
from studies.gradient_blending.common import *

def main():
    assert BACKEND=='flux' and os.environ.get('SLURM_JOB_ID')
    preserved();gate=read(GATES/'validation.json');smoke=read(GATES/'gpu-smoke.json')
    assert gate['passed'] and gate['core_hashes']==core_hashes()
    assert smoke['passed'] and smoke['core_hashes']==core_hashes()
    baseline=baseline_audit();assert baseline['config']['generation']['num_inference_steps']==20
    # Check the original source hashes for the two older cached RGB controls.
    checks=[]
    for prompt in ('ruins','underwater'):
        name='gwtf-erp4k' if prompt=='ruins' else 'gwtf-underwater'
        metadata=read(ROOT/'outputs'/name/'20260923/flux/metadata.json')
        for path,expected in metadata['source_hashes'].items():assert sha(ROOT/path)==expected,path
        checks.append(dict(prompt=prompt,original_source_files_checked=len(metadata['source_hashes'])))
    from studies.gradient_blending.reuse import main as reuse
    reuse()
    write(OUT/'backend-validation.json',dict(passed=True,backend='flux',job=os.environ['SLURM_JOB_ID'],
        operator_validation_job=gate['job'],model_independent_gpu_smoke_job=smoke['job'],
        core_hashes=core_hashes(),baseline_sha256=sha(OUT/'baseline.json'),cached_sources=checks,
        scope='User-requested extension: same three prompts, Old89, ERP, seed0, 20 benchmark FLUX steps, three fusion modes',
        offline_replay='The required SANA replay tests the model-independent RGB operator; no FLUX diagnostic baseline is regenerated.'))
    print('FLUX_BACKEND_AUDIT_PASSED',flush=True)
if __name__=='__main__':main()
