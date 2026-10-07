"""Materialize audited cached RGB controls without changing historical outputs."""
from pathlib import Path
import shutil
from studies.gradient_blending.common import *

def main():
    preserved();baseline=read(OUT/'baseline.json')
    for prompt in (('underwater','firework') if BACKEND=='sana' else PROMPTS):
        r=baseline['cached_cases'][prompt];folder=OUT/'cases'/prompt/'rgb';folder.mkdir(parents=True,exist_ok=True)
        target=folder/'final.png';source=Path(r['output_path']);assert sha(source)==r['image_sha256']
        if (folder/'status.json').exists() and read(folder/'status.json')['state']=='complete':continue
        for evidence in r['source_provenance']:assert Path(evidence).is_file(),evidence
        initialization=r['initialization'];initialization=initialization.get('record',initialization)
        assert initialization['initial_local_sha256']==baseline['initialization']['initial_local_sha256']
        assert r['schedule_sha256']==baseline['schedule_sha256'],'Cached schedule differs'
        eff=r['efficiency'];counts=eff.get('counts')
        if counts is None:
            counts=dict(denoiser=eff['actual_transformer_forward_invocations'],encode=eff['vae_calls']['encode'],
                        decode=eff['vae_calls']['decode'],initialize=0)
        assert counts==expected_counts(BACKEND,baseline['config']['generation']['num_inference_steps'],len(baseline['cameras']))
        provenance_checks={p:sha(p) for p in r['source_provenance']}
        if target.exists():assert sha(target)==r['image_sha256']
        else:shutil.copyfile(source,target)
        metadata=dict(backend=BACKEND,reused=True,prompt=prompt_record(ROOT/'prompts'/(prompt+'.txt'))[0],fusion=dict(mode='rgb'),terminal_fusion_mode='rgb',
            initialization=initialization,camera_geometry_sha256=r['camera_geometry_sha256'],schedule_sha256=r['schedule_sha256'],
            configuration=r['configuration'],cached_case=r,counts=counts,generation_seconds=eff.get('generation_seconds'),
            historical_runtime_seconds=eff.get('runtime_seconds'),peak_allocated_gib=eff.get('peak_allocated_gib'),solver_seconds=0.,
            timing_note='Historical generation-only time when explicitly recorded; broader runtime is not relabeled as generation-only.',
            artifacts={'final.png':r['image_sha256']},source_provenance_sha256=provenance_checks,job=r['job'],
            source_revision=baseline['source_revision'],reuse_evidence='Previously frozen evaluator inventory plus original source, exact image/prompt/Old89/schedule/GWTF/bridge/config/call-count verification')
        write(folder/'metadata.json',metadata);write(folder/'config.json',dict(configuration=r['configuration'],fusion=dict(mode='rgb'),reused=True))
        write(folder/'status.json',dict(state='complete',reused=True,image_sha256=sha(target),metadata_sha256=sha(folder/'metadata.json'),job=r['job']))
        print('REUSED',prompt,sha(target),flush=True)
if __name__=='__main__':main()
