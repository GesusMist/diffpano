"""Completion audit for all four explicitly requested backends, preserving the earlier report."""
from studies.gradient_blending.common import *

def main():
    preserved();lines=['SANA + FLUX + PIXELDIT + SD3.5: GRADIENT-DOMAIN BLENDING','',
        '36 matched logical cases: four backends, ruins/underwater/firework, rgb/poisson_mean/poisson_select.',
        'Original benchmark steps: SANA20, FLUX20, PixelDiT50, SD3.5 40. Old89, 80-degree FOV, seed0, local1024x1024, ERP4096x2048.',
        'The same validated operator and lambda0.1 are used everywhere. RGB controls are exact cached benchmark images, except the bit-identical instrumented SANA ruins rerun.',
        'PixelDiT operates directly on native RGB without VAE calls; SD3.5 preserves the local identity-residual VAE bridge.',
        'The original two-model report and completion record are retained. This extension adds 12 trajectories and six reused controls.',
        'Only three prompts per backend: descriptive results, no generalization or significance claim.','']
    statuses=[];all_cameras=set()
    for backend in BACKENDS:
        root=BASE_OUT if backend=='sana' else BASE_OUT/backend
        for prompt in PROMPTS:
            records=[]
            for mode in MODES:
                folder=root/'cases'/prompt/mode;status=read(folder/'status.json')
                assert status['state']=='complete',(backend,prompt,mode,status)
                m=read(folder/'metadata.json');assert sha(folder/'final.png')==m['artifacts']['final.png']
                steps=m['configuration']['generation']['num_inference_steps']
                assert steps==dict(sana=20,flux=20,pixeldit=50,sd35=40)[backend]
                assert m['counts']==expected_counts(backend,steps)
                if mode!='rgb':
                    assert len(m['fusions'])==steps+1 and m['fusions'][-1]['terminal']
                    assert all(r['mode']==mode and r['converged'] for r in m['fusions'])
                    assert m['fusion']['lambda_color']==.1
                records.append(m);all_cameras.add(m['camera_geometry_sha256'])
                statuses.append(dict(backend=backend,prompt=prompt,mode=mode,state='complete',job=status.get('job')))
            for field in ('camera_geometry_sha256','schedule_sha256'):
                assert len({m[field] for m in records})==1,(backend,prompt,field)
            assert len({m['initialization']['initial_local_sha256'] for m in records})==1
            lines.append('%s/%s: matched initialization/camera/schedule hashes; all interval/terminal fusions converged; expected calls.'%(backend,prompt))
        evaluation=read(root/'evaluation/summary.json')
        assert len(evaluation['rows'])==9 and len(evaluation['collections'])==3
        if backend in ('pixeldit','sd35'):
            assert all(max(r['absolute_difference_from_frozen'].values())<1e-4 for r in evaluation['rows'] if r['mode']=='rgb')
        for prompt in PROMPTS:
            for suffix in ('final-erp','fixed-views','detail-wrap'):
                assert (root/'figures'/(prompt+'-'+suffix+'.png')).is_file()
        lines+=['','='*72,backend.upper()+' REPORT','='*72,(root/'report.txt').read_text()]
    assert len(all_cameras)==1
    write(BASE_OUT/'extension-completion.json',dict(passed=True,cases=statuses,logical_case_count=36,source_preserved=True,
        camera_geometry_sha256=next(iter(all_cameras)),new_logical_cases=18,new_gradient_runs=12,reused_new_controls=6))
    (BASE_OUT/'four-models-report.txt').write_text('\n'.join(lines)+'\n')
    print('FOUR_MODEL_STUDY_COMPLETE',len(statuses),flush=True)
if __name__=='__main__':main()
