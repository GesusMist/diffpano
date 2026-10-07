"""Matched two-model completion audit and combined report."""
from studies.gradient_blending.common import *

def main():
    preserved();lines=['SANA + FLUX: GRADIENT-DOMAIN BLENDING PILOT','',
        'Scope: two backends, exact ruins/underwater/firework prompts, rgb/poisson_mean/poisson_select; 18 matched logical outputs.',
        'Both backends use their unchanged benchmark configuration, seed0, Old89, ERP4096x2048, local1024x1024 and20 steps.',
        'Five validated RGB controls are reused; SANA ruins is instrumented once and matches its historical PNG bit-for-bit.',
        'The operator and offline lambda study are shared; offline proposals come from the required SANA baseline.',
        'No per-model/per-prompt lambda tuning. The combined generation array permits at most two GPUs.',
        '', 'STATUS AND MATCHED INPUT AUDIT']
    statuses=[]
    for backend,root in [('sana',BASE_OUT),('flux',BASE_OUT/'flux')]:
        for prompt in PROMPTS:
            records=[]
            for mode in MODES:
                folder=root/'cases'/prompt/mode;status=read(folder/'status.json')
                statuses.append(dict(backend=backend,prompt=prompt,mode=mode,state=status['state'],job=status.get('job')))
                assert status['state']=='complete',(backend,prompt,mode,status)
                m=read(folder/'metadata.json');assert sha(folder/'final.png')==m['artifacts']['final.png']
                records.append(m)
            initial={m['initialization']['initial_local_sha256'] for m in records}
            cameras={m['camera_geometry_sha256'] for m in records}
            schedule={m['schedule_sha256'] for m in records}
            assert len(initial)==len(cameras)==len(schedule)==1
            assert all(m['counts']==dict(denoiser=1780,encode=3560,decode=1869,initialize=0) for m in records)
            lines.append('%s/%s: all3 complete; identical initialization, camera and schedule hashes; expected denoiser/VAE calls.'%(backend,prompt))
        lines+=['','='*72,backend.upper()+' REPORT','='*72,(root/'report.txt').read_text()]
    observations=BASE_OUT/'visual-review.txt'
    if observations.exists():lines+=['','VISUAL REVIEW AND FINAL INTERPRETATION',observations.read_text()]
    write(BASE_OUT/'completion.json',dict(passed=True,cases=statuses,logical_case_count=18,source_preserved=True))
    (BASE_OUT/'both-models-report.txt').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
