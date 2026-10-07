"""Finite gated two-backend pilot, at most two generation GPUs in total."""
import argparse
import subprocess
from studies.gradient_blending.common import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true',required=True);parser.parse_args()
    assert BACKEND=='sana','Launch combined pilot from default SANA study context'
    preserved();current=core_hashes()
    for file in ['validation.json','gpu-smoke.json','offline/replay.json']:
        record=read(GATES/file);assert record['passed'] and record['core_hashes']==current,file
    flux=read(BASE_OUT/'flux/backend-validation.json')
    assert flux['passed'] and flux['core_hashes']==current
    assert sha(BASE_OUT/'flux/baseline.json')==flux['baseline_sha256']
    assert read(GATES/'offline/replay.json')['pilot_lambda_color']==.1
    active=subprocess.check_output(['squeue','-u','shig','-h','-o','%i|%j'],universal_newlines=True)
    assert not any('|grad-pilot' in line for line in active.splitlines()),'Existing gradient pilot active; no duplicate submission'
    for backend,folder in [('sana',BASE_OUT),('flux',BASE_OUT/'flux')]:
        for prompt in PROMPTS:assert read(folder/'cases'/prompt/'rgb/status.json')['state']=='complete'
    pending=[]
    for i in range(12):
        root=BASE_OUT if i<6 else BASE_OUT/'flux';local=i%6
        folder=root/'cases'/PROMPTS[local//2]/MODES[1+local%2];status=folder/'status.json'
        if status.exists() and read(status)['state']=='complete':
            assert sha(folder/'final.png')==read(folder/'metadata.json')['artifacts']['final.png'];continue
        pending.append(i)
    if not pending:print('All 18 logical cases complete');return
    command=['sbatch','--parsable','--array='+','.join(map(str,pending))+'%2','studies/gradient_blending/pilot.slurm']
    output=subprocess.check_output(command,cwd=str(ROOT),universal_newlines=True).strip()
    job=next(line.split(';')[0] for line in output.splitlines() if line.split(';')[0].isdigit())
    write(BASE_OUT/'pilot-submission.json',dict(job=job,indices=pending,max_concurrent_generations=2,
        command=command,source_hashes=source_hashes(),logical_case_count=18,new_gradient_cases=12,
        scope_extension='Explicit user request to add FLUX, 2026-10-06',
        reused_controls=['sana/underwater/rgb','sana/firework/rgb','flux/ruins/rgb','flux/underwater/rgb','flux/firework/rgb']))
    print('PILOT_SUBMITTED',job,pending,flush=True)
if __name__=='__main__':main()
