import os
import re
import subprocess
import sys
import time
from studies.gradient_blending.common import *

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm validation required'
    preserved();baseline_audit();snapshot=core_hashes();runs=[]
    folder=OUT/'validation'/os.environ['SLURM_JOB_ID'];folder.mkdir(parents=True,exist_ok=True)
    for test_dir in ['studies/gradient_blending/tests','tests','studies/tt_cea/tests','studies/all_prompts/tests']:
        command=[sys.executable,'-m','unittest','discover','-s',test_dir,'-v']
        log=folder/(test_dir.replace('/','-')+'.log');start=time.perf_counter()
        with log.open('w') as f:result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        content=log.read_text();match=re.search(r'Ran (\d+) tests?',content)
        runs.append(dict(command=command,exit_code=result.returncode,seconds=time.perf_counter()-start,
                         tests=int(match.group(1)) if match else None,log=str(log),skipped='skipped=' in content))
        print(runs[-1],flush=True)
        if result.returncode:
            write(folder/'result.json',dict(passed=False,runs=runs));print(content[-18000:],flush=True);raise SystemExit(1)
    from studies.gradient_blending.synthetic import run
    synthetic=run();assert core_hashes()==snapshot;preserved()
    record=dict(passed=True,runs=runs,core_hashes=snapshot,synthetic_cases=len(synthetic),job=os.environ['SLURM_JOB_ID'])
    write(folder/'result.json',record);write(OUT/'validation.json',record);print('VALIDATION_PASSED',flush=True)
if __name__=='__main__':main()
