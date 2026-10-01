"""Execute validation commands and preserve actual exits/logs; never bless skips."""
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from studies.tt_cea.common import *

def main(config,development=False):
    started=time.perf_counter();load_settings(config)
    from studies.tt_cea.audit import main as audit
    audit(config)
    job=os.environ.get('SLURM_JOB_ID','local');out=ROOT/'validation-attempts'/job
    out.mkdir(parents=True,exist_ok=False);snapshot=study_hashes();commands=[]
    runs=[['python','-m','compileall','-q','studies/tt_cea'],['git','diff','--check'],
          ['python','-m','studies.tt_cea.metadata'],
          ['python','-m','unittest','discover','-s','studies/tt_cea/tests','-v']]
    if not development:runs += [['python','-m','unittest','discover','-s',d,'-v'] for d in ('tests','studies/gwtf_erp4k','studies/gwtf_underwater')]
    passed=True
    for i,cmd in enumerate(runs):
        path=out/('%02d.log'%i);t=time.perf_counter()
        with path.open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=REPO)
        content=path.read_text();import re
        count=re.search(r'Ran (\d+) tests?',content)
        skipped='skipped=' in content or '... skipped ' in content
        record=dict(command=cmd,exit_code=r.returncode,seconds=time.perf_counter()-t,log=str(path),
            tests=int(count.group(1)) if count else None,skipped=skipped)
        commands.append(record);print(record,flush=True)
        if r.returncode or skipped:passed=False;print(content[-12000:],flush=True);break
    unchanged=snapshot==study_hashes()
    result=dict(passed=passed and unchanged and not development,development=development,
        commands=commands,source_hashes=snapshot,sources_unchanged=unchanged,manifest_sha256=sha(ROOT/'manifest.json'),
        job=job,seconds=time.perf_counter()-started,python=sys.version,environment=dict(slurm_job=job,OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS')))
    atomic(out/'result.json',result)
    if result['passed']:
        from studies.tt_cea.metadata import certify_metadata_revision
        result['metadata_only_compatibility']=certify_metadata_revision()
        atomic(out/'result.json',result)
        verify_preservation();atomic(ROOT/'validation.json',result)
        print('VALIDATION PASSED',flush=True)
    elif not passed:raise SystemExit(1)
    elif development:print('DEVELOPMENT TESTS PASSED; NO PRODUCTION GATE CREATED',flush=True)
    else:raise RuntimeError('Sources changed during validation; revalidate the finished source tree')

if __name__=='__main__':
    p=parser(__doc__);p.add_argument('--development',action='store_true');a=p.parse_args();main(a.config,a.development)
