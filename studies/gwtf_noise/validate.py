"""Run the requested focused/full suites and freeze the scientific sources."""
import os
import sys
import time
import unittest
from studies.gwtf_noise.common import ROOT,write,sha,source_hashes,audit_controls,verify_preservation


def main():
    start=time.perf_counter();sys.path.insert(0,'tests');results={}
    for label,pattern in (('focused','test_gwtf_noise_initialization.py'),('full','test*.py')):
        suite=unittest.defaultTestLoader.discover('tests',pattern=pattern)
        t=time.perf_counter();r=unittest.TextTestRunner(verbosity=2).run(suite)
        results[label]=dict(count=r.testsRun,passed=r.wasSuccessful(),seconds=time.perf_counter()-t)
        if not r.wasSuccessful():raise SystemExit(1)
    audit_controls();verify_preservation()
    write(ROOT/'validation.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],suites=results,seconds=time.perf_counter()-start,
        compileall=True,diff_check=True,source_hashes=source_hashes(),manifest_sha256=sha(ROOT/'manifest.json')))
    print('GWTFlow full validation PASSED',results,flush=True)

if __name__=='__main__':main()
