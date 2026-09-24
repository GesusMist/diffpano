import os,time,unittest,sys
from studies.gwtf_erp4k.common import *
start=time.perf_counter();suites={}
sys.path.insert(0,'tests')
for label,folder,pattern in (('followup','studies/gwtf_erp4k','test_*.py'),('regression','tests','test_*.py')):
    t=time.perf_counter();suite=unittest.defaultTestLoader.discover(folder,pattern=pattern,top_level_dir=folder)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    suites[label]=dict(count=result.testsRun,passed=result.wasSuccessful(),seconds=time.perf_counter()-t)
    if not result.wasSuccessful():raise SystemExit(1)
manifest()
write(ROOT/'validation.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],suites=suites,seconds=time.perf_counter()-start,
    source_hashes=source_hashes(),manifest_sha256=sha(ROOT/'manifest.json'),compile=True,diff_check=True))
print('4K GWTFlow VALIDATION PASSED',suites,flush=True)
