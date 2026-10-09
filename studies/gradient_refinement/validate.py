"""Fresh validation against frozen implementation hashes; no historical gate edits."""
import os,time,unittest
from pathlib import Path
from studies.gradient_refinement.common import OUT,write,source_hashes
from studies.gradient_refinement.test_runner import flatten,REPLACED

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Run validation in a Slurm allocation'
    snapshot=source_hashes();runs=[];replaced=[]
    # One interpreter avoids repeated slow imports on the shared filesystem.
    # These directories have disjoint top-level test module names.
    for directory in ('tests','studies/tt_cea/tests','studies/gradient_blending/tests','studies/gradient_refinement/tests'):
        tests=list(flatten(unittest.TestLoader().discover(directory)))
        replaced.extend(t.id() for t in tests if t.id() in REPLACED)
        suite=unittest.TestSuite(t for t in tests if t.id() not in REPLACED)
        log=OUT/('tests-'+directory.replace('/','-')+'.log');start=time.perf_counter()
        with log.open('w') as stream:result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        row=dict(directory=directory,passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),seconds=time.perf_counter()-start,log=str(log))
        print(row,log.read_text()[-2500:],flush=True);runs.append(row)
    passed=all(r['passed'] for r in runs) and snapshot==source_hashes()
    write(OUT/'validation.json',dict(passed=passed,job=os.environ['SLURM_JOB_ID'],results=runs,source_hashes=snapshot,
        source_unchanged=snapshot==source_hashes(),replaced_historical_source_tests=sorted(set(replaced)),
        historical_gate_note='Original gates untouched; archived disabled-path, initializer and preserved-certificate-rejection checks replace only two incompatible source certificates.',time=time.time()))
    if not passed:raise SystemExit(1)
if __name__=='__main__':main()
