"""Fresh CPU unit/regression gates and independent full checkpoint hashes."""
import argparse,time,unittest,concurrent.futures
from studies.ournew_comparison.common import *


def result(name,**extra):
    m=freeze_check();preserved()
    write(OUT/'validation'/name,dict(source_sha256=digest(m['source_hashes']),manifest_sha256=sha(OUT/'manifest.json'),job=os.environ['SLURM_JOB_ID'],**extra))


def tests():
    from studies.gradient_refinement.test_runner import flatten,REPLACED
    snapshot=freeze_check();runs=[]
    for directory in ('tests','studies/tt_cea/tests','studies/gradient_blending/tests','studies/gradient_refinement/tests','studies/ournew_comparison/tests'):
        tests=list(flatten(unittest.TestLoader().discover(directory)))
        suite=unittest.TestSuite(t for t in tests if t.id() not in REPLACED)
        path=OUT/'validation'/(directory.replace('/','-')+'.log');path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as f:r=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
        rec=dict(directory=directory,passed=r.wasSuccessful(),tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),log=str(path))
        runs.append(rec);print(rec,path.read_text()[-4000:],flush=True)
    passed=all(r['passed'] for r in runs)
    result('cpu.json',passed=passed,runs=runs,replaced_prior_source_certificate_tests=sorted(REPLACED))
    if not passed:raise SystemExit(1)


def weights():
    m=freeze_check();old=m['weight_audit'];cache={};records=[]
    # Hash all used files independently, deduplicating only the exact resolved file.
    for item in old['files']:
        checked={}
        for method,evidence in item['methods'].items():
            path=Path(evidence['path']);actual=str(path.resolve())
            if actual not in cache:cache[actual]=sha(path)
            h=cache[actual];assert h==evidence['sha256'] and path.stat().st_size==evidence['bytes'],path
            checked[method]=dict(path=str(path),sha256=h,bytes=path.stat().st_size)
        assert len({v['sha256'] for v in checked.values()})==1
        records.append(dict(name=item['name'],methods=checked,identical=True));print('WEIGHT_VERIFIED',item['name'],flush=True)
    result('weights.json',passed=True,all_shared_files_identical=True,files=records,unique_resolved_files_hashed=len(cache))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=('tests','weights'));a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID');(tests if a.kind=='tests' else weights)()
