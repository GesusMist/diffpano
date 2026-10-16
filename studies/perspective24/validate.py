"""Focused CPU identity/native tests and independent checkpoint verification."""
import argparse,unittest
from studies.perspective24.common import *

def result(name,**values):
    m=freeze_check();write(OUT/'validation'/name,dict(source_sha256=digest(m['source_hashes']),manifest_sha256=sha(OUT/'manifest.json'),
        job=os.environ.get('SLURM_JOB_ID'),**values))

def tests():
    m=freeze_check();suite=unittest.defaultTestLoader.discover(str(ROOT/'studies/perspective24/tests'))
    path=OUT/'validation/tests.log'
    with path.open('w') as stream:run=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    result('cpu.json',passed=run.wasSuccessful(),tests=run.testsRun,failures=len(run.failures),errors=len(run.errors),log=str(path))
    print(path.read_text(),flush=True)
    if not run.wasSuccessful():raise SystemExit(1)

def weights():
    m=freeze_check();records=[];cache={}
    for item in m['weight_audit']['files']:
        record=item['methods']['diffpano'];p=Path(record['path']);resolved=str(p.resolve())
        if resolved not in cache:cache[resolved]=sha(p)
        assert cache[resolved]==record['sha256'] and p.stat().st_size==record['bytes'],p
        records.append(dict(name=item['name'],path=str(p),resolved=resolved,sha256=cache[resolved],bytes=p.stat().st_size))
        print('WEIGHT_VERIFIED',item['name'],flush=True)
    result('weights.json',passed=True,files=records,unique_files_hashed=len(cache))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=('tests','weights','all'));args=parser.parse_args()
    if args.kind in ('tests','all'):tests()
    if args.kind in ('weights','all'):weights()
