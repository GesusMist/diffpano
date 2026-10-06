import unittest
from .common import *
from .protocols import cpu_identity

def main():
    assert os.environ.get('SLURM_JOB_ID')
    suite=unittest.defaultTestLoader.loadTestsFromName('studies.panorama_metrics.tests.test_cpu')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    atomic(ROOT/'validation/cpu.json',dict(passed=True,tests=result.testsRun,job=os.environ['SLURM_JOB_ID'],
        protocol_sha256=cpu_identity(),source_hashes=source_hashes(),time=now()))
if __name__=='__main__':main()
