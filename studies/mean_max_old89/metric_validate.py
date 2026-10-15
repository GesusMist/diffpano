import unittest
from studies.mean_max_old89.validate import result
from studies.mean_max_old89.common import freeze_check

def main():
    freeze_check()
    suite=unittest.defaultTestLoader.loadTestsFromName('studies.mean_max_old89.tests_metrics')
    r=unittest.TextTestRunner(verbosity=2).run(suite)
    result('metrics-cpu.json',passed=r.wasSuccessful(),tests=r.testsRun)
    if not r.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
