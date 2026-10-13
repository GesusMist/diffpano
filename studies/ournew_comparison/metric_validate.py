"""Metric collection tests must run in the frozen evaluation environment."""
import unittest
from studies.ournew_comparison.validate import result
from studies.ournew_comparison.common import *
def main():
    freeze_check();suite=unittest.defaultTestLoader.loadTestsFromName('studies.ournew_comparison.tests_metrics')
    r=unittest.TextTestRunner(verbosity=2).run(suite)
    result('metrics-cpu.json',passed=r.wasSuccessful(),tests=r.testsRun)
    if not r.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
