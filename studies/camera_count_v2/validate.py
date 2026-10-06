"""CPU validation freezes the new source set without editing historical gates."""
import os
import subprocess
import sys
import unittest
from .common import *
def main():
    assert os.environ.get('SLURM_JOB_ID')
    assert_preserved()
    assert read(ROOT/'study.json')['rows']==rows()
    assert read(ROOT/'study.json')['expected']==126
    subprocess.check_call([sys.executable,'-m','compileall','-q',str(STUDY)],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    subprocess.check_call(['git','diff','--check'],cwd=str(REPO))
    suite=unittest.defaultTestLoader.discover(str(STUDY/'tests'),top_level_dir=str(REPO))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    # Existing relevant regressions execute with their unchanged implementations.
    suite=unittest.TestLoader().discover(str(REPO/'studies/camera_patching/tests'),top_level_dir=str(REPO/'studies/camera_patching/tests'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    assert_preserved()
    atomic(ROOT/'validation.json',dict(passed=True,time=now(),job=os.environ['SLURM_JOB_ID'],
        source_hashes=source_hashes(),geometry_identity=geometry_identity(),expected=126,
        sphere_flux_steps=20,notes='Study tests and unchanged camera-patching regression suite passed'))
    emit('VALIDATION_PASSED',expected=126,source_fingerprint=fingerprint())
if __name__=='__main__':main()
