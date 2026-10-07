"""CPU gate for extension scheduling, full-length mocked trajectories and cached controls."""
import os,sys,subprocess,unittest
from studies.gradient_blending.common import *

def main():
    assert os.environ.get('SLURM_JOB_ID')
    preserved();sources=source_hashes()
    # Existing operator gates remain byte-identical; these exercise new runner scheduling.
    suite=unittest.defaultTestLoader.loadTestsFromName('studies.gradient_blending.extension_checks.ExtensionChecks')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun==2
    for backend in ('pixeldit','sd35'):
        env=dict(os.environ,DIFFPANO_GRADIENT_BACKEND=backend)
        subprocess.run([sys.executable,'-m','studies.gradient_blending.prepare_extension'],check=True,env=env)
    assert sources==source_hashes()
    write(BASE_OUT/'extension-validation.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],tests=result.testsRun,
        core_hashes=core_hashes(),source_hashes=sources,backends=['pixeldit','sd35'],
        checks='50/40 real-length mock trajectories; exact RGB regression; schedule, all interval/terminal fusions, current-state reconstruction, original call counts; PixelDiT has no VAE calls; pinned checkpoints and six cached controls'))
    print('EXTENSION_VALIDATION_PASSED',flush=True)
if __name__=='__main__':main()
