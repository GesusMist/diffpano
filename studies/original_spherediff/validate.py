"""Execute focused wrapper validation on CPU, then freeze the audit gate."""
import os
import re
import subprocess
import time
from studies.original_spherediff.common import *


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    job = os.environ.get('SLURM_JOB_ID', 'local')
    out = ROOT / 'validation-attempts' / job
    out.mkdir(parents=True, exist_ok=False)
    commands = [[sys.executable, '-m', 'compileall', '-q', str(STUDY)], ['git', 'diff', '--check'],
                [sys.executable, '-m', 'studies.original_spherediff.audit'],
                [sys.executable, '-m', 'unittest', 'discover', '-s', 'studies/original_spherediff/tests', '-v']]
    results = []
    for i, command in enumerate(commands):
        started = time.perf_counter()
        p = subprocess.run(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = out / ('%02d.log' % i); log.write_text(p.stdout)
        m = re.search(r'Ran (\d+) tests', p.stdout)
        results.append(dict(command=command, exit_code=p.returncode, seconds=time.perf_counter()-started,
                            log=str(log), tests=int(m.group(1)) if m else 0))
        print(command, p.returncode, flush=True)
        if p.returncode:
            atomic(out/'failed.json', dict(passed=False, commands=results, job=job))
            raise RuntimeError(p.stdout[-6000:])
    atomic(ROOT/'validation.json', dict(passed=True, commands=results, job=job, source_hashes=source_hashes(),
                                       manifest_sha256=sha(ROOT/'manifest.json')))


if __name__ == '__main__':
    main()
