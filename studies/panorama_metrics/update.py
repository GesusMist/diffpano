"""One finite incremental pass; independent failures remain visible in the report."""
import fcntl
import traceback
from .common import *

def main():
    assert os.environ.get('SLURM_JOB_ID')
    frozen=read(ROOT/'validation/frozen.json');assert source_hashes()==frozen['source_hashes']
    with (ROOT/'.evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        failures=[]
        from .cpu import main as cpu
        from .features import main as gpu
        from .inventory import scan
        from .report import update
        for name,operation in [('cpu',cpu),('gpu',gpu)]:
            try:operation()
            except Exception as e:
                traceback.print_exc();failures.append(dict(stage=name,error=repr(e)))
        # Always record completed controls and failed/pending cases even after evaluator failures.
        rows=scan();update(rows)
        from studies.flux_step_controls.common import assert_preserved,manifest
        assert_preserved(manifest())
        atomic(ROOT/'updates'/(os.environ['SLURM_JOB_ID']+'.json'),dict(time=now(),failures=failures,
            frozen_sha256=sha(ROOT/'validation/frozen.json'),counts=dict(__import__('collections').Counter(r['completion_status'] for r in rows))))
        if failures:raise SystemExit(1)
if __name__=='__main__':main()
