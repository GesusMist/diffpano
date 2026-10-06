"""Validate the expanded inventory before replacing the old source freeze."""
import collections
import unittest
from .common import *

def main():
    assert os.environ.get('SLURM_JOB_ID')
    previous=ROOT/'history/20261002-before-camera-count-refresh/validation/frozen.json'
    old=read(previous);before=source_hashes()
    # These files define metric values and cache identities; they are unchanged.
    for name in ('features.py','render.py','seams.py','numerics.py','distributions.py','protocols.py'):
        path='studies/panorama_metrics/'+name
        assert before[path]==old['source_hashes'][path],('Metric implementation changed',path)
    from .protocols import cpu_identity
    from .features import Extractor,gpu_identity
    assert cpu_identity()==old['cpu_identity'] and gpu_identity()==old['gpu_identity']
    from .validate_cpu import main as validate_cpu
    validate_cpu()
    suite=unittest.defaultTestLoader.loadTestsFromName('studies.panorama_metrics.tests.test_camera_counts')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    from .inventory import scan
    rows=scan();summary=collections.Counter(r['completion_status'] for r in rows)
    assert len(rows)==426 and summary==dict(complete=411,blocked=6,cancelled=9),summary
    new=[r for r in rows if r['family']=='camera_count']
    assert sum(r['completion_status']=='complete' for r in new)==120
    # Validate all requested comparisons before extracting any new image features.
    from .report import key
    from .camera_counts import comparisons
    groups=collections.defaultdict(list)
    for r in rows:groups[key(r)].append(r)
    pending={r['id']:{k:None for k in PER_IMAGE} for r in rows}
    comp=comparisons(groups,lambda *args:[],pending)
    errors=[r for r in comp if r['status']=='incompatible_provenance']
    assert not errors,errors
    from .validate_gpu import validate
    extractor=Extractor();assert extractor.clip is not None and extractor.inception is not None
    validate(extractor)
    assert source_hashes()==before
    atomic(ROOT/'validation/camera_count_refresh.json',dict(passed=True,counts=dict(summary),new_complete=120,
        added_tests=result.testsRun,source_hashes=before,previous_freeze_sha256=sha(previous),
        cpu_identity=cpu_identity(),gpu_identity=gpu_identity(),job=os.environ['SLURM_JOB_ID'],time=now()))
    active=ROOT/'validation/frozen.json'
    if active.exists():
        present=read(active)
        if present['source_hashes']!=before:
            assert sha(active)==sha(previous),'Unexpected concurrent freeze change'
            active.unlink()  # Exact prior freeze remains in the checked history snapshot.
    from .freeze import main as freeze
    freeze()
    print('REFRESH_VALIDATED',dict(summary),'NEW_SCORED_NEXT',120,flush=True)
if __name__=='__main__':main()
