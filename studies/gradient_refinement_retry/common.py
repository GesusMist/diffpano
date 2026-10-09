"""Explicit tolerance-only retries alongside the frozen original study."""
import copy
from studies.gradient_refinement.common import *
from studies.gradient_refinement.common import source_hashes as original_source_hashes

RETRY = OUT / 'retries/rtol-1e-4'
INDICES = (12, 14, 28, 29, 30)
RTOL = 1e-4

def retry_source_hashes():
    result = original_source_hashes()
    for path in sorted((ROOT / 'studies/gradient_refinement_retry').rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.slurm', '.txt'):
            result[str(path.relative_to(ROOT))] = sha(path)
    return result

def check_scope(plan):
    original = read(RETRY / 'original-manifest.json')
    assert plan['indices'] == list(INDICES)
    assert original['source_hashes'] == original_source_hashes(), 'Original sources changed'
    assert plan['source_hashes'] == retry_source_hashes(), 'Retry sources changed'
    assert [r['index'] for r in plan['rows']] == list(INDICES)
    for row in plan['rows']:
        old = original['rows'][row['index']]
        expected = copy.deepcopy(old)
        assert old['semantic']['fusion']['relative_tolerance'] == 1e-5
        expected['semantic']['fusion']['relative_tolerance'] = RTOL
        expected['semantic_sha256'] = digest(expected['semantic'])
        expected['implementation_sha256'] = digest(plan['source_hashes'])
        assert row == expected, 'Retry changes more than tolerance/provenance'
    return original
