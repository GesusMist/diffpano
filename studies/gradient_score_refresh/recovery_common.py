"""Recovery of the two missing outputs required by the requested metrics."""
from studies.gradient_refinement.common import *
from studies.gradient_refinement.common import source_hashes as original_source_hashes
REF_OUT = OUT
BLEND_OUT = ROOT/'outputs/10.7gradient-blending-flux/seed0-v1'
SCORES = ROOT/'outputs/10.9gradient-metrics/seed0-v1'
RECOVERY = SCORES/'recovery'

def recovery_sources():
    result=original_source_hashes()
    for name in ('recovery_common.py','recover_prepare.py','recover.py'):
        path=ROOT/'studies/gradient_score_refresh'/name
        result[str(path.relative_to(ROOT))]=sha(path)
    return result

def folder(row):
    return Path(row['recovery_output'])

def case_complete(row,verify=True):
    p=folder(row)
    if not (p/'status.json').exists():return False
    status=read(p/'status.json')
    if status['state']!='complete':return False
    m=read(p/'metadata.json')
    if m.get('semantic_sha256')!=row['semantic_sha256'] or digest(m['source_hashes'])!=row['implementation_sha256']:return False
    if verify:
        assert sha(p/'metadata.json')==status['metadata_sha256']
        for name,value in m['artifacts'].items():assert sha(p/name)==value
    return True
