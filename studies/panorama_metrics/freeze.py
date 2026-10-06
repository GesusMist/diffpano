"""Freeze only a validated evaluator; output manifests live outside the source tree."""
from .common import *
from .protocols import cpu_identity,protocols
from .features import gpu_identity

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Validation/freeze requires a batch job'
    cpu=read(ROOT/'validation/cpu.json');gpu=read(ROOT/'validation/gpu.json')
    assert cpu['passed'] and cpu['protocol_sha256']==cpu_identity()
    assert gpu['passed'] and gpu['protocol_sha256']==gpu_identity()
    value=dict(source_hashes=source_hashes(),protocols=protocols(),cpu_gate_sha256=sha(ROOT/'validation/cpu.json'),
        gpu_gate_sha256=sha(ROOT/'validation/gpu.json'),reference_manifest_sha256=sha(ROOT/'reference_manifest.json'),
        cpu_identity=cpu_identity(),gpu_identity=gpu_identity(),frozen_at=now(),job=os.environ['SLURM_JOB_ID'])
    dest=ROOT/'validation/frozen.json'
    if dest.exists():
        old=read(dest)
        assert old['source_hashes']==value['source_hashes'] and old['cpu_identity']==value['cpu_identity'] and old['gpu_identity']==value['gpu_identity'],'Existing freeze differs; explicitly archive before a validated revision'
        print('FROZEN_ALREADY_VALID');return
    atomic(dest,value);print('EVALUATOR_FROZEN',sha(dest),flush=True)
if __name__=='__main__':main()
