#!/bin/bash
#SBATCH --job-name=dense-ij-validation
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
module purge
module load GCC/13.2.0 OpenMPI/4.1.6 Python/3.11.5 PyTorch/2.7.0
source /scratch/user/shig/virtual_envs/diffpano/bin/activate
cd /home/shig/diffpano
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
python -m compileall -q studies/planar_dense_ij
git diff --check
python - <<'PY'
import copy
import os
import torch
from studies.planar_dense_ij.common import *
from diffpano.planar import build_planar_patch_layout, extract_planar_patch
previous=Path('outputs/planar-small-ij/20260918-p256-s64')
v=read(previous/'validation.json')
assert v['passed'] and sha(previous/'manifest.json')==v['manifest_sha256']
for path,digest in v['source_hashes'].items():
    assert sha(path)==digest, 'Previously validated I/J source changed: '+path
old=read(previous/'manifest.json')
m=make_manifest()
for r in m['runs']:
    prior=next(p for p in old['runs'] if (p['backend'],p['method'])==(r['backend'],r['method']))
    expected=copy.deepcopy(prior['config'])
    factor=FACTORS[r['backend']]
    expected['native_multidiffusion'].update(patch_size=512//factor,stride=32//factor)
    assert r['config']==expected, 'Only patch and stride may change from the 256/64 study'
    for key in ('initial_native','initial_file_sha256','conditioning_sha256','baseline_metadata_sha256'):
        assert r[key]==prior[key],key
    initial=torch.load(r['initial_native'],map_location='cpu',weights_only=True)
    n=r['config']['native_multidiffusion']
    assert list(initial.shape[-2:])==[n['canvas_height'],n['canvas_width']]
    layout=build_planar_patch_layout(n['canvas_height'],n['canvas_width'],n['patch_size'],n['stride'])
    coverage=torch.zeros_like(initial[:,:1])
    for patch in layout.patches:
        crop=extract_planar_patch(initial,patch)
        assert crop.shape[-2:]==(n['patch_size'],n['patch_size'])
        coverage[:,:,patch.y:patch.y+patch.size,patch.x:patch.x+patch.size]+=1
    assert torch.all(coverage>0)
    for axis in ('y','x'):
        starts=sorted({getattr(p,axis) for p in layout.patches})
        assert all((b-a)*factor==32 for a,b in zip(starts,starts[1:]))
    print(r['backend'],r['method'],'patches',layout.num_patches,'RGB patch/stride',512,32,'calls',r['expected_guided_predictions'],flush=True)
m['previous_study_manifest']=str(previous/'manifest.json')
m['previous_study_manifest_sha256']=sha(previous/'manifest.json')
write(ROOT/'manifest.json',m)
write(ROOT/'validation.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],source_hashes=sources(),manifest_sha256=sha(ROOT/'manifest.json'),
    prior_algorithm_validation=str(previous/'validation.json'),prior_algorithm_source_hashes_unchanged=True,
    checks=['only patch/stride changed from 256/64','exact native/RGB alignment','full coverage','exact 32-pixel stride','same initial tensors and conditioning','8 paired configs']))
print('VALIDATION PASSED: 8 runs, 512 RGB patch, 32 RGB stride',flush=True)
PY
