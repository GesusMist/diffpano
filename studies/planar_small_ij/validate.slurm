#!/bin/bash
#SBATCH --job-name=small-ij-validation
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
python -m compileall -q studies/planar_small_ij
git diff --check
python -m unittest discover -s tests -p 'test_current_state_transition.py' -v
python -m unittest discover -s tests -p 'test_no_residual_planar.py' -v
python - <<'PY'
import os
import torch
from studies.planar_small_ij.common import *
from diffpano.planar import build_planar_patch_layout, extract_planar_patch
m=make_manifest()
for r in m['runs']:
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
write(ROOT/'manifest.json',m)
write(ROOT/'validation.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],source_hashes=sources(),manifest_sha256=sha(ROOT/'manifest.json')))
print('VALIDATION PASSED: 8 runs, 256 RGB patch, 64 RGB stride, exact native alignment and full coverage',flush=True)
PY
