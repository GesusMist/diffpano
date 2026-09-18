#!/bin/bash
#SBATCH --job-name=erp4k-validation
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
module purge
module load GCC/13.2.0 OpenMPI/4.1.6 Python/3.11.5 PyTorch/2.7.0
source /scratch/user/shig/virtual_envs/diffpano/bin/activate
cd /home/shig/diffpano
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
python -m compileall -q diffpano scripts tests studies/bridge_erp4k
git diff --check
python -m unittest discover -s studies/bridge_erp4k -p 'test_*.py' -v
python -m unittest discover -s tests -v
python - <<'PY'
import os
from studies.bridge_erp4k.common import *
make_manifest()
write_new(ROOT/'validation.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],compile=True,diff_check=True,focused_tests=4,regression_tests=226,source_hashes=source_hashes(),manifest_sha256=sha(ROOT/'manifest.json')))
print('ERP4K VALIDATION PASSED',flush=True)
PY
