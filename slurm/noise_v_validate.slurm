#!/bin/bash
#SBATCH --job-name=noise-v-preflight
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
module purge
module load GCC/13.2.0 OpenMPI/4.1.6 Python/3.11.5 PyTorch/2.7.0
source /scratch/user/shig/virtual_envs/diffpano/bin/activate
cd /home/shig/diffpano
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
python -m compileall -q diffpano scripts tests
git diff --check
python -m unittest discover -s tests -p test_erp_noise_initialization.py -v
python -m unittest discover -s tests -v
python - <<'PY'
import json,os
from scripts.noise_v_common import ROOT,read,source_hashes
p=ROOT/'execution.json';x=read(p);x['validation']=dict(passed=True,job=os.environ['SLURM_JOB_ID'],compile=True,diff_check=True,focused=True,full=True,source_hashes=source_hashes());p.write_text(json.dumps(x,indent=2)+'\n')
PY
python -m scripts.noise_v_preflight
