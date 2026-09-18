#!/bin/bash
#SBATCH --job-name=bridge-factorial-validation
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
python -m compileall -q diffpano scripts tests
git diff --check
python -m unittest discover -s tests -p test_bridge_factorial.py -v
if [[ "${1:-full}" == "focused" ]]; then exit 0; fi
python -m unittest discover -s tests -v
python - <<'PY'
import os
from scripts.bridge_factorial_common import ROOT,make_manifest,make_provenance,source_hashes,sha,write_new
make_manifest()
make_provenance()
write_new(ROOT/'validation.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],compile=True,diff_check=True,focused=True,full=True,source_hashes=source_hashes(),manifest_sha256=sha(ROOT/'manifest.json')))
PY
