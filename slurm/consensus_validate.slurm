#!/bin/bash
#SBATCH --job-name=consensus-pt-tests
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
module purge
module load GCC/13.2.0 OpenMPI/4.1.6 Python/3.11.5 PyTorch/2.7.0
source /scratch/user/shig/virtual_envs/diffpano/bin/activate
cd /home/shig/diffpano
export OMP_NUM_THREADS=2
python -m compileall -q diffpano scripts tests
git diff --check
python -m unittest discover -s tests -p test_consensus_audit.py -v
python -m unittest discover -s tests -v
python - <<'PY'
import json,os,hashlib
from pathlib import Path
from scripts.consensus_spatial_controls import ROOT
ROOT.mkdir(parents=True,exist_ok=True)
sources=list(Path('diffpano').rglob('*.py'))+list(Path('scripts').glob('consensus_*.py'))+list(Path('tests').glob('*.py'))
(ROOT/'validation.json').write_text(json.dumps(dict(passed=True,job=os.environ['SLURM_JOB_ID'],compile=True,diff_check=True,focused=True,full=True,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
PY
