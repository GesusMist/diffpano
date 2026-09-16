#!/bin/bash
#SBATCH --job-name=dense-no-tests
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
python -m unittest discover -s tests -p test_dense_no.py -v
python -m unittest discover -s tests -v
python - <<'PY'
import hashlib,json,os
from pathlib import Path
from scripts.dense_erp_experiment import NO_ROOT,check_pairing,load_experiment_config,config_differences
records={}
for backend in ('sd35','flux','sana','sd2','pixeldit'):
 l=load_experiment_config('configs/experiments/erp_later/'+backend+'-l.yaml')
 for e in ('n','o'):
  c=load_experiment_config('configs/experiments/erp_later/'+backend+'-'+e+'.yaml');check_pairing(c)
  records[backend+'-'+e]=config_differences(l.to_dict(),c.to_dict())
NO_ROOT.mkdir(parents=True,exist_ok=True)
baseline_root=Path('outputs/vae-residual-controls/20260915-dense-lm')
baseline_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for label in ('L','M','geometry') for p in (baseline_root/label).rglob('*') if p.is_file()}
(NO_ROOT/'validation.json').write_text(json.dumps(dict(passed=True,test_job=os.environ['SLURM_JOB_ID'],focused_tests=8,compileall_passed=True,diff_check_passed=True,config_diffs=records,baseline_artifact_hashes=baseline_hashes),indent=2)+'\n')
print('N/O regression and actual L metadata pairing gates PASSED',flush=True)
PY
