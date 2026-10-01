#!/bin/bash
#SBATCH --job-name=ttcea-validate
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
module purge
module load GCC/13.2.0 OpenMPI/4.1.6 Python/3.11.5 PyTorch/2.7.0
source /scratch/user/shig/virtual_envs/diffpano/bin/activate
cd /home/shig/diffpano
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
python -m studies.tt_cea.validate --config studies/tt_cea/experiment.yaml "$@"
