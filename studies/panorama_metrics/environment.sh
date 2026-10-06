#!/bin/bash
set -euo pipefail
module purge
module load WebProxy
source activate_venv diffpano
cd /home/shig/diffpano
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export TORCH_HOME=/scratch/user/shig/diffpano/evaluation-assets/torch
export HF_HOME=/scratch/user/shig/diffpano/evaluation-assets/hf
export MPLCONFIGDIR=/scratch/user/shig/diffpano/evaluation-assets/mpl
exec /scratch/user/shig/diffpano/evaluation-env/bin/python "$@"
