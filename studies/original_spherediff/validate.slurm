#!/bin/bash
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
module purge
module load WebProxy
source activate_venv spherediff
cd /home/shig/diffpano
export HF_HOME="$SCRATCH/SphereDiff/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$SCRATCH/SphereDiff/tmp"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
python -m studies.original_spherediff.validate
