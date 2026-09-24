#!/bin/bash
#SBATCH --job-name=gwtf-generation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
module purge
module load WebProxy
source activate_venv diffpano
cd /home/shig/diffpano
export HF_HOME="$SCRATCH/diffpano/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1
export TRANSFORMERS_CACHE="$HF_HOME/transformers" TMPDIR="$SCRATCH/diffpano/tmp"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
python -m studies.gwtf_noise.run "$1"
