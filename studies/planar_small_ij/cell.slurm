#!/bin/bash
#SBATCH --job-name=small-ij
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
module purge
module load WebProxy
source activate_venv diffpano
cd /home/shig/diffpano
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SCRATCH/diffpano/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TRANSFORMERS_CACHE="$HF_HOME/transformers" TMPDIR="$SCRATCH/diffpano/tmp"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
python -m studies.planar_small_ij.run --backend "$1" --method "$2"
