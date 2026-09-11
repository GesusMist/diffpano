#!/bin/bash
#SBATCH --job-name=detail-average-h
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

: "${1:?Usage: sbatch $0 BACKEND [BACKEND ...]}"
module purge
module load WebProxy
source activate_venv diffpano

REPO_ROOT="${DIFFPANO_ROOT:-$HOME/diffpano}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$SCRATCH/diffpano/outputs" "$SCRATCH/diffpano/hf_cache" "$SCRATCH/diffpano/tmp"
export HF_HOME="$SCRATCH/diffpano/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$SCRATCH/diffpano/tmp"
export PYTHONUNBUFFERED=1

for BACKEND in "$@"; do
    python -m scripts.detail_preserving_experiment --backend "$BACKEND"
done
