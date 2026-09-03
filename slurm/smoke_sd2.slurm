#!/bin/bash
#SBATCH --job-name=smoke-sd2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

CONFIG="${1:?Usage: sbatch $0 CONFIG.yaml}"
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

python scripts/generate.py --config "$CONFIG"
