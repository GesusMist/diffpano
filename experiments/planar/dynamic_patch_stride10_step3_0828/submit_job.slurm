#!/bin/bash
#SBATCH --job-name=planar-dynamic-s10-d3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=g003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

if [[ $# -ne 1 || ! -f "$1" ]]; then
  echo "usage: sbatch $0 CONFIG" >&2
  exit 2
fi
CONFIG="$1"

module purge
module load WebProxy
source activate_venv diffpano

REPO_ROOT="${DIFFPANO_ROOT:-$HOME/diffpano}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$SCRATCH/SphereDiff/hf_cache" "$SCRATCH/diffpano/tmp" logs
export HF_HOME="$SCRATCH/SphereDiff/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$SCRATCH/diffpano/tmp"
export PYTHONUNBUFFERED=1

echo "config=$CONFIG job=$SLURM_JOB_ID"
python scripts/planar_test.py --config "$CONFIG"
