#!/bin/bash
#SBATCH --job-name=planar-latent-strides
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=g003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-29%2
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err

set -euo pipefail

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

MANIFEST="experiments/planar/latent_stride_sweep_0827/manifest.txt"
CONFIG="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "No valid config for array task $SLURM_ARRAY_TASK_ID: $CONFIG" >&2
  exit 2
fi

echo "array_task=$SLURM_ARRAY_TASK_ID config=$CONFIG"
python scripts/planar_test.py --config "$CONFIG"
