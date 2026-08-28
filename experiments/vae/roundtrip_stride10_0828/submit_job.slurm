#!/bin/bash
#SBATCH --job-name=vae-roundtrip-s10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=g003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: sbatch $0 MODEL ROUNDTRIPS" >&2
  exit 2
fi
MODEL="$1"
ROUNDTRIPS="$2"

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

echo "model=$MODEL roundtrips=$ROUNDTRIPS job=$SLURM_JOB_ID"
python scripts/vae_patch_roundtrip.py \
  --model "$MODEL" \
  --roundtrips "$ROUNDTRIPS" \
  --input experiments/vae/image.png \
  --output-root test_outputs/vae_roundtrip_stride10_0828 \
  --patch-pixels 640 \
  --stride-latents 10 \
  --vae-chunk-size 4
