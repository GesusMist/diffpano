#!/bin/bash
#SBATCH --job-name=pixeldit-validate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

STAGE="${1:?Usage: sbatch $0 single|full|one-view|smoke|baseline}"
module purge
module load WebProxy
source activate_venv diffpano

REPO_ROOT="${DIFFPANO_ROOT:-$HOME/diffpano}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$SCRATCH/diffpano/hf_cache" "$SCRATCH/diffpano/tmp" "$SCRATCH/diffpano/pixeldit-validation"
export HF_HOME="$SCRATCH/diffpano/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$SCRATCH/diffpano/tmp"
export PYTHONUNBUFFERED=1

case "$STAGE" in
    single)
        python scripts/pixeldit_single_view_test.py \
            --output "$SCRATCH/diffpano/pixeldit-validation/single"
        ;;
    full)
        python scripts/pixeldit_full_image_test.py \
            --output "$SCRATCH/diffpano/pixeldit-validation/full"
        ;;
    one-view)
        python scripts/pixeldit_one_view_erp_test.py \
            --output "$SCRATCH/diffpano/pixeldit-validation/one-view"
        ;;
    smoke)
        python scripts/generate.py --config configs/pixeldit_smoke.yaml
        ;;
    baseline)
        python scripts/generate.py --config configs/pixeldit_standard_average.yaml
        ;;
    *)
        echo "Unknown stage: $STAGE" >&2
        exit 2
        ;;
esac
