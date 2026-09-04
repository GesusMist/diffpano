#!/bin/bash
#SBATCH --job-name=x0-validate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

STAGE="${1:?Usage: sbatch $0 single|one-erp|full sana|flux|sd2|pixeldit}"
BACKEND="${2:?Usage: sbatch $0 single|one-erp|full sana|flux|sd2|pixeldit}"
case "$BACKEND" in
    sana|flux|sd2|pixeldit) ;;
    *) echo "Unknown backend: $BACKEND" >&2; exit 2 ;;
esac

module purge
module load WebProxy
source activate_venv diffpano

REPO_ROOT="${DIFFPANO_ROOT:-$HOME/diffpano}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$SCRATCH/diffpano/hf_cache" "$SCRATCH/diffpano/tmp" \
    "$SCRATCH/diffpano/x0-consensus-validation/$BACKEND/$STAGE-$SLURM_JOB_ID"
export HF_HOME="$SCRATCH/diffpano/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$SCRATCH/diffpano/tmp"
export PYTHONUNBUFFERED=1

CONFIG="configs/x0_consensus_${BACKEND}.yaml"
OUTPUT="$SCRATCH/diffpano/x0-consensus-validation/$BACKEND/$STAGE-$SLURM_JOB_ID"
case "$STAGE" in
    single)
        python scripts/x0_consensus_single_view_test.py \
            --config "$CONFIG" --output "$OUTPUT"
        ;;
    one-erp)
        python scripts/x0_consensus_one_view_erp_test.py \
            --config "$CONFIG" --output "$OUTPUT"
        ;;
    full)
        python scripts/generate.py --config "$CONFIG"
        ;;
    *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
