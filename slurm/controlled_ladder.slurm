#!/bin/bash
#SBATCH --job-name=controlled-ladder
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
module purge
module load WebProxy
source activate_venv diffpano
cd "${DIFFPANO_ROOT:-$HOME/diffpano}"
export HF_HOME=/scratch/user/shig/diffpano/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export PYTHONUNBUFFERED=1
for PHASE in "$@"; do
    python -m scripts.controlled_ladder --phase "$PHASE"
done
