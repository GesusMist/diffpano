#!/bin/bash
#SBATCH --job-name=grad-extension
#SBATCH --account=132684267659
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err
set -euo pipefail
cd /home/shig/diffpano
gradient_index="${SLURM_ARRAY_TASK_ID:?}"
if (( gradient_index < 6 )); then
    export DIFFPANO_GRADIENT_BACKEND=pixeldit
else
    export DIFFPANO_GRADIENT_BACKEND=sd35
fi
exec bash studies/all_prompts/environment.sh diffpano -m studies.gradient_blending.pilot_case --index "$((gradient_index % 6))"
