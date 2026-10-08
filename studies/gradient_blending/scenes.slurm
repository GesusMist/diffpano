#!/bin/bash
#SBATCH --job-name=grad-flux-scenes
#SBATCH --account=132684267659
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err
set -euo pipefail
cd /home/shig/diffpano
export DIFFPANO_GRADIENT_BACKEND=flux DIFFPANO_GRADIENT_SUITE=flux-scenes20
exec bash studies/all_prompts/environment.sh diffpano -m studies.gradient_blending.scene_case --index "${SLURM_ARRAY_TASK_ID:?}"
