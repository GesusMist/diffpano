#!/bin/bash
#SBATCH --job-name=campatch-preflight
#SBATCH --account=132684267659
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err
set -euo pipefail
models=(sana flux sd35 pixeldit)
exec bash /home/shig/diffpano/studies/all_prompts/environment.sh diffpano -m studies.camera_patching.preflight --backend "${models[${SLURM_ARRAY_TASK_ID:?}]}"
