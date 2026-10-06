#!/bin/bash
#SBATCH --job-name=fsctrl-preflight
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
methods=(spherediff diffpano)
method="${methods[${SLURM_ARRAY_TASK_ID:?}]}"
exec bash /home/shig/diffpano/studies/all_prompts/environment.sh "$method" -m studies.flux_step_controls.run --preflight "$method"
