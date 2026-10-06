#!/bin/bash
#SBATCH --job-name=fsctrl-run
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
step_environment=diffpano
if (( SLURM_ARRAY_TASK_ID < 21 )); then step_environment=spherediff; fi
exec bash /home/shig/diffpano/studies/all_prompts/environment.sh "$step_environment" -m studies.flux_step_controls.run --index "${SLURM_ARRAY_TASK_ID:?}"
