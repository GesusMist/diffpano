#!/bin/bash
#SBATCH --job-name=allp21-run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err
set -euo pipefail
sweep_index="${SLURM_ARRAY_TASK_ID:?Array index required}"
sweep_environment=diffpano
if (( sweep_index >= 168 )); then sweep_environment=spherediff; fi
exec bash /home/shig/diffpano/studies/all_prompts/environment.sh "$sweep_environment" -m studies.all_prompts.run --index "$sweep_index" "$@"
