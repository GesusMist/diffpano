#!/bin/bash
#SBATCH --job-name=ournew-generate
#SBATCH --account=132684267659
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/10.9ournew-flux28/seed0-v1/logs/generation/%x-%A_%a.out
set -euo pipefail
cd /home/shig/diffpano
ournew_environment=diffpano
if (( SLURM_ARRAY_TASK_ID % 7 == 0 )); then ournew_environment=spherediff; fi
exec bash studies/all_prompts/environment.sh "$ournew_environment" -m studies.ournew_comparison.run "$@"
