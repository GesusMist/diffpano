#!/bin/bash
#SBATCH --job-name=followup24-generation
#SBATCH --account=132684267659
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/10.15layout24-flux216/seed0-v1/logs/generation/%x-%A_%a.out
set -euo pipefail
cd /home/shig/diffpano
exec bash studies/all_prompts/environment.sh diffpano -m studies.followup24.dispatch "$@"
