#!/bin/bash
#SBATCH --job-name=ournew-smoke
#SBATCH --account=132684267659
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=outputs/10.9ournew-flux28/seed0-v1/logs/validation/%x-%j.out
set -euo pipefail
cd /home/shig/diffpano
exec bash studies/all_prompts/environment.sh "$1" -m "$2"
