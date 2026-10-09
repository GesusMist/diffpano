#!/bin/bash
#SBATCH --job-name=gradref-gpu
#SBATCH --account=132684267659
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/10.8gradient-refinement-flux/seed0-v1/%x-%A_%a.out
set -euo pipefail
cd /home/shig/diffpano
exec bash studies/all_prompts/environment.sh diffpano -m "$@"
