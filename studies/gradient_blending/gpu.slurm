#!/bin/bash
#SBATCH --job-name=grad-study
#SBATCH --account=132684267659
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
cd /home/shig/diffpano
exec bash studies/all_prompts/environment.sh diffpano -m "$@"
