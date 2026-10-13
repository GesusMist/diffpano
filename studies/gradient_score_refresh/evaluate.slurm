#!/bin/bash
#SBATCH --job-name=grad-score-eval
#SBATCH --account=132684267659
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/10.9gradient-metrics/seed0-v1/logs/evaluation/%x-%j.out
set -euo pipefail
cd /home/shig/diffpano
exec bash studies/panorama_metrics/environment.sh -m studies.gradient_score_refresh.evaluate "$@"
