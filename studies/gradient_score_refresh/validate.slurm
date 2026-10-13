#!/bin/bash
#SBATCH --job-name=grad-score-check
#SBATCH --account=132684267659
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=outputs/10.9gradient-metrics/seed0-v1/logs/validation/%x-%j.out
set -euo pipefail
cd /home/shig/diffpano
exec bash studies/panorama_metrics/environment.sh -m unittest -v studies.gradient_score_refresh.test_metrics
