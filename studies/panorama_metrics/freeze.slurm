#!/bin/bash
#SBATCH --job-name=pmetrics-freeze
#SBATCH --account=132684267659
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
cd /home/shig/diffpano
bash studies/panorama_metrics/environment.sh -m studies.panorama_metrics.validate_cpu
bash studies/panorama_metrics/environment.sh -m pip freeze > outputs/metrics-evaluation/seed0-v1/environment-freeze.txt
exec bash studies/panorama_metrics/environment.sh -m studies.panorama_metrics.freeze
