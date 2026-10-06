#!/bin/bash
#SBATCH --job-name=pmetrics-cpu
#SBATCH --account=132684267659
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
cd /home/shig/diffpano
bash studies/panorama_metrics/environment.sh -m studies.panorama_metrics.validate_cpu
exec bash studies/panorama_metrics/environment.sh -m studies.panorama_metrics.cpu
