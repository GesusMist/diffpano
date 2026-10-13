#!/bin/bash
#SBATCH --job-name=ournew-metrics-cpu
#SBATCH --account=132684267659
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/10.9ournew-flux28/seed0-v1/logs/validation/%x-%j.out
set -euo pipefail
cd /home/shig/diffpano
exec bash studies/panorama_metrics/environment.sh -m studies.ournew_comparison.metric_validate
