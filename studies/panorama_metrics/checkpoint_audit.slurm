#!/bin/bash
#SBATCH --job-name=pmetrics-weights
#SBATCH --account=132684267659
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
cd /home/shig/diffpano
exec /sw/eb/sw/Anaconda3/2023.09-0/bin/python -m studies.panorama_metrics.checkpoint_audit
