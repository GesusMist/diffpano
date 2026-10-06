#!/bin/bash
#SBATCH --job-name=pmetrics-assets
#SBATCH --account=132684267659
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
module load WebProxy
cd /home/shig/diffpano
exec /sw/eb/sw/Anaconda3/2023.09-0/bin/python -m studies.panorama_metrics.assets
