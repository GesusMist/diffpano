#!/bin/bash
#SBATCH --job-name=grad-flux-scenes-finish
#SBATCH --account=132684267659
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
cd /home/shig/diffpano
export DIFFPANO_GRADIENT_BACKEND=flux DIFFPANO_GRADIENT_SUITE=flux-scenes20
exec python3 -m studies.gradient_blending.finish_scenes
