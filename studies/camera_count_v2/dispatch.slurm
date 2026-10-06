#!/bin/bash
#SBATCH --job-name=ccv2-dispatch
#SBATCH --account=132684267659
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
cd /home/shig/diffpano
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
exec /sw/eb/sw/Anaconda3/2023.09-0/bin/python -m studies.camera_count_v2.launch --dispatch
