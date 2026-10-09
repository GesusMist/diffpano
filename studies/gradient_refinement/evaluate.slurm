#!/bin/bash
#SBATCH --job-name=gradref-metrics
#SBATCH --account=132684267659
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/10.8gradient-refinement-flux/seed0-v1/%x-%j.out
set -euo pipefail
cd /home/shig/diffpano
exec bash studies/panorama_metrics/environment.sh -m studies.gradient_refinement.evaluate
