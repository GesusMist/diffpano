#!/bin/bash
#SBATCH --job-name=grad-figures
#SBATCH --account=132684267659
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
cd /home/shig/diffpano
bash studies/panorama_metrics/environment.sh -m studies.gradient_blending.figures
bash studies/all_prompts/environment.sh diffpano -m studies.gradient_blending.report
