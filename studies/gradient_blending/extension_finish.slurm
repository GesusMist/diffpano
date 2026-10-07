#!/bin/bash
#SBATCH --job-name=grad-extension-report
#SBATCH --account=132684267659
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
cd /home/shig/diffpano
bash studies/all_prompts/environment.sh diffpano -m studies.gradient_blending.four_model_report
bash studies/all_prompts/environment.sh diffpano -m studies.gradient_blending.extension_status
