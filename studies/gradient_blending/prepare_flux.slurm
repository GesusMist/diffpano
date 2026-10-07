#!/bin/bash
#SBATCH --job-name=grad-flux-audit
#SBATCH --account=132684267659
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
cd /home/shig/diffpano
export DIFFPANO_GRADIENT_BACKEND=flux
exec bash studies/all_prompts/environment.sh diffpano -m studies.gradient_blending.prepare_flux
