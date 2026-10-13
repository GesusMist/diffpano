#!/bin/bash
#SBATCH --job-name=gradref-report
#SBATCH --account=132684267659
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/10.8gradient-refinement-flux/seed0-v1/logs/reports/%x-%j.out
set -euo pipefail
cd /home/shig/diffpano
bash studies/panorama_metrics/environment.sh -m studies.gradient_refinement.figures
python3 -m studies.gradient_refinement.report
