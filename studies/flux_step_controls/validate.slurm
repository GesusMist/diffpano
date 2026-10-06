#!/bin/bash
#SBATCH --job-name=fsctrl-validate
#SBATCH --account=132684267659
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err
set -euo pipefail
exec bash /home/shig/diffpano/studies/all_prompts/environment.sh diffpano -m studies.flux_step_controls.validate
