#!/bin/bash
#SBATCH --job-name=gradref-unit
#SBATCH --account=132684267659
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/10.8gradient-refinement-flux/seed0-v1/%x-%j.out
set -euo pipefail
cd /home/shig/diffpano
exec bash studies/all_prompts/environment.sh diffpano "$@"
