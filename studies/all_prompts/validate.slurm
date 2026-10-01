#!/bin/bash
#SBATCH --job-name=allp21-validate
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
exec bash /home/shig/diffpano/studies/all_prompts/environment.sh diffpano -m studies.all_prompts.validate "$@"
