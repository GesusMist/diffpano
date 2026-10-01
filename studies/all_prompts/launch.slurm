#!/bin/bash
#SBATCH --job-name=allp21-launch
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
exec bash /home/shig/diffpano/studies/all_prompts/environment.sh diffpano -m studies.all_prompts.launch "$@"
