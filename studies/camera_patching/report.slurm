#!/bin/bash
#SBATCH --job-name=campatch-report
#SBATCH --account=132684267659
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
exec bash /home/shig/diffpano/studies/all_prompts/environment.sh diffpano -m studies.camera_patching.report --require-complete
