#!/bin/bash
#SBATCH --job-name=ccv2-audit
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
bash studies/all_prompts/environment.sh diffpano -m studies.camera_count_v2.audit
bash studies/all_prompts/environment.sh diffpano -m unittest studies.camera_count_v2.tests.test_geometry -v
