#!/bin/bash
#SBATCH --account=142684268708
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=fc026
set -euo pipefail
exec bash studies/all_prompts/environment.sh diffpano -m studies.paper_eval_v2.native run "$@"
