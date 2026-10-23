#!/bin/bash
#SBATCH --job-name=uniform-faster
#SBATCH --account=142684268708
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/shig/diffpano/outputs/10.19uniform-resume-faster/logs/generation/%x-%A_%a.out
set -euo pipefail
cd /home/shig/diffpano
exec bash studies/all_prompts/environment.sh diffpano -m studies.faster_uniform_resume.dispatch "$@"
