#!/bin/bash
#SBATCH --job-name=uniform-faster-eval
#SBATCH --account=142684268708
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:45:00
#SBATCH --output=/home/shig/diffpano/outputs/10.19uniform-resume-faster/logs/evaluation/%x-%j.out
set -euo pipefail
cd /home/shig/diffpano
exec bash studies/panorama_metrics/environment.sh -m studies.faster_uniform_resume.evaluate --allow-partial
