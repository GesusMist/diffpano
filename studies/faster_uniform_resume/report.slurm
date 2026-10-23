#!/bin/bash
#SBATCH --job-name=uniform-faster-report
#SBATCH --account=142684268708
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/shig/diffpano/outputs/10.19uniform-resume-faster/logs/report/%x-%j.out
set -euo pipefail
cd /home/shig/diffpano
exec bash studies/panorama_metrics/environment.sh -m studies.faster_uniform_resume.report
