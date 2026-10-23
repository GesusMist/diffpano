#!/bin/bash
#SBATCH --job-name=perspective-report
#SBATCH --account=142684268708
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/10.16report_images/perspective/provenance/render-%j.out
set -euo pipefail
cd /home/shig/diffpano
exec bash studies/panorama_metrics/environment.sh -m studies.perspective_figures.build "$@"
