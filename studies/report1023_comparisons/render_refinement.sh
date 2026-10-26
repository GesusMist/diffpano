#!/bin/bash
#SBATCH --job-name=refinement_percent
#SBATCH --account=142684268708
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --chdir=/home/shig/diffpano
#SBATCH --output=/home/shig/diffpano/outputs/10.23report/logs/refinement-percent-%j.log
set -euo pipefail
exec bash studies/panorama_metrics/environment.sh -m studies.report1023_comparisons.redo_refinement --workers 2
