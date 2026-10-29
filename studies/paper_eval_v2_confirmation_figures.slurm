#!/bin/bash
#SBATCH --account=142684268708
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
source studies/paper_eval_v2/clean_environment.sh
unset OPENROUTER_API_KEY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2_confirmation_figures
