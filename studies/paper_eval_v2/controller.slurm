#!/bin/bash
#SBATCH --account=142684268708
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit through the shared resource journal}"
source studies/paper_eval_v2/clean_environment.sh
unset OPENROUTER_API_KEY
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.controller run "$@"
