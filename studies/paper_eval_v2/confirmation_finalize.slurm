#!/bin/bash
#SBATCH --account=142684268708
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit through the shared resource journal}"
source studies/paper_eval_v2/clean_environment.sh
unset OPENROUTER_API_KEY
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.controller finalize "$@"
