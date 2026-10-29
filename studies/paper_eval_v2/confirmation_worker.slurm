#!/bin/bash
#SBATCH --account=142684268708
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --exclude=fc026
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit through the shared resource journal}"
source studies/paper_eval_v2/clean_environment.sh
unset OPENROUTER_API_KEY
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.confirmation_worker "$@"
