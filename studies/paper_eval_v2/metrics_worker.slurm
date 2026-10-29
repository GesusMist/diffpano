#!/bin/bash
#SBATCH --job-name=paper-v2-metrics
#SBATCH --account=142684268708
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset OPENROUTER_API_KEY
: "${SLURM_SUBMIT_DIR:?Must run through the recorded shared-resource launcher}"
cd "$SLURM_SUBMIT_DIR"
exec bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.metrics "$@"
