#!/bin/bash
#SBATCH --account=142684268708
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the audited repository}"
unset OPENROUTER_API_KEY
export PYTHONDONTWRITEBYTECODE=1
exec bash studies/all_prompts/environment.sh diffpano -m studies.paper_eval_v2.baseline_validation all "$@"
