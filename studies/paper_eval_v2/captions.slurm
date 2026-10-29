#!/bin/bash
#SBATCH --account=142684268708
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set +x
set -euo pipefail
source studies/paper_eval_v2/clean_environment.sh
exec bash studies/paper_eval_v2/captions_worker.sh run "$@"
