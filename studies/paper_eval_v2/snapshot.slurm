#!/bin/bash
#SBATCH --account=142684268708
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
set -euo pipefail
source studies/paper_eval_v2/clean_environment.sh
unset OPENROUTER_API_KEY
exec bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.snapshot "$@"
