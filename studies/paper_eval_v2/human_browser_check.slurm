#!/bin/bash
#SBATCH --account=142684268708
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
source studies/paper_eval_v2/clean_environment.sh
exec bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.human_browser_check "$@"
