#!/bin/bash
#SBATCH --account=142684268708
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:45:00
set -euo pipefail
module load WebProxy
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec python3 -m studies.paper_eval_v2.dataset_download "$@"
