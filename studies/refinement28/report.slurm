#!/bin/bash
#SBATCH --job-name=refinement28-report
#SBATCH --account=142684268708
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:20:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the intended checkout}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec bash studies/all_prompts/environment.sh diffpano -m studies.refinement28.report
