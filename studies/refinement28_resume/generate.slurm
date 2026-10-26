#!/bin/bash
#SBATCH --job-name=refinement28-resume
#SBATCH --account=142684268708
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --exclude=fc026
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the intended checkout}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec bash studies/all_prompts/environment.sh diffpano -m studies.refinement28_resume.run "$@"
