#!/bin/bash
#SBATCH --job-name=diffpano-test
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

module purge
module load GCC/13.2.0 OpenMPI/4.1.6 Python/3.11.5 PyTorch/2.7.0
source /scratch/user/shig/virtual_envs/diffpano/bin/activate
REPO_ROOT="${DIFFPANO_ROOT:-$HOME/diffpano}"
cd "$REPO_ROOT"

bash -n slurm/*.slurm
python -m compileall -q diffpano experiments/legacy_spherical scripts spherediff tests
python -c "import diffpano, scripts.generate"
python -m pip install --dry-run --no-deps --no-build-isolation -e .
python -m scripts.generate --help >/dev/null
python -m scripts.download_models --help >/dev/null
python -m unittest discover -s tests -v
