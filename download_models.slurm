#!/bin/bash
#SBATCH --job-name=spherediff_download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/shig/SphereDiff/logs/%x.%j.out
#SBATCH --error=/home/shig/SphereDiff/logs/%x.%j.err

set -euo pipefail

if declare -F deactivate >/dev/null 2>&1; then
    deactivate || true
fi
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

module purge
module load WebProxy
source activate_venv spherediff

cd "$HOME/SphereDiff"

mkdir -p "$SCRATCH/SphereDiff/hf_cache"
mkdir -p "$SCRATCH/SphereDiff/tmp"
mkdir -p "$HOME/SphereDiff/logs"

export HF_HOME="$SCRATCH/SphereDiff/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$SCRATCH/SphereDiff/tmp"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "HF_HOME=$HF_HOME"
echo "Python: $(which python)"
python -c "import sys, torch; print(sys.executable); print('torch', torch.__version__); print('cuda', torch.version.cuda)"

# Default downloads only the SANA model used by a.slurm.
# Override at submit time, for example:
#   sbatch --export=ALL,MODELS="SANA LTX" download_models.slurm
#   sbatch --export=ALL,MODELS="ALL" download_models.slurm
MODELS="${MODELS:-SANA}"

python download_models.py $MODELS
