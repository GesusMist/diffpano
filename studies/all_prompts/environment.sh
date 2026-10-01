#!/bin/bash
set -euo pipefail
# The submitting process may have inherited a venv whose Python cannot run
# after module purge. Strip only that inherited venv from PATH first.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    sweep_old_bin="$VIRTUAL_ENV/bin"
    sweep_clean_path=""
    IFS=: read -ra sweep_parts <<< "$PATH"
    for sweep_part in "${sweep_parts[@]}"; do
        if [[ "$sweep_part" != "$sweep_old_bin" ]]; then
            sweep_clean_path="${sweep_clean_path:+$sweep_clean_path:}$sweep_part"
        fi
    done
    export PATH="$sweep_clean_path"
    unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT
fi
sweep_environment="$1"
shift
module purge
module load WebProxy
source activate_venv "$sweep_environment"
cd /home/shig/diffpano
if [[ "$sweep_environment" == "spherediff" ]]; then
    export HF_HOME="$SCRATCH/SphereDiff/hf_cache"
else
    export HF_HOME="$SCRATCH/diffpano/hf_cache"
fi
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$SCRATCH/diffpano/tmp"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
exec python "$@"
