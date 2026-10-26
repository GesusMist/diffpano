#!/bin/bash
#SBATCH --job-name=fid-standalone85
#SBATCH --account=142684268708
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:45:00
#SBATCH --output=/home/shig/diffpano/outputs/10.23fid-standalone85/seed0-v1/logs/%x-%j.out
set -euo pipefail
# Remove an inherited venv before the original wrapper purges its libraries.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    fid_old_bin="$VIRTUAL_ENV/bin"
    fid_clean_path=""
    IFS=: read -ra fid_parts <<< "$PATH"
    for fid_part in "${fid_parts[@]}"; do
        if [[ "$fid_part" != "$fid_old_bin" ]]; then
            fid_clean_path="${fid_clean_path:+$fid_clean_path:}$fid_part"
        fi
    done
    export PATH="$fid_clean_path"
    unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT
fi
cd /home/shig/diffpano
exec bash studies/panorama_metrics/environment.sh -m studies.standalone_fid.run
