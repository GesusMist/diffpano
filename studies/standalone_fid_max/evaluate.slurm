#!/bin/bash
#SBATCH --job-name=fid-max-blending
#SBATCH --account=142684268708
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:45:00
#SBATCH --output=/home/shig/diffpano/outputs/10.23fid-standalone85/max-blending-20views/logs/%x-%j.out
set -euo pipefail
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    maxfid_old_bin="$VIRTUAL_ENV/bin"
    maxfid_clean_path=""
    IFS=: read -ra maxfid_parts <<< "$PATH"
    for maxfid_part in "${maxfid_parts[@]}"; do
        if [[ "$maxfid_part" != "$maxfid_old_bin" ]]; then
            maxfid_clean_path="${maxfid_clean_path:+$maxfid_clean_path:}$maxfid_part"
        fi
    done
    export PATH="$maxfid_clean_path"
    unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT
fi
cd /home/shig/diffpano
exec bash studies/panorama_metrics/environment.sh -m studies.standalone_fid_max.run
