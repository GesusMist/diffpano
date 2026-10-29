#!/bin/bash
#SBATCH --account=142684268708
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
set +x
set -euo pipefail
source studies/paper_eval_v2/clean_environment.sh
if [[ "${1:-}" == run ]]; then
    variant_judge_env="${DIFFPANO_OPENROUTER_ENV_FILE:-${HOME:?}/.config/diffpano/openrouter.env}"
    [[ -f "$variant_judge_env" && ! -L "$variant_judge_env" && -O "$variant_judge_env" ]]
    [[ "$(stat -c '%a' "$variant_judge_env")" == 600 ]]
    [[ "$(stat -c '%a' "$(dirname "$variant_judge_env")")" == 700 ]]
    source "$variant_judge_env"
    set +x
    [[ -n "${OPENROUTER_API_KEY:-}" ]]
fi
exec bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2_variant_judges "$@"
