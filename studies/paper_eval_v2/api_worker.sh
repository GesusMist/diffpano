#!/bin/bash
# Private API configuration is loaded only in this worker, never in sbatch args.
set +x
set -euo pipefail
paper_eval_api_env="${DIFFPANO_OPENROUTER_ENV_FILE:-${HOME:?}/.config/diffpano/openrouter.env}"
if [[ -e "$paper_eval_api_env" ]]; then
    if [[ -L "$paper_eval_api_env" || ! -f "$paper_eval_api_env" || ! -O "$paper_eval_api_env" ]]; then
        echo 'Private API configuration must be a regular, user-owned, non-symlink file.' >&2
        exit 2
    fi
    if [[ "$(stat -c '%a' "$paper_eval_api_env")" != 600 || "$(stat -c '%a' "$(dirname "$paper_eval_api_env")")" != 700 ]]; then
        echo 'Private API configuration requires file mode 600 and directory mode 700.' >&2
        exit 2
    fi
    source "$paper_eval_api_env"
    set +x
fi
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo 'OpenRouter key is absent on this execution host; paid execution is pending.' >&2
    exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
exec bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.vlm "$@"
