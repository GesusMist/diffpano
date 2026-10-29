#!/bin/bash
# Same private loader and persistent accounting as the shared judge worker.
set +x
set -euo pipefail
paper_eval_caption_env="${DIFFPANO_OPENROUTER_ENV_FILE:-${HOME:?}/.config/diffpano/openrouter.env}"
if [[ -e "$paper_eval_caption_env" ]]; then
    if [[ -L "$paper_eval_caption_env" || ! -f "$paper_eval_caption_env" || ! -O "$paper_eval_caption_env" ]]; then
        echo 'Private API configuration must be a regular, user-owned, non-symlink file.' >&2
        exit 2
    fi
    if [[ "$(stat -c '%a' "$paper_eval_caption_env")" != 600 || "$(stat -c '%a' "$(dirname "$paper_eval_caption_env")")" != 700 ]]; then
        echo 'Private API configuration requires file mode600 and directory mode700.' >&2
        exit 2
    fi
    source "$paper_eval_caption_env"
    set +x
fi
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo 'OpenRouter key is absent on this execution host; paid captions are pending.' >&2
    exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
exec bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.captions "$@"
