#!/bin/bash
# Remove an inherited virtualenv before module purge. activate_venv invokes
# python3 while discovering its module dependencies.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    paper_eval_old_bin="$VIRTUAL_ENV/bin"
    paper_eval_clean_path=""
    IFS=: read -ra paper_eval_path_parts <<< "$PATH"
    for paper_eval_part in "${paper_eval_path_parts[@]}"; do
        if [[ "$paper_eval_part" != "$paper_eval_old_bin" ]]; then
            paper_eval_clean_path="${paper_eval_clean_path:+$paper_eval_clean_path:}$paper_eval_part"
        fi
    done
    export PATH="$paper_eval_clean_path"
    unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT
fi
