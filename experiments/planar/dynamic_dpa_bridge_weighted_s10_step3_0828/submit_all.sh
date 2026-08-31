#!/bin/bash
set -euo pipefail

cd "${DIFFPANO_ROOT:-$HOME/diffpano}"
ROOT="experiments/planar/dynamic_dpa_bridge_weighted_s10_step3_0828"

while IFS= read -r config; do
  [[ -z "$config" ]] && continue
  sbatch "$ROOT/submit_job.slurm" "$config"
done < "$ROOT/manifest.txt"
