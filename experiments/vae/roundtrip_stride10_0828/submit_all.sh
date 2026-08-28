#!/bin/bash
set -euo pipefail

cd "${DIFFPANO_ROOT:-$HOME/diffpano}"
SCRIPT="experiments/vae/roundtrip_stride10_0828/submit_job.slurm"

for model in sana flux sd2; do
  for roundtrips in 1 2 5 10 20; do
    sbatch "$SCRIPT" "$model" "$roundtrips"
  done
done
