#!/bin/bash
#SBATCH --job-name=pf-diag-report
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/home/shig/diffpano/logs/pf-diag-report.%j.out
#SBATCH --error=/home/shig/diffpano/logs/pf-diag-report.%j.err

set -euo pipefail

module purge
module load GCC/13.2.0 OpenMPI/4.1.6 Python/3.11.5 PyTorch/2.7.0
source /scratch/user/shig/virtual_envs/diffpano/bin/activate

cd /home/shig/diffpano

python - <<'PY'
from pathlib import Path

import torch

path = max(Path("test_outputs/diagnostics").glob("pixel_fusion_*.pt"), key=lambda item: item.stat().st_mtime)
payload = torch.load(path, map_location="cpu", weights_only=True)
print(f"diagnostics_file={path}")
for key in (
    "erp_grid_min",
    "erp_grid_max",
    "erp_pixel_center_max_error",
    "perspective_round_trip_mean_error",
    "perspective_round_trip_max_error",
):
    print(f"{key}={payload[key].tolist()}")
PY
