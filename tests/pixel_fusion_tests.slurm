#!/bin/bash
#SBATCH --job-name=pf-unit
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/shig/diffpano/logs/pf-unit.%j.out
#SBATCH --error=/home/shig/diffpano/logs/pf-unit.%j.err

set -euo pipefail

module purge
module load GCC/13.2.0 OpenMPI/4.1.6 Python/3.11.5 PyTorch/2.7.0
source /scratch/user/shig/virtual_envs/diffpano/bin/activate

cd /home/shig/diffpano

bash -n \
    a.slurm \
    tests/pixel_fusion_tests.slurm \
    tests/pixel_fusion_model_smoke.slurm \
    tests/pixel_fusion_flux_fp32_smoke.slurm \
    tests/projection_diagnostics_report.slurm \
    tests/pixel_fusion_full_erp_smoke.slurm
python -m py_compile \
    output_metadata.py \
    generate_static_wallpaper.py \
    generate_live_wallpaper.py \
    pipelines_ours/pixel_fusion.py \
    pipelines_ours/pipeline_spherical_sana.py \
    pipelines_ours/pipeline_spherical_flux.py
python -m unittest discover -s tests -v
