#!/bin/bash
#SBATCH --job-name=pf-flux-fp32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=/home/shig/diffpano/logs/pf-flux-fp32.%j.out
#SBATCH --error=/home/shig/diffpano/logs/pf-flux-fp32.%j.err

set -euo pipefail

module purge
module load GCC/13.2.0 OpenMPI/4.1.6 Python/3.11.5 PyTorch/2.7.0
source /scratch/user/shig/virtual_envs/diffpano/bin/activate

cd /home/shig/diffpano

export HF_HOME=/scratch/user/shig/SphereDiff/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/home/shig/diffpano/.tmp
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" /home/shig/diffpano/test_outputs/flux_fp32_diagnostics

python generate_static_wallpaper.py --config_add \
    pipeline_cls=SphericalFluxPipeline \
    pretrained_model_name_or_path=black-forest-labs/FLUX.1-dev \
    variant=None \
    mixed_precision=bf16 \
    enable_model_cpu_offload=True \
    call_kwargs.prompt_txt_path=data/prompts/ruins.txt \
    call_kwargs.num_inference_steps=1 \
    call_kwargs.height=512 \
    call_kwargs.width=512 \
    call_kwargs.n_spherical_points=26500 \
    call_kwargs.erp_height=256 \
    call_kwargs.erp_width=512 \
    call_kwargs.pixel_fusion_config.pixel_fusion_enabled=True \
    call_kwargs.pixel_fusion_config.warp_mode=standard \
    call_kwargs.pixel_fusion_config.aggregation_mode=weighted_average \
    call_kwargs.pixel_fusion_config.weight_mode=distance_to_boundary \
    call_kwargs.pixel_fusion_config.reinjection_mode=noise_consistent \
    call_kwargs.pixel_fusion_config.vae_chunk_size=4 \
    call_kwargs.pixel_fusion_config.save_diagnostics=True \
    call_kwargs.pixel_fusion_config.diagnostics_dir=/home/shig/diffpano/test_outputs/flux_fp32_diagnostics \
    save_path=/home/shig/diffpano/test_outputs/pixel_fusion_flux_fp32_smoke
