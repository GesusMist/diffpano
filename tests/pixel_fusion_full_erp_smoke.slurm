#!/bin/bash
#SBATCH --job-name=pf-full-erp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/shig/diffpano/logs/pf-full-erp.%j.out
#SBATCH --error=/home/shig/diffpano/logs/pf-full-erp.%j.err

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
mkdir -p "$TMPDIR" /home/shig/diffpano/test_outputs

python generate_static_wallpaper.py --config_add \
    pipeline_cls=SphericalSanaPipeline \
    pretrained_model_name_or_path=Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers \
    variant=bf16 \
    mixed_precision=bf16 \
    enable_model_cpu_offload=True \
    call_kwargs.prompt_txt_path=data/prompts/ruins.txt \
    call_kwargs.num_inference_steps=1 \
    call_kwargs.height=512 \
    call_kwargs.width=512 \
    call_kwargs.n_spherical_points=2600 \
    call_kwargs.erp_height=2048 \
    call_kwargs.erp_width=4096 \
    call_kwargs.pixel_fusion_config_path=/home/shig/diffpano/configs/pixel_fusion_default.yaml \
    save_path=/home/shig/diffpano/test_outputs/pixel_fusion_full_erp_smoke
