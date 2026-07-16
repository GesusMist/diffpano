#!/bin/bash
#SBATCH --job-name=planar-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/home/shig/diffpano/logs/planar-smoke.%j.out
#SBATCH --error=/home/shig/diffpano/logs/planar-smoke.%j.err

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

NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"
PLANAR_FUSION_CONFIG_PATH="${PLANAR_FUSION_CONFIG_PATH:-configs/planar_patch_test.yaml}"
RUN_LABEL="${RUN_LABEL:-planar_patch_sana}"
OUTPUT_PREFIX="/home/shig/diffpano/test_outputs/${RUN_LABEL}_${NUM_INFERENCE_STEPS}step_${SLURM_JOB_ID}"

python generate_static_wallpaper.py --config_add \
    pipeline_cls=PlanarPatchSanaPipeline \
    pretrained_model_name_or_path=Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers \
    variant=bf16 \
    mixed_precision=bf16 \
    enable_model_cpu_offload=False \
    call_kwargs.prompt_txt_path=data/prompts/ruins.txt \
    call_kwargs.num_inference_steps="$NUM_INFERENCE_STEPS" \
    call_kwargs.height=2048 \
    call_kwargs.width=4096 \
    call_kwargs.use_resolution_binning=False \
    call_kwargs.planar_fusion_config_path="$PLANAR_FUSION_CONFIG_PATH" \
    save_path="$OUTPUT_PREFIX"

python - "$OUTPUT_PREFIX" <<'PY'
import glob
import json
import sys

from PIL import Image


prefix = sys.argv[1]
images = sorted(glob.glob(f"{prefix}_*.png"))
if len(images) != 1:
    raise AssertionError(f"Expected one smoke image for {prefix}, found {images}")
with Image.open(images[0]) as image:
    if image.size != (4096, 2048):
        raise AssertionError(f"Expected 4096x2048 output, got {image.size}")
metadata_path = images[0][:-4] + ".json"
with open(metadata_path, "r", encoding="utf-8") as handle:
    metadata = json.load(handle)
if metadata["runtime"]["planar_latent_shape"][-2:] != [64, 128]:
    raise AssertionError(f"Unexpected planar latent shape: {metadata['runtime']['planar_latent_shape']}")
if metadata["runtime"]["planar_patch_count"] != 72:
    raise AssertionError(f"Unexpected planar patch count: {metadata['runtime']['planar_patch_count']}")
print(f"Validated planar smoke image and metadata: {images[0]}")
PY
