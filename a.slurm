#!/bin/bash
#SBATCH --job-name=diffpano
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/shig/diffpano/logs/%x.%j.out
#SBATCH --error=/home/shig/diffpano/logs/%x.%j.err

set -euo pipefail

if declare -F deactivate >/dev/null 2>&1; then
    deactivate || true
fi
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

module purge
module load WebProxy
source activate_venv diffpano

cd "$HOME/diffpano"

mkdir -p "$SCRATCH/diffpano/outputs"
mkdir -p "$SCRATCH/SphereDiff/hf_cache"
mkdir -p "$SCRATCH/diffpano/tmp"

export HF_HOME="$SCRATCH/SphereDiff/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$SCRATCH/diffpano/tmp"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "SCRATCH=$SCRATCH"
nvidia-smi

# Defaults are chosen for a relatively small single-A100 smoke run.
# Override at submit time, for example:
#   sbatch --export=ALL,PIPELINE=FLUX,PROMPT_NAME=ruins a.slurm
#   sbatch --export=ALL,PIPELINE=LTX,PROMPT_NAME=aurora a.slurm
PIPELINE="${PIPELINE:-SANA}"
PROMPT_NAME="${PROMPT_NAME:-ruins}"
SUBDIR="${SUBDIR:-slurm_${SLURM_JOB_ID}}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
ENABLE_MODEL_CPU_OFFLOAD="${ENABLE_MODEL_CPU_OFFLOAD:-True}"
PIXEL_FUSION_CONFIG_PATH="${PIXEL_FUSION_CONFIG_PATH:-}"

case "$PIPELINE" in
    SANA)
        SCRIPT="generate_static_wallpaper.py"
        TASK_NAME="StaticWallpapers"
        PIPELINE_CLS="SphericalSanaPipeline"
        PRETRAINED_MODEL="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
        VARIANT="bf16"
        N_SPHERICAL_POINTS="${N_SPHERICAL_POINTS:-2600}"
        ;;
    FLUX)
        SCRIPT="generate_static_wallpaper.py"
        TASK_NAME="StaticWallpapers"
        PIPELINE_CLS="SphericalFluxPipeline"
        PRETRAINED_MODEL="black-forest-labs/FLUX.1-dev"
        VARIANT="None"
        N_SPHERICAL_POINTS="${N_SPHERICAL_POINTS:-26500}"
        ;;
    LTX)
        SCRIPT="generate_live_wallpaper.py"
        TASK_NAME="LiveWallpapers"
        PIPELINE_CLS="SphericalLTXPipeline"
        PRETRAINED_MODEL="a-r-r-o-w/LTX-Video-0.9.1-diffusers"
        VARIANT="None"
        N_SPHERICAL_POINTS="${N_SPHERICAL_POINTS:-2600}"
        ;;
    HUNYUAN)
        SCRIPT="generate_live_wallpaper.py"
        TASK_NAME="LiveWallpapers"
        PIPELINE_CLS="SphericalHunyuanVideoPipeline"
        PRETRAINED_MODEL="hunyuanvideo-community/HunyuanVideo"
        VARIANT="None"
        N_SPHERICAL_POINTS="${N_SPHERICAL_POINTS:-7000}"
        ;;
    *)
        echo "Unknown PIPELINE='$PIPELINE'. Use SANA, FLUX, LTX, or HUNYUAN." >&2
        exit 2
        ;;
esac

PROMPT_TXT_PATH="data/prompts/${PROMPT_NAME}.txt"
SAVE_PATH="$SCRATCH/diffpano/outputs/${TASK_NAME}/${PIPELINE_CLS}/${PROMPT_NAME}"

if [[ ! -f "$PROMPT_TXT_PATH" ]]; then
    echo "Prompt file not found: $PROMPT_TXT_PATH" >&2
    exit 2
fi

echo "SCRIPT=$SCRIPT"
echo "PIPELINE=$PIPELINE"
echo "PIPELINE_CLS=$PIPELINE_CLS"
echo "PRETRAINED_MODEL=$PRETRAINED_MODEL"
echo "PROMPT_TXT_PATH=$PROMPT_TXT_PATH"
echo "SAVE_PATH=$SAVE_PATH"
echo "N_SPHERICAL_POINTS=$N_SPHERICAL_POINTS"
echo "PIXEL_FUSION_CONFIG_PATH=${PIXEL_FUSION_CONFIG_PATH:-disabled}"

python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('device count', torch.cuda.device_count())"

CONFIG_ARGS=(
    "pipeline_cls=${PIPELINE_CLS}"
    "pretrained_model_name_or_path=${PRETRAINED_MODEL}"
    "variant=${VARIANT}"
    "mixed_precision=${MIXED_PRECISION}"
    "enable_model_cpu_offload=${ENABLE_MODEL_CPU_OFFLOAD}"
    "call_kwargs.n_spherical_points=${N_SPHERICAL_POINTS}"
    "call_kwargs.prompt_txt_path=${PROMPT_TXT_PATH}"
    "save_path=${SAVE_PATH}"
)

if [[ -n "$PIXEL_FUSION_CONFIG_PATH" ]]; then
    if [[ "$PIPELINE" != "SANA" && "$PIPELINE" != "FLUX" ]]; then
        echo "Pixel fusion is currently wired only for SANA and FLUX." >&2
        exit 2
    fi
    if [[ ! -f "$PIXEL_FUSION_CONFIG_PATH" ]]; then
        echo "Pixel-fusion config not found: $PIXEL_FUSION_CONFIG_PATH" >&2
        exit 2
    fi
    CONFIG_ARGS+=("call_kwargs.pixel_fusion_config_path=${PIXEL_FUSION_CONFIG_PATH}")
fi

python "$SCRIPT" --config_add "${CONFIG_ARGS[@]}"
