# DiffPano

DiffPano generates 360° images with pretrained diffusion models by denoising perspective views sampled from a global spherical latent representation, fusing their predicted-clean RGB views in ERP space, and writing scheduler-consistent corrections back to the sphere. The implementation supports SANA and FLUX image backends; SphereDiff-derived video adapters remain available for HunyuanVideo and LTX-Video.

DiffPano builds on the spherical latent representation introduced by [SphereDiff](https://github.com/pmh9960/SphereDiff). The shared baseline/reference path is documented under [`spherediff/`](spherediff/README.md); DiffPano-specific projection, LPW, fusion, VAE bridging, reinjection, and write-back code lives under [`diffpano/`](diffpano/).

## Algorithm

```text
config → spherical x_t → perspective latent views → model prediction → predicted-clean x0
       → VAE decode → projection / LPW → RGB fusion → VAE residual bridge
       → noise-consistent reinjection → exclusive spherical write-back → x_{t-1}
```

The VAE bridge preserves the current identity-relative correction: `encode(fused_rgb) - encode(original_rgb)` is applied to the original predicted-clean latent representation. Projection and fusion geometry remains FP32.

## Repository layout

- `diffpano/`: reusable DiffPano package and model adapters
- `spherediff/`: SphereDiff baseline/reference namespace and configuration
- `experiments/planar/`: isolated no-warp planar ablation
- `scripts/`: user-facing entrypoints only
- `slurm/`: cluster resources, environment setup, and command invocation
- `prompts/`: five-line directional prompt files
- `tests/`: unit and regression tests
- `docs/USAGE.md`: full configuration and operating guide

## Installation

```bash
conda create -n diffpano python=3.10
conda activate diffpano
# Install the PyTorch build appropriate for your CUDA environment first.
pip install -r requirements.txt
pip install -e .
```

## Model preparation

Set either `model.path` (preferred when non-null) or `model.id` in `config.yaml`, then optionally prefetch a remote model:

```bash
python scripts/download_models.py --config config.yaml
```

Gated Hugging Face models require an authenticated account with access.

## Generate

Edit [`config.yaml`](config.yaml), then run:

```bash
python scripts/generate.py --config config.yaml
```

A run is saved as `outputs/<experiment-name>/<run-id>/` with `result.png` or `result.mp4`, the fully resolved `config.yaml`, `metadata.json`, `run.log`, and an optional `intermediates/` directory.

For SLURM:

```bash
mkdir -p logs
sbatch slurm/generate_a100.slurm config.yaml
```

## Experiments and baselines

Run the planar no-warp ablation separately:

```bash
python scripts/planar_test.py --config experiments/planar/config.yaml
```

Run the SphereDiff baseline path with pixel fusion disabled:

```bash
python scripts/generate.py --config spherediff/config.yaml
```

See [`docs/USAGE.md`](docs/USAGE.md) for all fields, debugging, testing, troubleshooting, and adding a backend.
