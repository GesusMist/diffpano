# DiffPano

DiffPano generates 360° panoramas with a selectable global ERP pipeline. The original `erp_rgb_state` mode persists an equirectangular three-channel diffusion/RGB state. The new `erp_x0_consensus` mode persists only fused predicted-clean ERP RGB. Both have shape `[B, 3, H_erp, W_erp]`; the sphere is used only for coordinate conversion between the ERP and ordinary perspective cameras.

## Original state semantics (`erp_rgb_state`)

The same synchronous ERP geometry and fusion code supports two deliberately different local-model families:

```text
SANA / FLUX / SD2 no-sphere          PixelDiT no-sphere

ERP RGB                              ERP pixel diffusion state x_t
  → perspective RGB                    → perspective pixel state x_t
  → VAE encode                         → PixelDiT flow prediction
  → one latent diffusion step          → one first-order flow step
  → VAE decode                         → perspective pixel state x_next
  → ERP fusion                         → ERP fusion
```

PixelDiT has no VAE and no latent representation. Its persistent ERP tensor is the actual floating-point pixel diffusion state expected by the model, initialized directly with an unclamped Gaussian. This removes the RGB-to-latent boundary from the architecture under test without changing its camera cover, projection, or fusion.

Every view at one timestep reads the same frozen ERP source. Each model call advances exactly one perspective state by one scheduler interval; all proposals are then inverse-warped and fused once. This is a synchronous Jacobi-style update, so camera order cannot change a deterministic result and uncovered pixels retain their previous value.

## Predicted-clean consensus (`erp_x0_consensus`)

This second pipeline never persists a noisy diffusion state. It samples one backend-native Gaussian noise tensor per camera slot before the loop and reuses that tensor bit-for-bit at every timestep. The first timestep starts from native noise. Every later timestep projects the frozen clean ERP into each camera, encodes it only when the backend is latent, applies the backend scheduler's exact forward noising equation with the fixed camera noise, predicts the clean state, decodes clean RGB, inverse-warps it, and fuses all clean proposals.

Noise is never stored in ERP space and never passes through projection or fusion. PixelDiT stays in native RGB throughout; SANA, FLUX, and SD2 use their existing loaded VAE and model objects without duplicate loading. See [docs/USAGE.md](docs/USAGE.md) for the exact scheduler equations and invariants.

`erp_x0_consensus` never fuses noisy diffusion states. ERP synchronization is performed exclusively on predicted-clean RGB images.

```text
SANA / FLUX / SD2: clean ERP RGB -> clean VAE latent -> fixed native
noise at the current level -> model -> predicted-clean latent -> clean RGB

PixelDiT: clean ERP RGB -> fixed native pixel noise at the current shifted
flow time -> upstream model_fn -> predicted-clean RGB (no VAE)
```

## PixelDiT provenance and solver

The adapter uses the official NVIDIA PixelDiT model, checkpoint loader, Gemma-2 setup, CHI prompt preprocessing, classifier-free guidance, and flow schedule from the pinned upstream commit `41f73006ae532b0b41fee72b181dc22891a5a01a`. The checked-in baseline uses the official stage-3 1024px configuration and `pixeldit_t2i_v1.pth` checkpoint.

Official single-image PixelDiT normally uses second-order multistep FlowDPM-Solver. DiffPano intentionally calls the upstream `model_fn` and `dpm_solver_first_update` primitive for one stateless first-order update per synchronized panorama interval, because ERP fusion changes the state and invalidates ordinary per-view multistep history. The official order-1 and order-2 complete samplers remain available only in the standalone validation script.

## Installation

Install a CUDA-compatible PyTorch build first, then install the normal project:

```bash
pip install -r requirements.txt
pip install -e .
```

For PixelDiT, install the pinned official checkout and the extra reference dependencies:

```bash
bash scripts/setup_pixeldit.sh
pip install -r requirements-pixeldit.txt
```

The PixelDiT model and Gemma-2 text encoder download from Hugging Face on first use. To prepare only the checkpoint:

```bash
python scripts/download_models.py --config configs/pixeldit_standard_average.yaml
```

## Generation and validation

Run the recommended native-pixel baseline with:

```bash
python scripts/generate.py --config configs/pixeldit_standard_average.yaml
```

Run predicted-clean ERP consensus with any supported backend:

```bash
python scripts/generate.py --config configs/x0_consensus_sana.yaml
python scripts/generate.py --config configs/x0_consensus_flux.yaml
python scripts/generate.py --config configs/x0_consensus_sd2.yaml
python scripts/generate.py --config configs/x0_consensus_pixeldit.yaml
```

Validate the integration in increasing geometric complexity:

```bash
# DiffPano order 1 versus official order 1, same noise/schedule/prompt.
python scripts/pixeldit_single_view_test.py --save-official-order-two

# Complete schedule through ViewDenoiser, without projection or fusion.
python scripts/pixeldit_full_image_test.py

# One ERP perspective projection, model step sequence, and inverse projection.
python scripts/pixeldit_one_view_erp_test.py

# Small 89-camera standard-warp/plain-average smoke run.
python scripts/generate.py --config configs/pixeldit_smoke.yaml
```

The real-checkpoint scripts are intentionally separate from the CPU unit suite. See [docs/USAGE.md](docs/USAGE.md) for equations, configuration, diagnostics, and expected validation behavior.

## Supported research options

- Camera covers: `spherediff_fixed` and deterministic per-step `spherediff_rotated`.
- Warping: inverse ray-based `standard` projection and RGB Laplacian Pyramid Warping (`lpw`).
- Independent nearest or bilinear sampling for ERP-to-view and view-to-ERP.
- Fusion: `average`, `weighted_average`, or `detail_preserving_average` with all existing weights.
- Initialization: `erp_rgb_noise` and `latent_native_bootstrap` for latent-diffusion backends; direct unclamped `pixel_gaussian` for PixelDiT.
- Local denoisers: SANA, FLUX, Stable Diffusion 2, and PixelDiT.

Projection, LPW, accumulation, and fusion run in FP32 and never clamp the evolving state. Model precision and conditioning stay inside each adapter. Only preview/final image conversion maps values to a displayable range.

## Repository layout

```text
diffpano/                         persistent ERP-RGB/pixel-state implementation
diffpano/erp_x0_pipeline.py       predicted-clean ERP consensus pipeline
diffpano/noise.py                 fixed backend-native per-camera noise bank
diffpano/pipelines/pixeldit.py    lazy official PixelDiT adapter
diffpano/pipelines/pixeldit_solver.py  shifted schedule and first-order step
configs/pixeldit_*.yaml           smoke and recommended baseline experiments
scripts/pixeldit_*_test.py        staged real-checkpoint validation
scripts/x0_consensus_*_test.py    backend-neutral clean-consensus validation
experiments/legacy_spherical/     archived spherical-latent DiffPano path
spherediff/                       SphereDiff baseline namespace/config
```

Run the archived SphereDiff baseline separately:

```bash
python -m experiments.legacy_spherical.generate --config spherediff/config.yaml
```

## Intentional theoretical caveats

For latent-diffusion backends, `VAE_encode(random_RGB)` is not guaranteed to follow the native Gaussian prior and generally `encode(decode(z)) != z`. PixelDiT removes those two mismatches. Both families still expose geometry effects: nearest projection copies samples, while bilinear projection and overlap averaging can shrink variance and introduce spatial correlation. Per-step pixel-state diagnostics and `scripts/test_pixel_state_warp_statistics.py` quantify that effect.

## Tests

```bash
python -m unittest discover -s tests -v
```

The CPU suite requires no model download. It covers geometry/fusion behavior, synchronous camera-order invariance, direct PixelDiT Gaussian initialization, official flow-schedule and order-1 primitive equivalence, directional conditioning, actual view metadata, and projection-only state statistics.
