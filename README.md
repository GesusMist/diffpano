# DiffPano

DiffPano generates 360° panoramas while maintaining one persistent global state: an equirectangular RGB canvas, `erp_rgb_t` with shape `[B, 3, H_erp, W_erp]`. The sphere is used only to convert coordinates between that canvas and ordinary perspective cameras. No feature, noise, or diffusion state is stored on spherical points in the main implementation.

## Algorithm

Every perspective view at one timestep reads the same frozen canvas. Each local model call advances its view by exactly one scheduler step; proposals are then warped back and fused together in RGB space.

```text
                         ERP RGB I_t
                              │
            ┌─────────────────┼─────────────────┐
            ↓                 ↓                 ↓
      perspective A     perspective B     perspective C
            ↓                 ↓                 ↓
       VAE + local       VAE + local       VAE + local
      diffusion step    diffusion step    diffusion step
            ↓                 ↓                 ↓
       RGB proposal       RGB proposal       RGB proposal
            │                 │                 │
            └─────────────────┼─────────────────┘
                              ↓
                    warp back + RGB fusion
                              ↓
                         ERP RGB I_t-1
```

This is a synchronous (Jacobi-style) update. Camera order cannot change a deterministic result, and uncovered ERP pixels retain their previous value. The final ERP canvas is the panorama; there is no spherical-latent rendering pass.

## Supported research options

- Camera covers: `spherediff_fixed` (the original dense-equator 89-view cover) and `spherediff_rotated` (one deterministic global 3D rotation per step).
- Warping: exact inverse ray-based `standard` projection and RGB Laplacian Pyramid Warping (`lpw`).
- Spatial interpolation: independently configurable `nearest` or `bilinear` for ERP→perspective and perspective→ERP.
- LPW: `jacobian` or `none` LOD, with separate `nearest` or `linear` LOD interpolation.
- Fusion: `average`, `weighted_average`, or `detail_preserving_average`.
- Weights: `uniform`, `cosine`, `gaussian`, `distance_to_boundary`, and SphereDiff-style `spherediff_center`.
- Initialization: global `erp_rgb_noise` or one-time `latent_native_bootstrap`.
- Local denoisers: SANA, FLUX, and Stable Diffusion 2. The global API is RGB-only and can accept a future direct-pixel `ViewDenoiser` without changes to geometry or fusion.

Projection, LPW, accumulation, and fusion run in FP32. Latent packing, scaling/shift conventions, model precision, prompt embeddings, and scheduler calls stay inside each backend adapter. VAE posterior encoding uses mode/mean deterministically.

## Installation and generation

Install a CUDA-compatible PyTorch build first, then:

```bash
pip install -r requirements.txt
pip install -e .
python scripts/generate.py --config config.yaml
```

The typed loader rejects stale spherical-state fields and unsupported option values. Runs are stored as `outputs/<group>/<experiment>/<run>`; a null output group becomes the current `YYYY-MM-DD`. See [`config.yaml`](config.yaml) and [`docs/USAGE.md`](docs/USAGE.md) for the complete option matrix.

## Repository layout

```text
diffpano/                         new persistent-ERP-RGB implementation
diffpano/pipelines/               local SANA/FLUX/SD2 ViewDenoiser adapters
experiments/legacy_spherical/     archived spherical-latent DiffPano path
spherediff/                       SphereDiff baseline namespace/config
scripts/generate.py               new main generation entrypoint
tests/                            CPU geometry, fusion, LPW, and sync tests
```

Run the archived SphereDiff baseline separately:

```bash
python -m experiments.legacy_spherical.generate --config spherediff/config.yaml
```

## Intentional theoretical caveats

This architecture exposes rather than hides three research mismatches:

1. `VAE_encode(random_RGB)` is not guaranteed to follow the pretrained model's native Gaussian latent prior.
2. Nearest projection copies samples, while bilinear projection changes Gaussian variance and introduces spatial correlation (although a linear combination of Gaussian samples remains Gaussian).
3. Generally `encode(decode(z)) != z`; cross-view RGB fusion moves the local latent path farther from vanilla latent diffusion.

These are explicit initialization/interpolation variables. The main path must not reintroduce a persistent spherical latent to avoid them.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite uses mocked denoising and requires no model download. It covers the ERP seam, poles, exact nearest/bilinear behavior, irregular inverse footprints, round trips, weights/DPA, LPW/LOD, rigid camera-cover rotation, full coverage, synchronous camera-order invariance, and the no-spherical-state architecture rule.
