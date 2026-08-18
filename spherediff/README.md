# SphereDiff baseline

This directory identifies the SphereDiff-derived/reference path used for baseline comparisons. The spherical geometry and model adapters are shared with `diffpano/` to avoid maintaining divergent copies. Selecting the baseline configuration disables DiffPano pixel fusion and therefore preserves the original SphereDiff denoising and weighted latent write-back path.

DiffPano-specific projection, LPW fusion, VAE residual bridging, reinjection, diagnostics, and exclusive write-back live under `diffpano/`. New research functionality should not be added here.

Run the baseline with:

```bash
python scripts/generate.py --config spherediff/config.yaml
```

`LEGACY_USAGE.md` maps original command arguments to the canonical config.
