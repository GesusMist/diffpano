# Legacy spherical DiffPano

This directory preserves the pre-refactor image implementation for comparison. Its persistent state is a Fibonacci-sphere latent tensor. The optional legacy DiffPano branch decodes predicted-clean views, temporarily fuses RGB in ERP space, applies an identity-relative VAE bridge and scheduler-consistent correction, then writes patches back to the sphere.

It is deliberately isolated from the new `diffpano` package. New ERP-RGB orchestration must not import anything from `experiments.legacy_spherical.diffpano_legacy`.

The image baseline can be launched with:

```bash
python -m experiments.legacy_spherical.generate --config spherediff/config.yaml
```

Keep `fusion.enabled: false` for the original SphereDiff weighted spherical-latent baseline. The archived code is retained as a research reference and receives only compatibility fixes.
