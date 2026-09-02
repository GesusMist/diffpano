# SphereDiff baseline/reference

SphereDiff remains available as the spherical-latent baseline. It maintains a persistent latent tensor attached to Fibonacci sphere points, dynamically gathers perspective patches, denoises them, and center-weighted-writes them back to the sphere.

This is distinct from main DiffPano, which stores only ERP RGB globally. The baseline implementation is preserved in `experiments/legacy_spherical/diffpano_legacy/`; this directory supplies the reference namespace and baseline configuration.

```bash
python -m experiments.legacy_spherical.generate --config spherediff/config.yaml
```

Set `fusion.enabled: false` in the baseline config. Legacy spherical DiffPano can instead enable its temporary RGB fusion/reinjection branch for historical comparison.
