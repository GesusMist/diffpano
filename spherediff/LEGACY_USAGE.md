# SphereDiff and legacy DiffPano commands

The historical spherical-latent implementation is no longer the main CLI. Run it explicitly through its archived module:

```bash
python -m experiments.legacy_spherical.generate --config spherediff/config.yaml
```

The checked-in baseline config uses SANA and `fusion.enabled: false`, selecting SphereDiff's persistent Fibonacci-sphere latent with center-weighted write-back. To reproduce the former DiffPano experiment instead, copy that legacy-format config and enable its archived pixel-fusion/reinjection settings.

Main ERP-RGB experiments use:

```bash
python scripts/generate.py --config config.yaml
```
