# DiffPano ERP-RGB usage

## Main invariant

`diffpano.erp_pipeline.ERPRGBPipeline` carries only an FP32 tensor shaped `[B,3,H_erp,W_erp]` between global diffusion timesteps. A view is projected from a frozen source ERP, VAE-encoded deterministically, advanced through one backend scheduler step, decoded, inverse-projected to its irregular ERP footprint, and accumulated. Fusion occurs once after the complete cover.

The new package does not import the archived reinjection, owner-map, spherical write-back, Fibonacci latent sampling, or final spherical rendering modules.

## Prompt format

`prompt.path` may contain one global non-empty line or five directional lines: north, upper/equatorial, equator, lower/equatorial, and south. The five lines expand to 20 longitude/latitude anchors. SANA/FLUX text embeddings are computed once before denoising and indexed by camera direction thereafter.

## Configuration

The checked-in [`config.yaml`](../config.yaml) spells out every accepted field. Unknown fields are rejected.

### Camera sampling

- `sampling.strategy: spherediff_fixed` repeats SphereDiff's 89-view, 80° dense-equator cover at every step.
- `sampling.strategy: spherediff_rotated` applies one seed-deterministic SO(3) rotation to the whole cover per step. Relative angles and overlap are preserved; cameras are not independently jittered.
- `view.height`, `view.width`, `view.fov_x`, and `view.fov_y` define perspective images and pinhole rays.

### Projection and LPW

`warp.mode` is `standard` or `lpw`. Standard projection uses inverse resampling in both directions. ERP longitude wraps periodically; pole reflection includes the required half-panorama turn.

Spatial interpolation is independent:

```yaml
warp:
  erp_to_perspective:
    interpolation: nearest
  perspective_to_erp:
    interpolation: bilinear
```

LPW adds `levels`, `lod_mode: jacobian|none`, `lod_interpolation: nearest|linear`, and `vertical_padding_mode: reflect|replicate`. Spatial nearest chooses a source pixel; LOD nearest chooses a pyramid level. They are separate controls.

### RGB fusion

`fusion.mode` selects `average`, `weighted_average`, or `detail_preserving_average`. Weighted modes support `uniform`, `cosine`, `gaussian`, `distance_to_boundary`, and `spherediff_center`; the last uses `exp(-distance_from_center / spherediff_temperature)` in perspective space and warps that confidence alongside RGB.

DPA uses `alpha`, `power`, and `epsilon`; `alpha: 0` reduces to ordinary weighted averaging. `uncovered_mode: keep_previous` is the required safe default.

### Initialization

`erp_rgb_noise` samples one global Gaussian RGB field. `mean`, `std`, and optional `clamp`/range are explicit because arbitrary Gaussian values are not automatically valid VAE images.

`latent_native_bootstrap` samples independent native Gaussian latents for the initial camera cover, decodes and warps them to ERP, and fuses once. After bootstrap, the ERP RGB tensor is the only persistent state.

### Backends and scheduler assumptions

`model.pipeline` is `sana` or `flux`. Both adapters explicitly perform deterministic VAE encode, model forward/guidance, one scheduler call, and VAE decode. FLUX packing/unpacking and shifts remain inside its adapter. Multistep solvers are rejected: reconstructing each moving RGB patch does not provide valid shared solver history. Scheduler call state is reset for independent same-timestep views.

### Performance and diagnostics

`performance.view_batch_size` batches same-sized local view work while retaining synchronous ERP semantics. `performance.vae_chunk_size` limits VAE batches. Geometry grids, masks, ERP rays, weights, and LOD maps are cached by camera configuration. Only streaming full-ERP accumulators persist within a step; an `N_views × ERP` stack is never retained.

Debug output is off by default. Enable `debug.enabled` and individual save flags for ERP frames, masks, weights, or LOD data. Step logs include camera count, coverage, multi-contributor percentage, accumulated-weight statistics, and stage timings.

## Commands

```bash
python scripts/generate.py --config config.yaml
python scripts/download_models.py --config config.yaml
python -m unittest discover -s tests -v
```

The main output is `outputs/<experiment>/<run-id>/result.png` plus the resolved config, metadata, and log.

## Legacy and baseline

The previous DiffPano algorithm (persistent spherical latent, temporary ERP fusion, VAE residual bridge, scheduler-consistent reinjection, and spherical write-back) is archived under [`experiments/legacy_spherical/`](../experiments/legacy_spherical/). The SphereDiff reference namespace and baseline config remain under [`spherediff/`](../spherediff/).

```bash
python -m experiments.legacy_spherical.generate --config spherediff/config.yaml
```
