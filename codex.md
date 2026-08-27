# DiffPano Codex handoff

Updated: 2026-08-27 (America/Chicago)

This is the architecture-first handoff for future work in `/home/shig/diffpano`. Read it completely, then inspect the current worktree before editing. The repository is intentionally dirty; preserve existing and unrelated changes.

## The project in one paragraph

DiffPano is a 360-degree panorama generator built on SphereDiff. Its persistent diffusion state is an irregular set of latent features attached to Fibonacci-distributed points on a sphere. At every diffusion timestep, overlapping perspective latent patches are sampled from that sphere and denoised by an ordinary pretrained 2D model. Original SphereDiff blends the resulting latent patches back onto the sphere. DiffPano instead adds an optional RGB consensus loop: convert each patch's predicted-clean latent `x0` to RGB, project all RGB views to a temporary equirectangular panorama (ERP), fuse their overlaps there, sample the consensus ERP back into each perspective camera, carry only that RGB correction through the VAE, reinject it consistently into the scheduler update, and write the corrected patches back to the spherical state.

The planar code under `experiments/planar/` is not the main product. It is a no-warp diagnostic branch created to isolate seam formation in overlap fusion, VAE conversion, reinjection, and write-back.

## The most important mental model

There are three different representations and they must not be conflated:

1. **Persistent spherical latent state**: the actual diffusion state, shaped conceptually as `[B, C, F=1, N_sphere]`. It lives on Fibonacci sphere points and persists across timesteps.
2. **Temporary perspective views**: square grids gathered from subsets of spherical points. The pretrained 2D model denoises these grids. They overlap on the sphere.
3. **Temporary ERP RGB canvas**: a pixel-space coordination surface used to make all views agree. It is not the persistent latent and is not independently diffused.

In compact form:

```text
                         persistent state
                  spherical latent z_t [B,C,1,N]
                              |
              gather overlapping perspective patches
                              |
                  pretrained 2D latent denoiser
                              |
        model output + predicted-clean x0 + ordinary x_(t-1)
                              |
              optional DiffPano RGB-consensus branch
     decode x0 -> project to ERP -> fuse -> sample to views
     -> VAE residual bridge -> scheduler-consistent reinjection
                              |
        exclusive or weighted write-back to spherical z_(t-1)
                              |
                       repeat next timestep
                              |
       final per-view VAE decode and weighted rendering to ERP
```

“Pixel diffusion” in this project does **not** mean that a diffusion network runs directly on ERP pixels. SANA or FLUX still performs latent diffusion on perspective patches. Pixel space is where predicted-clean views are reconciled before their correction is returned to latent/scheduler space.

## Why pixel fusion operates on predicted-clean `x0`

At a diffusion step, the current patch `x_t` and its scheduler update `x_(t-1)` are still noisy. The VAE is trained to decode clean image latents, so decoding those noisy states gives a poor coordination signal. DiffPano derives the model's predicted-clean patch `x0` from the scheduler prediction type and sigma schedule, then decodes that.

For the currently supported spherical image backends, the schedulers use flow prediction:

```text
x_t = x0 + sigma_t * flow
x0  = x_t - sigma_t * flow
```

The model still computes its normal scheduler step to obtain the unmodified `x_(t-1)`. Pixel fusion proposes a corrected clean target, and reinjection modifies the update without pretending the corrected clean latent already lives at timestep `t-1`.

## End-to-end spherical ERP pipeline

### 1. Load experiment, model, and prompts

The canonical entrypoint is `scripts/generate.py --config <yaml>`. It loads a typed `ExperimentConfig`, seeds host/Torch RNGs, creates a standardized output directory, builds a registered backend, and passes a flattened `PixelFusionConfig` into the model adapter.

Spherical backends currently registered:

- `sana`: Spherical SANA image generation with DiffPano pixel fusion.
- `flux`: Spherical FLUX.1 image generation with DiffPano pixel fusion.
- `hunyuan_video`: SphereDiff-derived video adapter; pixel fusion must be disabled.
- `ltx_video`: SphereDiff-derived video adapter; pixel fusion must be disabled.

`planar_sana`, `planar_flux`, and `planar_sd2` belong to `scripts/planar_test.py`, not normal spherical generation. SD2 has no spherical adapter yet.

### 2. Build directional conditioning

A spherical prompt file is intended to contain five non-empty lines representing broad vertical regions:

1. north/upward pole;
2. upper/equatorial directions;
3. equator;
4. lower/equatorial directions;
5. south/downward pole.

Each line is expanded across longitudes `0, 90, 180, 270` degrees, producing 20 prompt anchors at latitudes `-90, -10, 0, 10, 90` degrees. Every denoising view uses the prompt anchor with highest cosine similarity to its camera direction.

Important current mismatch: `load_directional_prompts()` accepts one line and expands it to five for config/metadata, but the SANA and FLUX spherical pipeline bodies reopen the original file and assert exactly five lines. Therefore use a five-line file for spherical runs until that backend mismatch is fixed. One-line prompts currently work in the planar runner.

### 3. Initialize the persistent sphere

`SphericalFunctions.fibonacci_sphere(N)` creates approximately uniform unit directions. A Gaussian latent feature is attached to each direction:

```text
sphere directions: [N, 3]
persistent latent: [B, C, 1, N]
```

`sphere.num_points` is the main spatial-capacity control for the persistent representation. Raising ERP output resolution without raising the appropriate spherical/model capacity only increases the temporary canvas resolution; it does not create more independent latent degrees of freedom.

SANA usually uses `N=2600`. FLUX experiments use a much larger `N`, commonly `26500`, because one FLUX spherical point carries a packed transformer token corresponding to a `2x2` block of raw VAE latents.

### 4. Build the overlapping camera cover

The current view layout is fixed in `horizontal_and_vertical_view_dirs_v3_fov_xy_dense_equator()` rather than fully configurable.

- Perspective FOV: `80 x 80` degrees.
- Requested overlap fraction: `0.6` horizontally and vertically.
- Latitude bands: `0, ±22.5, ±45, ±67.5, ±90` degrees.
- More cameras are placed near the equator than near the poles.
- Total: 89 perspective views per denoising step.

The 89-view cover is spherical: views on opposite sides of the ERP left/right boundary overlap normally in 3D. It is not an ultra-wide flat sliding window.

`sphere.fov` exists in typed configuration but is reserved and must remain `null`. Changing camera FOV/layout currently requires deliberate geometry code work and corresponding coverage tests.

### 5. Dynamically gather a perspective latent patch

For each camera, `dynamic_latent_sampling()`:

1. projects all Fibonacci points into that camera;
2. estimates a square patch size from FOV and sphere density;
3. selects visible points for the perspective footprint;
4. orders the irregular points into a square 2D grid for the model;
5. computes a center-confidence score
   `exp(-distance_from_view_center / weighted_average_temperature)`.

The score is used for SphereDiff-style spherical overlap write-back and exclusive ownership. It is separate from the RGB ERP aggregation weight map.

In the successful 20-step SANA spherical run with `N=2600`, all 89 patches contained 400 points, i.e. `20x20` latent grids.

Backend detail:

- SANA reshapes selected spherical features directly to `[B,C,H_lat,W_lat]`.
- FLUX stores packed-token features on the sphere. A selected token grid is passed to the transformer as `[B,H_token*W_token,C_packed]`; for VAE work it is unpacked to raw `[B,16,2H_token,2W_token]` latents and packed again afterward.

### 6. Denoise each perspective patch

Each patch is conditioned by the nearest directional prompt. The backend transformer predicts its native flow/noise representation, classifier-free or embedded guidance is applied, and the backend scheduler computes the normal proposal `x_(t-1)`.

Scheduler state is reset for every independent patch at the same global timestep. This is essential because all 89 patches are alternative views of one spherical state, not a temporal sequence of scheduler calls.

When pixel fusion is disabled, each patch proposal is immediately accumulated into the spherical next-state with the original center-weighted SphereDiff rule.

When pixel fusion is enabled for the current step, the code stores for every view:

- current `x_t` patch;
- model output;
- predicted-clean `x0`;
- ordinary scheduler proposal `x_(t-1)`;
- spherical indices and center scores;
- stable patch ID, camera direction, and FOV.

### 7. Decode predicted-clean views to RGB

The VAE decodes predicted-clean patch latents into `[views,3,H_px,W_px]` RGB tensors in the VAE's native `[-1,1]` convention. Scaling and optional shift come from the VAE config; they are not hard-coded. VAE calls run under inference mode and may be chunked using `performance.vae_chunk_size`.

FLUX uses adapters around this stage to unpack transformer tokens into raw VAE latents. This packs latent points, not four perspective patches.

### 8. Project and fuse in canonical ERP space

All projection/fusion geometry is promoted to FP32 even when the model runs BF16/FP16.

For standard warping, every ERP pixel is converted to a world ray, mapped into each perspective camera, checked for front-hemisphere and image-bound validity, and sampled from the perspective RGB view. This produces projected RGB, a validity mask, and a projected confidence map for every contributing view.

The ERP canvas has correct spherical topology:

- longitude is periodic, so the left/right boundary represents the same meridian;
- views crossing that meridian contribute to both edges;
- ERP-to-view sampling uses horizontal wrap;
- filtering across poles reflects latitude and rolls longitude by half a turn;
- uncovered pixels are masked and never divided by zero.

`sphere.erp_height` and `sphere.erp_width` set both the temporary per-step fusion canvas and the final panorama size. They affect memory and projection cost strongly.

The two warp paths are:

- `standard`: direct single-scale perspective -> ERP projection, aggregation, then ERP -> perspective sampling.
- `lpw`: Laplacian Pyramid Warping. Build perspective Laplacian levels, use projection-Jacobian footprint to assign suitable scale/LOD, project and fuse coefficients at matching ERP pyramid levels, reconstruct a seam-aware ERP, then perform the reverse multiscale operation back to perspective views.

LPW is meant to reduce distortion/aliasing and preserve detail under perspective-to-ERP resampling, especially toward the poles. It is not merely a different scalar overlap weight.

### 9. Sample ERP consensus back into the original views

The fused ERP is sampled with the same camera directions, FOVs, and view sizes. A validity mask is sampled alongside RGB. Invalid samples fall back to the original predicted-clean RGB view, so incomplete ERP coverage does not inject black pixels.

Interpolation can be `nearest` or `bilinear`. Nearest is the current root default and is useful when studying whether interpolation itself creates blur; bilinear is smoother but can mix detail.

### 10. Cross the non-invertible VAE with the residual bridge

Directly using `encode(fused_rgb)` as the new clean latent introduces two changes at once:

1. the desired RGB consensus correction;
2. unrelated VAE round-trip drift from `encode(decode(x0)) != x0`.

The identity-preserving bridge isolates the first change:

```text
original_rgb          = decode(original_x0)
original_roundtrip    = encode(original_rgb)
fused_roundtrip       = encode(fused_rgb)
rgb_fusion_delta      = fused_roundtrip - original_roundtrip
target_x0             = original_x0 + rgb_fusion_delta
```

Both encodes are deterministic posterior mean/mode. In the spherical DiffPano pipeline this bridge is always used when pixel fusion is active; there is currently no spherical config switch to bypass it. The planar ablation has been used to compare bridged and direct re-encoding behavior.

For FLUX, `original_x0` is unpacked to VAE space before applying the delta, then repacked into transformer/scheduler space.

### 11. Reinject the corrected clean target into the scheduler state

The recommended/default mode is `noise_consistent`. For a flow scheduler:

```text
clean_correction = strength * (target_x0 - original_x0)
corrected_flow   = original_flow - clean_correction / sigma_t
corrected_x_prev = scheduler.step(corrected_flow, t, current_x_t)
```

This has two useful invariants:

- strength `0` exactly preserves the original scheduler proposal;
- a zero RGB correction preserves the original model flow and update.

The implementation currently guards `noise_consistent` to `flow_prediction` schedulers with explicit sigma schedules. That matches the spherical SANA/FLUX paths. Supporting epsilon, v-prediction, or sample prediction for the full reinjection step would require scheduler-specific validation even though predicted-clean extraction contains partial support.

Other implemented reinjection modes are ablations:

- `replace`: mix the ordinary `x_(t-1)` proposal directly with fused clean `x0` by `strength`.
- `weighted_replace`: same, but multiplied by the valid mask.
- `residual`: add `strength * (target_x0 - original_x0)` to the ordinary `x_(t-1)` proposal.

These alternatives are operational but are not as principled about noise level as `noise_consistent`.

### 12. Write corrected views back to the persistent sphere

There are two implemented modes:

- `weighted_average`: preserve original SphereDiff behavior. Every overlapping corrected patch contributes to a point using its center score, and values are normalized by accumulated score.
- `exclusive`: build a stable owner map. Every spherical point is assigned to the covering patch with highest center score; ties go to the lowest stable patch ID. Each covered point is then written exactly once.

Exclusive write-back is the current DiffPano default. Its motivation is that RGB views have already reached a global consensus in ERP space; a second latent average can reintroduce blur and VAE/view inconsistency. Ownership is cached while the complete geometry signature is unchanged.

For points not covered by any view, exclusive mode can:

- `error` (default), making geometry holes explicit; or
- use `weighted_average_fallback` for uncovered points only.

### 13. Repeat and render the final ERP

The corrected spherical state becomes `z_(t-1)` and the process repeats.

After the final denoising step, the irregular spherical latent cannot be decoded in one global VAE call. The pipeline gathers the same perspective views, decodes them independently, pastes them to an ERP with SphereDiff's center-weighted projector, and divides by accumulated weights. This final rendering pass is separate from per-step pixel fusion.

Therefore:

- ERP is used during DiffPano fusion as a consensus workspace;
- the sphere remains the persistent state;
- the final image is also ERP, produced from decoded spherical views;
- final-render blending can still create visible seams even if denoising consensus is good, which is one reason the planar seam studies exist.

## Current option matrix

The table below distinguishes actual runtime choices from reserved ideas.

| Pipeline part | Implemented choices | Current root default | Important notes |
|---|---|---|---|
| Backend | `sana`, `flux`; video baseline adapters | `sana` | Pixel fusion only for SANA/FLUX. SD2 is planar-only. |
| Persistent sphere density | positive `sphere.num_points` | `2600` | Model-dependent; FLUX commonly uses `26500`. |
| Camera cover/FOV | fixed 89-view, 80-degree cover | fixed | `sphere.fov` is reserved and must be null. |
| Prompt conditioning | five directional lines -> 20 anchors | `prompts/ruins.txt` | Spherical adapters currently require exactly five physical lines. |
| Fusion on/off | `fusion.enabled` | `true` | False selects original SphereDiff latent write-back. |
| Fusion time window | `start_ratio`, `end_ratio` in `[0,1]` | `0`, `1` | Inclusive normalized step window. |
| Fusion cadence | `every_n_steps >= 1` | `1` | Uses absolute step index modulo N. |
| Warp | `standard`, `lpw` | `lpw` | Single-scale vs Laplacian Pyramid Warping. |
| ERP aggregation | `average`, `weighted_average`, `detail_preserving_average` | DPA | Independent of spherical write-back mode. |
| View confidence | `uniform`, `cosine`, `gaussian`, `distance_to_boundary` | boundary distance | Ignored by plain average. |
| LPW levels | positive integer | `4` | Actual pyramid can stop early at tiny sizes. |
| LPW LOD | `jacobian`, `none` | `jacobian` | None forces finest/zero LOD. |
| LOD interpolation | `nearest`, `linear` | `nearest` | Maps projection footprint to pyramid-level confidence. |
| ERP -> view interpolation | `nearest`, `bilinear` | `nearest` | Applies to RGB and coverage sampling. |
| Perspective pyramid vertical pad | `reflect`, `replicate` | `reflect` | ERP filtering itself always uses spherical seam/pole padding. |
| VAE bridge | identity-preserving residual bridge | always active | No spherical on/off switch today. |
| VAE encode stochasticity | field exists | effectively deterministic | The paired bridge forces posterior mean/mode. |
| Reinjection | `noise_consistent`, `replace`, `weighted_replace`, `residual` | noise-consistent | Full noise-consistent path currently flow-only. |
| Reinjection strength | float | `1.0` | Normally use `[0,1]`; not range-clamped. |
| Spherical write-back | `exclusive`, `weighted_average` | exclusive | This is after RGB fusion/reinjection. |
| Exclusive owner | `max_center_weight` | same | Only implemented owner policy. |
| Uncovered points | `error`, `weighted_average_fallback` | error | Fallback affects uncovered points only. |
| Time travel | fields exist | disabled | Enabling raises `NotImplementedError`; not wired. |
| VAE chunking | positive integer | `1` | Trades speed for peak memory. |
| Projection chunking | positive integer | `1` | Views accumulated per projection chunk. |
| Precision | model FP16/BF16/FP32 | BF16 | Projection/fusion still FP32. |
| Diagnostics | masks, ERP, owner map, geometry, timings | disabled | Saved only when explicitly requested. |

## ERP aggregation choices in detail

### Average

```text
fused = sum(valid_i * value_i) / sum(valid_i)
```

This ignores configured view-confidence maps. It is the cleanest arithmetic baseline.

### Weighted average

```text
fused = sum(valid_i * weight_i * value_i) / sum(valid_i * weight_i)
```

Available patch-local weights:

- `uniform`: all in-bounds pixels equal;
- `cosine`: separable cosine falloff to all patch boundaries;
- `gaussian`: radial Gaussian with fixed sigma `0.5` in normalized patch coordinates;
- `distance_to_boundary`: confidence is the minimum distance to a patch edge, normalized to one.

These weights are projected with the views. They should not be confused with `sphere.weighted_average_temperature`, which controls center confidence on spherical latent points and final SphereDiff rendering.

### Detail-preserving average (DPA)

The implemented DPA computes an ordinary weighted estimate and a coefficient-magnitude-weighted estimate:

```text
detail_weight = valid * view_weight * (abs(value) + eps)^power
detail        = sum(detail_weight * value) / sum(detail_weight)
fused         = ordinary + alpha * (detail - ordinary)
```

- `alpha=0` reduces to ordinary weighted average.
- Larger `power` favors locally larger-magnitude RGB values or Laplacian coefficients.
- With `warp=lpw`, DPA works on multiscale Laplacian coefficients and is closer to the intended detail-preserving use.
- With `warp=standard`, it acts directly on RGB values.

This is LookingGlass-inspired, but it is **not** the original specification's explicit `winner_take_most` softmax/temperature method. There is currently no `high_frequency_mode` or `high_frequency_temperature` runtime field.

## LPW choices in detail

LPW addresses the fact that one perspective source pixel can cover very different ERP footprints depending on camera direction and latitude.

1. Build a Gaussian/Laplacian pyramid for all perspective predicted-clean RGB views.
2. Estimate an ERP-space source footprint from derivatives of the perspective projection grid.
3. Convert footprint to `log2` LOD.
4. Give each Laplacian level `nearest` or linearly interpolated confidence.
5. Project and aggregate matching coefficients into ERP levels.
6. Reconstruct with validity-normalized, longitude-periodic, pole-aware filtering.
7. Build a masked ERP pyramid, sample every level back into each camera, and reconstruct the fused view.

The coarsest Gaussian residual always contributes so that a base signal survives even where fine LOD selection is sparse.

## What is not implemented yet

Do not assume these ideas from the archived `research_requirements.md` are available merely because related config names were proposed:

- Time travel / scheduler jump-and-redenoise. Fields are reserved; enabling it raises.
- Explicit spherical-area weighting in ERP aggregation.
- `winner_take_most` or temperature-softmax high-frequency fusion.
- Configurable uncovered-ERP fill policies such as keep-previous or nearest-valid. Current fusion masks invalid ERP regions and falls back to original RGB when sampling views.
- A spherical switch to disable the VAE residual bridge.
- Configurable camera layouts or FOV through YAML.
- Batch size greater than one in spherical image adapters.
- A spherical SD2 adapter.
- Pixel fusion for HunyuanVideo or LTX-Video.
- A truly pixel-domain diffusion model.

Potential cleanup/consistency issue: the final SANA/FLUX rendering loops call dynamic sampling with the loop's `_fov` variable while separately naming `fov_vae`. All current view FOVs are 80 degrees, so this is behaviorally identical today, but it should be corrected before supporting heterogeneous per-view FOVs.

## Backend-specific notes

### SANA

- Model: `Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers`.
- Typical precision/guidance: BF16, guidance `4.5`.
- Raw VAE latent-to-pixel ratio: 32 pixels per latent point.
- Spherical state stores the direct SANA latent grid features.
- Flow-prediction scheduler supports current noise-consistent reinjection.
- A full 20-step spherical DiffPano result completed successfully as job `19433921`:
  `test_outputs/pipeline_full_20step_0818/full-sphere-sana-forest-20step/20260818-074350-19433921/result.png`.
- That validation used 89 views x 400 points at every step, `N=2600`, standard warp, arithmetic RGB average, uniform weights, exclusive write-back, and a `256x512` ERP.

### FLUX.1

- Model: `black-forest-labs/FLUX.1-dev`.
- Typical precision/guidance: BF16, guidance `3.5`; CPU offload is available.
- Raw VAE latent-to-pixel ratio: 8.
- Every `2x2` raw VAE-latent block is packed into one transformer token, so a token spans `16x16` image pixels.
- The spherical state stores packed token feature vectors, not four independent patches.
- Pixel fusion temporarily unpacks token grids to 16-channel VAE latents, applies the shared RGB pipeline, then repacks.
- FLUX pixel-fusion smoke outputs exist, but the recent full 20-step job `19433922` timed out at the deliberately short 30-minute smoke limit and produced only `config.yaml` and `run.log`. Do not claim that full FLUX spherical validation completed.

### SD2 and future backends

- SD2 (`sd2-community/stable-diffusion-2-base`) has been integrated only into the planar diagnostic branch.
- SD2 raw VAE ratio is 8 pixels per latent.
- SD3/SD3.5 and FLUX.2 are not implemented. Their raw VAE ratio is also 8; a transformer patch size of 2 means a token spans 16 pixels.
- Prior official references used for those future-model facts:
  - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_sd3.py
  - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2.py

## Model cache

Cluster Hugging Face cache root:

```text
/scratch/user/shig/SphereDiff/hf_cache/hub
```

Known snapshots:

```text
/scratch/user/shig/SphereDiff/hf_cache/hub/models--Efficient-Large-Model--Sana_1600M_1024px_BF16_diffusers/snapshots/e18f82ddb8233fa4d979c2613f41a3ca4c5fc730
/scratch/user/shig/SphereDiff/hf_cache/hub/models--Efficient-Large-Model--Sana_1600M_1024px_BF16_diffusers/snapshots/e2b3c0cbffebcd09d83805e88b9f5f106afc74ac
/scratch/user/shig/SphereDiff/hf_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21
/scratch/user/shig/SphereDiff/hf_cache/hub/models--sd2-community--stable-diffusion-2-base/snapshots/f5bc1bd97485577aa0b946fa8a9004e2ec147402
```

`model.path` takes precedence over `model.id`. Direct snapshots are useful on offline compute nodes.

## Main spherical implementation map

- `scripts/generate.py`: canonical config-driven entrypoint and standardized outputs.
- `config.yaml`: complete root/default experiment.
- `diffpano/config.py`: typed experiment config, all validation, and conversion to `PixelFusionConfig`.
- `diffpano/initialization.py`: deterministic seeds and directional prompt loader.
- `diffpano/geometry.py`: SphereDiff spherical coordinates, Fibonacci points, fixed camera cover, dynamic point gathering/order, and legacy final ERP paste.
- `diffpano/pipelines/sana.py`: SANA model invocation and spherical denoising loop.
- `diffpano/pipelines/flux.py`: FLUX model invocation, packing adapters, and spherical denoising loop.
- `diffpano/diffusion.py`: model-independent predicted-clean RGB-fusion orchestration.
- `diffpano/projection.py`: FP32 perspective/ERP grids, seam/pole padding, cache, and ERP-to-view extraction.
- `diffpano/lpw.py`: Laplacian Pyramid Warping and projection-derived LOD.
- `diffpano/fusion.py`: weight maps, average, weighted average, DPA, and chunked ERP accumulation.
- `diffpano/vae.py`: VAE scaling/shift adapters, chunked encode/decode, and identity-preserving residual bridge.
- `diffpano/reinjection.py`: prediction-type handling and scheduler-consistent correction.
- `diffpano/writeback.py`: weighted and exclusive spherical write-back plus ownership diagnostics.
- `diffpano/diagnostics.py`: masks, projection checks, norms, timing, and temporary ERP exports.
- `diffpano/metadata.py`: compact runtime geometry, seeds, timesteps, and output metadata.
- `spherediff/`: baseline/reference namespace; fusion disabled retains original weighted latent write-back.
- `docs/USAGE.md`: field-by-field operating guide.

## Canonical root defaults

The important active defaults in `config.yaml` are:

```yaml
model:
  pipeline: sana
  precision: bf16

generation:
  num_inference_steps: 20
  guidance_scale: 4.5
  height: 1024
  width: 1024

sphere:
  num_points: 2600
  erp_height: 2048
  erp_width: 4096
  weighted_average_temperature: 0.1

fusion:
  enabled: true
  start_ratio: 0.0
  end_ratio: 1.0
  every_n_steps: 1
  warp: {mode: lpw}
  lpw:
    levels: 4
    lod_mode: jacobian
    lod_interpolation: nearest
    erp_vertical_padding_mode: reflect
    erp_to_perspective_interpolation_mode: nearest
  aggregation:
    mode: detail_preserving_average
    weight_mode: distance_to_boundary
    alpha: 1.0
    power: 1.0
    epsilon: 1.0e-6

reinjection:
  mode: noise_consistent
  strength: 1.0

writeback:
  mode: exclusive
  owner_mode: max_center_weight
  owner_map_static: true
  uncovered_mode: error
```

`generation.height/width` describe perspective model/view resolution. `sphere.erp_height/width` describe the fusion/final panorama canvas. They are different controls.

## Diagnostics and reproducibility

When enabled, the system can record:

- fused and original predicted-clean ERP frames;
- valid, overlap, contributor-count, and accumulated-weight masks;
- camera directions and FOVs;
- owner IDs, owner scores, coverage, write counts, and patch histogram;
- VAE round-trip error and RGB-fusion latent delta norms;
- original scheduler-update norm, reinjection norm, and their ratio;
- projection center/round-trip checks;
- stage timings for decode, projection/LPW, aggregation, reconstruction, encode, and reinjection.

Metadata records resolved prompts, seed, timesteps, spherical point count, patches per step, patch-point histograms, fusion-applied steps, model/config, environment, output path, and Slurm job ID.

Projection grids, LOD maps, patch weights, and static owner maps are cached using geometry/content signatures. Geometry and fusion use FP32 for numerical stability.

## Baseline versus DiffPano

With `fusion.enabled: false`:

```text
patch x_t -> model -> scheduler x_(t-1)
-> center-weighted latent accumulation on the sphere
```

With `fusion.enabled: true`:

```text
all patch x0 predictions -> RGB ERP consensus
-> bridged latent clean targets -> reinjected x_(t-1)
-> exclusive or weighted spherical write-back
```

Both use the same spherical points, camera cover, dynamic sampling, model invocation, prompts, scheduler, and final ERP renderer. This makes fusion-on/off the closest SphereDiff baseline comparison.

## Planar seam-ablation appendix

The planar branch deliberately removes spherical projection. It uses a persistent rectangular latent canvas and overlapping rectangular model patches. Its purpose is to determine whether seams originate in overlap density, RGB/latent aggregation, VAE round-trip drift, scheduler reinjection, write-back, or final patch decoding.

Important files:

```text
experiments/planar/fusion.py
experiments/planar/pipeline.py
experiments/planar/model_pipelines.py
scripts/planar_test.py
```

Implemented planar backends: SANA, FLUX.1, and SD2. The planar branch supports latent fusion or RGB fusion, optional VAE residual bridge, exclusive or weighted write-back, the same three aggregation modes, and configurable latent patch size/stride.

Completed recent work, kept here only for reproducibility:

- Five-method one-patch-high matrix: 30/30 results (3 models x 2 prompts x 5 fusion/write-back methods).
- Original MultiDiffusion SD2 flat-panorama baselines: 2/2 results.
- DiffPano SD2 latent runs with MultiDiffusion geometry: 2/2 results.
- Latent stride sweep: 34/34 results, covering strides 4/8/12/16/20 for all models and stride 40 additions for SD2/FLUX.
- Result roots:
  - `test_outputs/planar_1patch_high_matrix_0826` (34 result images total).
  - `test_outputs/planar_latent_stride_sweep_0827` (34 result images total).
- Config roots:
  - `experiments/planar/matrix_1patch_high/`.
  - `experiments/planar/latent_stride_sweep_0827/`.

The matrix uses `640x4096`, one physical patch high, 20 steps, seed 1234, and prompts `xfjord`/`xvalley`. The five methods are latent average; direct RGB encode with exclusive/weighted write-back; and residual-bridge RGB with exclusive/weighted write-back.

MultiDiffusion comparison: both original MultiDiffusion and the planar latent mode denoise overlapping global-latent patches and average proposed next states. MultiDiffusion is still a flat canvas, not spherical: no Fibonacci state, perspective warp, longitude wrap, pole handling, or ERP consensus. Original MultiDiffusion uses SD2 latent `64x64` windows, stride 8, and 171 patches per step at `640x4096`; standard one-patch-high DiffPano SD2 uses `80x80`, stride 20, and 23 patches. Final decode paths also differ.

Planar physical ratios:

| Model | Raw VAE pixel ratio | `640x640` patch latent |
|---|---:|---:|
| SD2 | 8 | `80x80` |
| FLUX.1 | 8 | `80x80` raw VAE latent, packed to `40x40` tokens internally |
| SANA | 32 | `20x20` |

## Slurm/HPRC operating notes

The user works on TAMU HPRC and GPU/model testing must be submitted through Slurm rather than run on the login node.

Typical job environment:

```bash
module purge
module load WebProxy
source activate_venv diffpano
export PYTHONPATH="$HOME/diffpano${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SCRATCH/SphereDiff/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
```

Use repository-resident configs. Compute-node `/tmp` is node-local and cannot see login-node temporary files.

Useful status commands:

```bash
squeue -j JOBID -r -o '%.20i %.28j %.10T %.10M %.10l %R'
sacct -X -j JOBID --format=JobID%22,State%24,Elapsed,ExitCode -n -P
```

Operational history:

- `slurm/smoke_planar.slurm` requests 30 minutes; full experiments use longer launchers.
- Node `g003` once hung reading a SciPy shared object from `/sw`; later arrays excluded it.
- `afterok` dependents can disappear automatically when a prerequisite fails.
- Completed, failed, cancelled, or timed-out jobs leave `squeue`; use `sacct` for terminal status.
- Array allocation IDs can differ from `ArrayJobId_task`; use logs and `sacct` to map them.
- Direct cached model snapshots avoid offline tokenizer/model-ID resolution failures.
- Earlier `ModuleNotFoundError: diffpano` was fixed in launchers by exporting repo-root `PYTHONPATH`.

## Output layout

Canonical spherical runs are saved as:

```text
<output.directory>/<experiment.name>/<timestamp-jobid>/
├── result.png or result.mp4
├── config.yaml
├── metadata.json
├── run.log
└── intermediates/   # only when requested
```

## Validation state

- Spherical SANA: full 20-step pixel-fusion result verified (`19433921`).
- Spherical FLUX: smoke-level artifacts exist; the recent 20-step 30-minute job timed out (`19433922`), so a longer full validation is still needed.
- Spherical video pixel fusion: unsupported by validation and config.
- Planar implementation syntax/config/unit checks passed during development.
- `tests/test_config.py`: 5 tests passed during planar extension work.
- `tests/test_planar_patch_fusion.py`: 12 tests passed during planar extension work.
- All 68 planar/baseline/sweep result images summarized above were verified on disk.

Do not generalize planar completion into proof that spherical ERP LPW/DPA is seam-free. The planar tests diagnose components; spherical projection, pole behavior, ERP sampling, and final spherical rendering remain additional variables.

## Current worktree and safety

At this handoff the tracked modified files are:

```text
diffpano/config.py
diffpano/initialization.py
experiments/planar/fusion.py
experiments/planar/pipeline.py
scripts/planar_test.py
slurm/smoke_planar.slurm
```

Important untracked paths:

```text
codex.md
experiments/planar/model_pipelines.py
experiments/planar/matrix_1patch_high/
experiments/planar/latent_stride_sweep_0827/
multidiffusion/
prompts/xfjord.txt
prompts/xvalley.txt
experiments/planar/fusion.py.orig
```

`experiments/planar/fusion.py.orig` predates the latest work and was deliberately left untouched. Do not delete it without checking. No Git commit has been made for the current worktree.

The integrated `apply_patch` helper can fail because `/home/shig/.codex` is a writable symlink to `/scratch/user/shig/.codex`, causing the sandbox setup to abort. The system `apply_patch` command under an approved shell was used as a fallback. Always inspect exact diffs after editing.

## Recommended priorities for future work

1. Treat spherical SANA ERP generation as the main path; use planar results only to choose fusion/reinjection/write-back hypotheses.
2. Run a sufficiently long full FLUX spherical validation and record whether failure is runtime, memory, geometry, or quality.
3. Compare spherical `standard` vs `lpw`, then aggregation modes, with identical seed/model/prompts and saved intermediate ERP/masks.
4. Inspect final-render seams separately from per-step fused-ERP seams.
5. Fix the one-line/five-line spherical prompt mismatch.
6. Correct the final-loop `_fov` naming before heterogeneous FOV work.
7. Add missing research options only as explicit ablations: bridge bypass, spherical-area weights, and high-frequency winner-take-most.
8. Implement time travel only after ordinary pixel fusion is stable and validated.

## Suggested opening for a new conversation

```text
Read /home/shig/diffpano/codex.md completely, then inspect the current /home/shig/diffpano worktree. Treat the spherical ERP/pixel-fusion architecture as the main project. The planar experiments are seam-diagnosis ablations, not the primary pipeline. Preserve existing changes and continue all work in this repository.
```
