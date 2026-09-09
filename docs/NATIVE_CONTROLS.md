# Controlled native, trajectory, noise, fusion, and geometry experiments

Implementation starts from local `no_sphere` commit
`320ad528a566be371f9f2b7cb3f4d3b26bd9c79c` (clean working tree).

## Phase A: native baseline and single-patch trajectories

`canvas.mode: planar` and `global_pipeline.mode: native_multidiffusion`
select a persistent native canvas. `native_multidiffusion` geometry is in raw
latent cells or PixelDiT pixels. It is independent of `planar` RGB geometry
and the preserved `PLANAR_NATIVE20_GEOMETRY` preset.

| Backend | State | Native patch | Native canvas H×W | RGB patch | Initialization | Step |
|---|---|---:|---:|---:|---|---|
| SD2 | raw 4-channel latent | 64 | 64×128 | 512 | Gaussian × DDIM init sigma | DDIM step |
| SANA | raw 32-channel latent | 32 | 32×64 | 1024 | Gaussian × scheduler init sigma | configured first-order DPM step |
| FLUX | raw 16-channel latent | 128 | 128×256 | 1024 | unscaled Gaussian | packed FlowMatch Euler step |
| PixelDiT | 3-channel pixel state | 1024 | 1024×2048 | 1024 | unscaled Gaussian | existing official first-order DPM update |

Channel counts and spatial factors are read from the actual loaded backend;
the table describes the configured checkpoints. All presets have 50% overlap,
one global prompt, seed 0, and uniform arithmetic average. No resampling, VAE
encoding, patch decoding, clean recovery, or renoising occurs in the native
loop. Decode the final global latent canvas once, using existing VAE
scaling/shift conventions; enabled VAE tiling is output-only for this path.
The installed AutoencoderDC has an incomplete tiling path, so SANA presets
use an untiled full-canvas decode. PixelDiT output is already RGB. Intermediate native decoding is not enabled.

Each independent scheduler integration resets the scheduler step index/history.
FLUX derives its packed grid and image IDs from the raw patch shape and rejects
a mismatch with the scheduler's prepared RGB resolution. Its dynamic shift is
computed once in `prepare`; the loaded scheduler applies it once. SANA's clean
path requires flow prediction and flow sigmas. Forward noising is checked against
the same schedule sigma used in clean recovery, using scheduler index lookup.

Run the native configs in SD2, SANA, FLUX, PixelDiT order:

```bash
sbatch slurm/generate_a100.slurm configs/experiments/native_multidiffusion/sd2.yaml
```

After the native controls, compare single-patch trajectories:

```bash
python -m scripts.single_patch_trajectory --config configs/experiments/trajectory/sd2.yaml
```

The runner uses one backend instance, one conditioning object and schedule,
and exactly the same Gaussian for A and B. A advances native state using the
ordinary first-order step. B starts from native noise, predicts clean, then
reconstructs each later noisy input from the changing clean prediction and the
original epsilon; B never integrates a scheduler step. There are N guided model
predictions in each trajectory. With true FLUX CFG, each guided prediction has
two transformer forwards in both trajectories. Output includes `native_final.png`,
`x0_renoise_final.png`, the shared `initial_epsilon.pt`, resolved config, and
`trajectory.json` with state moments, scheduler parameters, counts and runtime.
Intermediate noisy native and clean predictions are different quantities, so
cross-trajectory image differences are reported only for the decoded endpoints.

`debug.overlap_disagreement: true` measures proposal differences BEFORE fusion
in native and planar RGB/x0 pipelines. Immediate horizontal/vertical lattice
neighbors with actual overlap are compared once each (O(N) neighbor edges),
including edge-aligned patches. Per-pair MAE and RMSE are averaged equally;
maximum MAE and pair count are also logged. These metrics are native units for
the native baseline and decoded RGB units for RGB/x0 pipelines. They should not
be numerically compared across those representations without accounting for units.

## Interpretation and order

- If native MultiDiffusion fails, basic model/scheduler/patch infrastructure may
  still be wrong. A passing mock suite alone does not establish image quality.
- If native works and one-patch x0-renoise fails, the trajectory is implicated.
- If one-patch x0 works but overlapping independent-noise x0 fails, cross-patch
  stochastic inconsistency is a candidate cause.
- If global consistent noise reduces disagreement, noise consistency matters.
- If pre-fusion proposals agree but seams remain, inspect fusion weighting.
- If planar controls work but paired ERP runs fail, investigate projection,
  resampling, and geometry next.

Keep expensive runs sequential and inspect the output of each stage before
advancing the scientific decision tree. Later matrices are configuration only;
unit tests do not justify automatically launching every matrix entry.

## Phase B: aligned noise and fusion controls

`global_pipeline.clean_consensus.noise_binding: camera_index` retains the
existing independent fixed Gaussian per patch. The planar-only
`global_native_canvas` alternative creates one **raw** native Gaussian and
retrieves exact crops. The persistent x0 canvas is still clean RGB: patches are
encoded separately, renoised using their global noise crops, predicted clean,
decoded, then fused using the existing RGB accumulator.

Every RGB canvas dimension, patch size, stride and origin (including final edge
origins) must map to integral native coordinates. The actual backend spatial
factor is used: the configured checkpoints have SANA 32, SD2 8, FLUX 8 and
PixelDiT 1. For example, latent factor 8 rejects stride 50 but accepts stride 48;
SANA factor 32 rejects stride 200 but accepts 192. Clean-native and noise-native
shapes must match exactly; neither tensor is resized. FLUX noise is stored in
raw latent cells, not packed tokens.

The defaults keep the legacy `native20` RGB geometry preset. Explicit
`planar.geometry_preset: custom` permits aligned layouts or supported full model
resolutions. Phase B presets use the same meaningful local resolutions as Phase
A with 50% overlap: SD2 512/256, SANA/FLUX/PixelDiT 1024/512 (RGB patch/stride).
The loaded backend validates the coordinate mapping, so no factor is guessed by
the noise implementation.

CPU/GPU noise storage retains one global field. Seed storage regenerates that
same entire CPU field for each retrieval; it is exact but may be slower. All
three storage modes use the same CPU Gaussian stream before device transfer.

**Limitation:** global consistent latent noise does not imply identical
independently encoded clean latents in overlaps. VAE patch encodings see different
context and boundaries. This experiment isolates noise consistency, not every
source of latent disagreement.

`configs/experiments/planar_x0_noise/` contains paired independent/global configs.
`configs/experiments/planar_fusion/` holds four configurations per backend:

| Suffix | Fusion | Weight |
|---|---|---|
| a | average | uniform |
| b | weighted_average | cosine |
| c | weighted_average | distance_to_boundary |
| d | detail_preserving_average | distance_to_boundary |

All fusion presets use global native noise and keep model, seed, prompt, layout,
steps and guidance fixed. Run PixelDiT noise comparisons first, then aligned
SD2/SANA/FLUX comparisons, then fusion controls. Inspect pre-fusion disagreement
alongside images. Changes in final seams alone do not identify their cause.

## Phase C: orchestration and geometry configurations

The four-backend representation matrix uses the existing native presets, the
new `representation/*-rgb.yaml` files and the independent-noise x0 presets:

```bash
python -m scripts.run_ablation_matrix --backend all --experiment native,rgb,x0 --dry-run
python -m scripts.run_ablation_matrix --backend sana,flux --experiment trajectory --dry-run
```

Available selections: `native`, `trajectory`, `rgb`, `x0`, `noise`, `fusion`,
`paired`, `erp_later`. The runner prints commands with `--dry-run`, or executes
only explicitly selected entries sequentially. Use it within a suitable GPU
allocation when executing. It does not submit Slurm jobs or launch the matrix
on import. On Grace, single configs can use the existing `generate_a100.slurm`;
trajectory runs can use the command below inside an allocation:

```bash
python -m scripts.single_patch_trajectory --config configs/experiments/trajectory/sd2.yaml
```

`planar_erp_pairs/` has RGB-state and x0 pairs for each backend. Pairs retain
prompt, seed, steps, guidance, fusion, output dimensions, and local resolution;
they change exact planar crop/placement to existing ERP projection/fusion.
X0 pairs both use independent slot noise because exact global latent cropping
is intentionally planar-only. Camera coverage and contributor counts differ
from the rectangular lattice; these are geometric controls, not an assertion
that every resampling footprint or random spatial slot is matched.

`erp_later/` prepares standard average, standard weighted average, standard DPA,
existing corrected LPW+DPA, and corrected LPW+DPA with rotated cameras. They use
the existing algorithms unchanged. Run these only after the planar controls
and noise/fusion comparisons have been understood; no complete expensive matrix
has been launched automatically.

## Real-validation results and report reproduction

The completed single-patch controls found meaningful ordinary native images but
visible fixed-epsilon x0-renoise degradation for all four tested backends,
including SD2. See [NATIVE_CONTROLS_REPORT.md](NATIVE_CONTROLS_REPORT.md) for
job IDs, measured results, images, limitations and the resulting decision to
leave noise/fusion/ERP experiments gated.

Trajectory configs can be explicitly listed for sequential execution in one
Grace allocation:

```bash
sbatch --time=01:00:00 slurm/trajectory_a100.slurm \
  configs/experiments/trajectory/sd2.yaml \
  configs/experiments/trajectory/sana.yaml \
  configs/experiments/trajectory/flux.yaml \
  configs/experiments/trajectory/pixeldit.yaml
```

For plotting on Grace, load `GCC/13.2.0 matplotlib/3.8.2`. Reporting scripts do
not load model weights or run inference:

```bash
python scripts/plot_overlap_mae.py --output outputs/native-controls/report/native-overlap.png PATH_TO_METADATA_JSON ...
python scripts/summarize_trajectories.py --output outputs/native-controls/report PATH_TO_TRAJECTORY_JSON ...
```

The overlap script saves PNG/PDF and a JSON source/value index, with separate
panels for each native representation. The trajectory script validates the
recorded fairness controls and step counts before saving PNG/PDF pairs and CSV.
Experiment artifacts remain in the existing ignored `outputs/` tree; the report
links to them in this checkout.
