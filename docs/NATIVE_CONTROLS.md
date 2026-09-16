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

## Three-way model-implied endpoint control

The additive experiment compares unchanged native sampling, unchanged
fixed-initial-noise reconstruction, and model-implied endpoint reconstruction.
Run the full regression suite first. Then, in Grace, submit each backend
independently with the same prepared trajectory config, for example:

```bash
sbatch --time=00:20:00 slurm/endpoint_trajectory_a100.slurm configs/experiments/trajectory/sd2.yaml
```

The new entrypoint is `python -m scripts.three_way_trajectory --config CONFIG`
inside a GPU allocation. It saves separate trajectory images, shared epsilon,
per-step CSV/JSON, fairness checks and actual N/N/N guided-prediction counts in
`outputs/endpoint-controls/`. The native one-step diagnostic reuses A's prediction
and adds no model evaluations. All three loops remain in native coordinates;
latent backends decode only the three final endpoints.

`scripts/plot_endpoint_controls.py --output REPORT_DIR TRAJECTORY_JSON ...`
creates default-matplotlib triptychs, per-step error plots and summary tables.
September 9 results restored quality for all four models by preserving the
implied endpoint; see the appended experiment in
[NATIVE_CONTROLS_REPORT.md](NATIVE_CONTROLS_REPORT.md) for numerical errors,
FLUX's bfloat16 rounding qualification, failed-job history and limitations.
No new panorama method or same-timestep Time Travel is implemented.

## Planar RGB implied-endpoint consensus

`global_pipeline.mode: implied_endpoint_consensus` selects the additive planar
experiment. It uses `native_multidiffusion` geometry mapped exactly to RGB by the
loaded backend factor, one global prompt, fixed patches and average/uniform
fusion. It preserves local noisy states; only predicted-clean RGB is shared.
Current local model-implied endpoints are retained during each step, never fused
or reused across steps. Latent clean predictions undergo deterministic VAE
roundtrips. Patch batching is currently restricted to 1.

```bash
sbatch slurm/rgb_endpoint_a100.slurm configs/experiments/implied_endpoint_consensus/sd2.yaml
```

This paired launcher runs native MultiDiffusion and RGB endpoint consensus from
one shared global initialization and adds both one-patch latent roundtrip controls.
Use `python -m scripts.generate --config CONFIG` for the new method alone.
All four paired presets match the successful native controls exactly except mode,
experiment name and output group. Results go to `outputs/rgb-endpoint-controls/`.
Generate comparisons with `python -m scripts.plot_rgb_endpoint_controls --output
outputs/rgb-endpoint-controls/report` followed by the four `comparison.json` paths.
`result.png` renders terminal local states; `final_fused_clean.png` also retains
the final clean consensus before terminal output decoding. Full diagnostics and
scientific interpretation are appended to `NATIVE_CONTROLS_REPORT.md`.

## Training-free LookingGlass residual controls

The updated correction uses no trained bridge or learned parameters. Run from
`~/diffpano` on Grace. Specs in `configs/experiments/vae_residual/` point to the
exact saved A40 F/B initial tensors and settings; a different GPU model,
configuration, conditioning hash or schedule is rejected. Output directories
must be new, so reruns need a new `output` path in a copied spec.

After the full CPU regression suite passes, run phase CD for each latent backend:

```bash
sbatch --job-name=residual-cd-sd2 slurm/vae_residual.slurm CD configs/experiments/vae_residual/sd2.json
sbatch --job-name=residual-cd-sana slurm/vae_residual.slurm CD configs/experiments/vae_residual/sana.json
sbatch --job-name=residual-cd-flux slurm/vae_residual.slurm CD configs/experiments/vae_residual/flux.json
```

Each job checks real-VAE identity recovery before image trajectories. B is a
matched no-roundtrip oracle; C explicitly decodes/reencodes each clean prediction;
D adds `z0 - E(D(z0))` to that roundtrip. C/D have no patch fusion.

Inspect those results and pass the full regression suite before phase G:

```bash
sbatch --job-name=residual-g-sd2 slurm/vae_residual.slurm G configs/experiments/vae_residual/sd2.json
sbatch --job-name=residual-g-sana slurm/vae_residual.slurm G configs/experiments/vae_residual/sana.json
sbatch --job-name=residual-g-flux slurm/vae_residual.slurm G configs/experiments/vae_residual/flux.json
```

G uniformly averages the VAE residuals on a temporary native-coordinate canvas,
then adds exact residual crops to the encoded fused RGB crops. All proposals use
frozen source states. Model-implied endpoints are neither averaged nor changed.
`PlanarImpliedEndpointConsensusPipeline(..., residual_correction=True)` enables G;
the default remains the existing F behavior. Pixel-native and learned-bridge
combinations are rejected.

Render saved results with the matplotlib environment:

```bash
python scripts/plot_vae_residual_controls.py --output outputs/vae-residual-controls/report configs/experiments/vae_residual/{sd2,sana,flux}.json
```

Use `--single-only` before G has completed. See the appended September 10 section
in `NATIVE_CONTROLS_REPORT.md` for the numerical and image evidence.

Validate the saved experiment manifests without loading models:

```bash
python scripts/audit_vae_residual_controls.py --output outputs/vae-residual-controls/report/audit.json configs/experiments/vae_residual/{sd2,sana,flux}.json
```

## Stable Diffusion 3.5 A–G ladder

`model.pipeline: sd35` selects `SD35ViewDenoiser`, using the official SD3 pipeline
components and the pinned SD3.5 Medium checkpoint. The seven YAML configs live
in the existing `trajectory`, `native_multidiffusion`, `implied_endpoint_consensus`
and `vae_residual` categories. The protocol
`configs/experiments/trajectory/sd35-ladder.json` maps each scientific label to
its config and corresponding output directory.

Use the controlled runner for these scientific labels; in particular, D/G's
correction is selected by the runner's phase, not inferred from an experiment
name by the ordinary `generate` CLI. After official sanity and full regression
validation, run and inspect each prerequisite before submitting its successor:

```bash
sbatch --job-name=sd35-A slurm/controlled_ladder.slurm A
# Inspect A before B.
sbatch --job-name=sd35-B slurm/controlled_ladder.slurm B
# Inspect A/B equivalence before C/D.
sbatch --job-name=sd35-CD slurm/controlled_ladder.slurm C D
# Inspect identity recovery before E.
sbatch --job-name=sd35-E slurm/controlled_ladder.slurm E
# Inspect native MultiDiffusion before F, and F before G.
sbatch --job-name=sd35-F slurm/controlled_ladder.slurm F
sbatch --job-name=sd35-G slurm/controlled_ladder.slurm G
```

Completed outputs are never overwritten. For a fresh repetition, copy the
protocol and choose new output paths, then pass it with `--protocol` to
`python -m scripts.controlled_ladder --phase LABEL` in an allocation.

The checkpoint's static shift is 3.0; raw state is 16-channel BCHW at RGB factor
8. Local patches are 1024²; E/F/G use 1024×2048 with three patches and 50% overlap.
Network/VAE weights are bf16; guided predictions and native arithmetic are fp32,
matching the existing SD2/SANA arithmetic convention. All A–G share 40 steps,
CFG 4.5, sequence length 256, seed 0 and the existing native-control prompt.

After G finishes, generate the requested contact sheet and curves:

```bash
python scripts/plot_controlled_ladder.py --protocol configs/experiments/trajectory/sd35-ladder.json
```

Results and the cross-model interpretation are appended under **Stable Diffusion
3.5 A–G Validation** in `NATIVE_CONTROLS_REPORT.md`. No ERP or extra-method
experiments are part of this ladder.

## Experiment H: G with detail-preserving RGB average (all five backends)

H reuses each saved G control (PixelDiT uses F, its pixel-native G equivalent)
with exactly one config-field change: `fusion.mode=detail_preserving_average`.
The manifest is `configs/experiments/vae_residual/h-all-models.json`; each backend
has a corresponding `configs/experiments/vae_residual/<backend>-h.yaml`.
Experiment names retain the baseline setting to keep the complete config
comparison exact; `comparison.json` identifies the scientific label H and the
actual fusion mode is recorded in both the config and consensus audit.

The pipeline reuses `PlanarFusionAccumulator`, with uniform spatial weights,
alpha 1, power 1, and epsilon 1e-6. For each signed, unclamped clean RGB channel,
DPA computes `sum(rgb * (abs(rgb)+epsilon)) / sum(abs(rgb)+epsilon)`.
This is the existing magnitude-based operator, not a learned detail filter.
It applies at every clean-RGB synchronization and to the final decoded local
outputs. Native VAE residuals continue to use G's uniform arithmetic average in
exact raw-native coordinates. The original local implied endpoint is retained.
PixelDiT has no encode, decode, or VAE residual correction.

The runner requires the exact saved global initial tensor and conditioning hash,
matching actual scheduler timesteps/sigmas (or PixelDiT's official schedule),
model metadata, geometry, Python/Torch/CUDA versions, and NVIDIA A40 GPU type.
It checks the complete loaded backend details for SD3.5. No A–G reruns or training
are involved. It refuses to overwrite an H output directory.

After running the CPU regression suite, launch each backend separately:

```bash
sbatch slurm/detail_preserving.slurm sd2
sbatch slurm/detail_preserving.slurm sana
sbatch slurm/detail_preserving.slurm flux
sbatch slurm/detail_preserving.slurm sd35
sbatch slurm/detail_preserving.slurm pixeldit
```

Outputs are stored alongside G at
`outputs/vae-residual-controls/20260910-lookingglass-v1/<backend>/H/`.
Each contains the manifest entry, the saved initialization, a pre-generation
runtime check, a real-VAE identity preflight for latent backends, paired-control
metadata and counts, and `generation/{result.png,final_fused_clean.png,metadata.json,steps.csv}`.

Once all five complete, render and audit without loading models:

```bash
module purge
module load GCC/13.2.0 matplotlib/3.8.2
python scripts/report_detail_preserving.py
```

The report goes to `outputs/vae-residual-controls/report/H/`, with paired full
images, central overlap crops, curves, metrics, and `audit.json`.

## Experiment I: current-state-consistent transition

I derives directly from G, with ordinary uniform RGB fusion and the same
training-free synchronized native VAE residual. Its sole config addition is
`consensus_transition: {mode: preserve_current_state}`. The implicit historical
default remains `preserve_prefusion_endpoint`; `to_dict()` omits that default
field to preserve exact saved A–H config snapshots. G/H files are unchanged.

The four configs are `configs/experiments/vae_residual/<backend>-i.yaml` and the
manifest is `configs/experiments/vae_residual/i-all-models.json`. Use the guarded
runner, which loads G's saved global initialization and checks the complete
config, conditioning, geometry, actual schedule, runtime model details, and
software versions before generation. After the full regression suite passes,
run in this order, with each succeeding only after the preceding job completes:

```bash
sbatch slurm/current_state.slurm sd35
sbatch slurm/current_state.slurm flux
sbatch slurm/current_state.slurm sana
sbatch slurm/current_state.slurm sd2
```

The order can also be enforced with Slurm `--dependency=afterok:<preceding_job>`.
PixelDiT is not part of the Experiment I GPU runs. Results sit beside G/H in
`outputs/vae-residual-controls/20260910-lookingglass-v1/<backend>/I/`.
`comparison.json` identifies I, its transition mode, config, source G, hashes,
job, GPU, runtime, memory, and actual denoiser count. Generation metadata contains
prompt/config and the prepared schedule. `transition_patches.{json,csv}` stores
all per-patch diagnostics; `generation/steps.csv` contains patch mean/max values
at each timestep.

`diffpano/current_state_transition.py` is separate from the unchanged
`EndpointPrediction.reconstruct_next()`. It consumes frozen source states,
corrected clean proposals, and the existing endpoint helper's actual prepared
scheduler coefficients. Flow uses `c + (sigma_next/sigma)*(x-c)`; DDIM uses
`alpha_next*c + (sigma_next/sigma)*(x-alpha*c)`. There is no second prediction.
At current sigma=0, only an already-consistent clean terminal state with
next_sigma=0 is accepted; any other case raises explicitly. No sigma is clamped.

The diagnostic G transition is a counterfactual on I's same source state and
corrected clean, not a sample from the independently evolved historical G run.
Both actual tensor differences and the signed analytical identity are checked:
`next_I-next_G = -(sigma_next/sigma)*alpha*(corrected_clean-original_clean)`.
For flow, alpha=1-sigma. Diagnostics include current-state and original-clean
standard deviations, normalized MAEs (denominator floor 1e-12), and absolute
identity errors. Counterfactual tensors never enter I's trajectory.

Once all four runs complete:

```bash
module purge
module load GCC/13.2.0 matplotlib/3.8.2
python scripts/report_current_state.py
```

This audits the four paired runs and writes images, overlap crops, diagnostic
curves, numerical summary, and audit JSON in
`outputs/vae-residual-controls/report/I/`.

## Experiment J: planar current-state consensus without residual correction

J uses the existing current-state helper with `E(fused_RGB)` as its replacement
clean, and no VAE residual computation or addition. The four configs are
`configs/experiments/implied_endpoint_consensus/<backend>-j.yaml`, with manifest
`j-all-models.json` in that directory. The explicit transition setting is
`{mode: preserve_current_state, vae_residual_correction: false}`. Absent residual
flags retain historical I behavior and serialization; all A–I settings remain
reproducible. J disables the historical diagnostic-only own-RGB encode so it
never computes `z0-E(D(z0))`; each patch has just the fused-RGB encode.

After the full suite passes, run SD3.5, FLUX, SANA, SD2 in that order:

```bash
sbatch slurm/no_residual_planar.slurm sd35
sbatch slurm/no_residual_planar.slurm flux
sbatch slurm/no_residual_planar.slurm sana
sbatch slurm/no_residual_planar.slurm sd2
```

Use Slurm afterok dependencies for strict sequencing. The runner checks the
saved F config, initial tensor and conditioning hashes, actual scheduler and
model metadata, software versions, and GPU type. Results are stored alongside
G/H/I at `outputs/vae-residual-controls/20260910-lookingglass-v1/<backend>/J/`.
The scientific label is J in the comparison JSON; the inherited experiment name
is preserved to keep model-setting comparisons exact. `repository.json` records
HEAD, branch, and dirty worktree status. Metadata includes the full resolved
config, prompt path, seed, schedule, geometry, warp, job, and transition.

Render the F/J/G/I 2x2 after all four complete using the matplotlib module env:

```bash
python -m scripts.report_no_residual_planar
```

The report and paired audit are in `outputs/vae-residual-controls/report/J/`.
The same-current-state diagnostics compare counterfactual F-style and actual J
transitions on J's trajectory; final image comparisons use the saved F/G/I runs.

## Experiment K: fixed perspective-native trajectories with transient ERP RGB

Use `configs/experiments/erp_later/k-all-models.json` and the four `*-k.yaml`
configs. Global pipeline mode `erp_local_current_consensus` requires ERP canvas,
`cube6_fixed` sampling with rotation disabled, standard warp, average/uniform
fusion, and explicit current-state/no-residual transition. The defaults and all
historical A–I configurations remain available unchanged.

Before GPU submission, run the full regression suite and CPU geometry preflight:

```bash
python -m unittest discover -s tests -v
python -m scripts.erp_standard_current_experiment --geometry-only
sbatch slurm/erp_standard_current.slurm all
```

The runner also requires the passed J gate artifact and completed paired J
metadata. `all` processes SD3.5, FLUX, SANA and SD2 in order, records any failure,
and continues independent models before reporting an aggregate failure. Each
model's output directory must not already exist. Do not overwrite earlier runs.
For an individual backend pass its name instead of `all`.

After successful runs:

```bash
python -m scripts.report_erp_standard_current
```

Results live alongside J at
`outputs/vae-residual-controls/20260910-lookingglass-v1/BACKEND/K/`.
`final_erp.png` is fused from the **decoded terminal native states**;
`final_consensus_erp.png` separately saves the last transient clean consensus.
Six `final_view_XX.png` images, five clean-ERP snapshots, contributor count
map/tensor, initial local native tensors/checksums, camera hashes, complete
resolved configuration/runtime/provenance, and per-step/per-view diagnostics
are saved. Reports under `outputs/vae-residual-controls/report/K/` contain full
uncropped J/K comparisons, six-view contacts, diagnostic plots, wrap metrics,
and an audit. Full resolution source images remain available independently of
the contact-sheet display scale.

The ERP contains clean RGB only. No VAE residual, fixed-initial-noise renoising,
ERP latent field, LPW, DPA, time travel, or extra denoiser calls occur in K.

## Experiments L/M: compact dense fixed-view ERP controls

Use `configs/experiments/erp_later/{backend}-{l,m}.yaml` for SD3.5, FLUX,
SANA, SD2 or PixelDiT. Both use the shared dense streaming core; original K is
unchanged. First run `python -m scripts.dense_geometry_preflight`, focused dense
tests, the full unittest suite, compileall and `git diff --check`. The runner
requires the shared passed `outputs/vae-residual-controls/20260915-dense-lm/validation.json`
gate; it must record actual successful validation rather than bypassing checks.

```bash
python -m unittest discover -s tests -p test_dense_consensus.py -v
python -m unittest discover -s tests -v
python -m compileall -q diffpano scripts tests
squeue -u "$USER"
sbatch slurm/dense_erp.slurm configs/experiments/erp_later/sd35-l.yaml
```

The dense launcher uses A100 resources for all five backends. The initial
A40 submissions were cancelled while pending because Grace estimated a 6–9 hour
wait; original K used A40 for the four latent models, so runtime comparisons
are not hardware-matched. Full resolution, native schedules/checkpoints and guidance are
unchanged. Do not rerun an existing output directory. The generic generation
entry point routes dense configs to the same compact runner so it does not
produce extra artifacts.

After all ten runs complete, run `python -m scripts.report_dense_erp`. It audits
settings, actual schedules, hashes, counts and the exact two-file-per-run
storage policy and saves one `K-L-M.png` contact sheet. Model outputs are
`outputs/vae-residual-controls/20260915-dense-lm/{L,M}/{backend}/final_result.png`
and `metadata.json`. Two shared geometry JSONs contain both production-resolution
checks; no intermediate images or tensors are written.

## Dense N/O spatial controls

The N/O configs in `configs/experiments/erp_later/` derive directly from L.
N enables existing DPA; O adds five-level LPW with per-level DPA, no Jacobian
band-confidence heuristic, and periodic ERP reconstruction. All geometry and
native model settings remain L's. Run `sbatch slurm/dense_no_validate.slurm`
and wait for its full regression and actual-L-metadata pairing gate to pass.
Then submit each `*-n.yaml`, followed by each `*-o.yaml`, using
`sbatch slurm/dense_erp.slurm CONFIG`. The runner refuses existing run directories.
Outputs are under `outputs/vae-residual-controls/20260916-dense-no/{N,O}/BACKEND`.
Each has nine 10%-interval predicted-clean consensus snapshots, one terminal
final image and one metadata JSON. After all ten runs, use
`sbatch slurm/dense_no_report.slurm` for the artifact audit and single FLUX
progression image. Existing L/M results are inputs, never overwritten.
