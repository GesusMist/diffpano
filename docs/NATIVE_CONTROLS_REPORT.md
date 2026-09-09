# Native-control implementation and validation report

Inspected local branch: `no_sphere`.
Inspected local commit: `320ad528a566be371f9f2b7cb3f4d3b26bd9c79c`.
The working tree was clean before implementation. Existing configs/tests and
`PLANAR_NATIVE20_GEOMETRY` were preserved.

## Current real-model conclusion (continuation completed)

**For this prompt, seed and prepared schedules, ordinary single-patch native
sampling works for all four backends, while fixed-epsilon x0-renoising visibly
degrades all four.** This implicates the tested single-patch trajectory before
cross-patch noise, fusion or ERP geometry. It does not demonstrate a distinction
between SD2 diffusion and the three flow models: SD2 degrades too.

| Backend | Native MultiDiffusion | Single-patch native good? | x0-renoise good? | Guided predictions A/B |
|---|---|---|---|---|
| SD2 | Meaningful landscape | Yes | No: severe saturation/posterization | 30/30 |
| SANA | Meaningful; central tonal seam | Yes | No: distorted inset, white surround | 20/20 |
| FLUX | Meaningful; vertical tonal seams | Yes | No: stylized/posterized, lost natural detail | 20/20 |
| PixelDiT | Meaningful; vertical sky seams | Yes | No: severe clipping/saturation, lost detail | 50/50 |

All A/B controls passed. The shared Gaussian is saved with a checksum; each pair
uses one backend and conditioning object, one unchanged schedule, identical CFG,
local resolution and initialized state. The four comparisons ran sequentially
in completed Grace job **19699396**, node **g071**, Slurm elapsed **7:54**.
FLUX and PixelDiT single-patch schedules also match their native MultiDiffusion
schedules exactly. No scheduler shift or x0 formula was tuned during validation.

**Decision gates:** none of the single-patch x0 controls is sufficiently valid
for the requested noise-consistency experiments. Therefore independent/global
noise pairs and fusion ablations were intentionally not run. ERP experiments
are not scientifically justified yet. There is no measured noise/fusion/ERP
benefit or degradation to report. Prepared configs remain available.

Evidence artifacts (original runs and failed-job history are detailed below):

- [Native overlap curves](../outputs/native-controls/report/native-overlap.png) and [PDF](../outputs/native-controls/report/native-overlap.pdf).
- [SD2 A/B](../outputs/native-controls/report/sd2-trajectory.png), [SANA A/B](../outputs/native-controls/report/sana-trajectory.png), [FLUX A/B](../outputs/native-controls/report/flux-trajectory.png), [PixelDiT A/B](../outputs/native-controls/report/pixeldit-trajectory.png).
- [Trajectory metrics CSV](../outputs/native-controls/report/trajectory-summary.csv); each comparison also has a PDF and full source JSON.

Limits: one prompt/seed per backend; visual judgments, not a broad benchmark.
Decoded endpoint L1/RMSE measure differences, not quality. Native proposal MAE
units differ by backend. Native MultiDiffusion and single-patch sanity images
use different canvas geometries/noise realizations; only the within-pair A/B
comparison is an exact shared-noise control. These observations identify where
the current formulation fails, not a theorem about diffusion or flow matching.
Earlier “pending/not yet run” entries below are preserved chronological history
and are superseded by the verified outcomes in this section.

## Phase A

Implemented native-state protocol/mixin, native planar sampling and arbitrary-C
uniform fusion, native backend steps for all four models, FLUX actual-patch
geometry validation, scheduler semantics checks, single-patch native/x0
comparison, and optional pre-fusion overlap diagnostics. Details, dimensions,
initialization, solver integration, and decoding behavior are in
[NATIVE_CONTROLS.md](NATIVE_CONTROLS.md).

New files:

- `diffpano/pipelines/native_state.py`
- `diffpano/native_multidiffusion.py`
- `diffpano/overlap.py`
- `diffpano/trajectory.py`
- `scripts/single_patch_trajectory.py`
- `tests/test_native_multidiffusion.py`
- `tests/test_native_schedulers.py`
- Four native and four trajectory YAMLs under `configs/experiments/`
- `prompts/native_control.txt`
- This report and the control guide

Modified files:

- `diffpano/config.py`
- `diffpano/metadata.py`
- `diffpano/pipelines/{sana,flux,sd2,pixeldit,clean_prediction}.py`
- `diffpano/planar_pipeline.py`
- `scripts/generate.py`

Validation in progress on TAMU Grace:

- Static Python compilation/parsing succeeds.
- `git diff --check` succeeds.
- CPU job `19698876`, short partition, exhausted its initial 20-minute limit
  during shared-software dependency imports. No test results were emitted.
  An in-allocation process trace confirmed Python was reading modules under `/sw`.
- Duplicate job `19698878` resulted from an initially timed-out submission;
  it was canceled once both jobs appeared in the queue.
- Replacement CPU job `19698963` uses a one-hour limit and runs
  `python -m unittest discover -s tests -v` after compileall.
- Test output: `logs/phase-a.19698963.{out,err}`.

Phase A CPU gate **passed**: job `19698963` ran all 117 tests successfully in
446.864 seconds (test runtime excludes environment startup). Existing tests pass.
Real model quality remains a separate validation requirement.

## Phase B

Implemented after the 117-test Phase A gate passed:

- Existing independent `FixedPatchNoiseBank` remains unchanged.
- `GlobalNativeNoiseBank` samples one raw global native Gaussian and uses exact
  crops; CPU/GPU/seed storage preserve the same Gaussian stream.
- Actual backend factors are used (configured checkpoints: SANA 32, FLUX/SD2 8,
  PixelDiT 1). Fractional dimensions, strides and origins are rejected.
- Encoded clean and native noise shapes must match exactly before renoising.
- Explicit custom RGB geometry preserves the existing native-20 default.
- Eight noise comparison configs and sixteen controlled fusion configs added.
- Eight new tests cover exact overlaps, storage reproducibility, alignment,
  pre-fusion disagreement, batch/order invariance, shape checks and config controls.

Phase B gate passed: job `19699021` ran all 125 tests successfully in 92.376
seconds. Results: `logs/phase-b.19699021.{out,err}`.

## Phase C

Added four RGB representation configs (completing the 12-entry matrix with
existing native/x0 controls), sixteen planar/ERP paired configs, twenty optional
later ERP configs, and `scripts/run_ablation_matrix.py`. No ERP algorithms
changed. Focused job `19699111` passed all four configuration/orchestration
tests in 4.892 seconds. After disabling SANA tiling, job `19699199` again
passed all four checks in 4.928 seconds. All 72 experiment YAMLs validate. The dry-run commands
for the 12-entry representation matrix and two-backend trajectories also pass.

## Real-model evidence

SD2 job `19699005` completed on Grace node `g086`.

- Output: `outputs/native-controls/sd2-native_multidiffusion-global_native_canvas-average-uniform/20260907-202838-19699005/result.png`
- Visual inspection: recognizable stone ruins in a coherent green valley with
  mountains; useful positive native-baseline evidence, not a claim of universal correctness.
- Recorded runtime: 479.221 seconds (generation entry point, including model
  initialization, excluding Python/module startup).
- Peak GPU memory: 2.466 GiB allocated, 2.883 GiB reserved.
- Actual state: 4 channels, factor 8, 64×128 native canvas, 64×64 patches.
- Actual scheduler: DDIM, order 1, epsilon prediction, initialization sigma 1.
- Three patches, 100% coverage, 50% overlap coverage, 30 timesteps (958 to 1).
- Native pre-fusion overlap MAE mean: 0.006314 at the first step, 0.000721 at the last.

SANA attempt `19699097` completed its native steps but failed in the optional
final AutoencoderDC tiling path (`tile_latent_min_width` missing in the installed
version). New SANA experiment configs now disable tiling and use a full-canvas
decode. The failed job elapsed 2 minutes 41 seconds; generation runtime/peak memory
were not saved because decoding failed. Rerun `19699131` is queued. No SANA
image-quality claim is made yet.

SD2 command:

```bash
sbatch slurm/generate_a100.slurm configs/experiments/native_multidiffusion/sd2.yaml
```

It was launched only after the 117-test Phase A suite passed. No claim is made that the base
code is correct or that x0 consensus is broken. SD2 provides a first meaningful native image; the other backends and
the single-patch trajectory comparisons still require real validation.


## Trajectory validation status

The runner enforces bit-identical initial Gaussian state, uses the same backend,
conditioning, guidance and schedule for A/B, and records N guided predictions
for each path. CPU mocks verified 3 native versus 3 x0 predictions. Configured
real comparisons have SD2 30/30, SANA 20/20, FLUX 20/20, PixelDiT 50/50 predicted
call counts; these real comparisons have **not** been run yet. Thus no conclusion
about x0 trajectory quality follows from the unit tests or SD2 native image.

## Remaining real experiments

After the queued SANA rerun completes, inspect its image, then run FLUX and
1024px PixelDiT native baselines. Next run single-patch comparisons for all four
backends. Only then interpret PixelDiT noise comparisons, aligned latent-noise
comparisons and fusion/ERP controls. All configs and runner commands are prepared;
the expensive matrix has not been automatically submitted.

Final checkout review: branch `no_sphere`, unchanged HEAD
`320ad528a566be371f9f2b7cb3f4d3b26bd9c79c`; implementation is left as local
uncommitted changes. `git diff --check` passes.

## Continuation: SANA native result verified

SANA rerun `19699131` completed on `g100` (Slurm elapsed 2:02).
Command: `sbatch --job-name=native-sana --time=01:00:00 slurm/generate_a100.slurm configs/experiments/native_multidiffusion/sana.yaml`.
Logs: `logs/native-sana.19699131.{out,err}`.
Output: `outputs/native-controls/sana-native_multidiffusion-global_native_canvas-average-uniform/20260907-210237-19699131/result.png` (metadata/config alongside).
The earlier queued status above is superseded by this verified completion; failed attempt `19699097` remains part of the history.

- Visually inspected: coherent recognizable ruins, vegetation, layered mountains and sky. No gross patch discontinuity in the thumbnail; this is one prompt/seed, not a general quality evaluation.
- Runtime 106.992 s; peak allocated 9.572 GiB, reserved 11.445 GiB.
- Raw canvas `[1,32,32,64]`; three `[1,32,32,32]` patches, stride 16; local RGB 1024 square, output 1024×2048.
- `DPMSolverMultistepScheduler`, solver order 1, flow prediction, flow sigmas, init sigma 1; configured flow shift 3.
- 20 timesteps: 999,982,963,944,922,899,874,847,817,785,749,710,666,617,562,499,428,345,249,136.
- Pre-fusion overlap mean MAE first/last: 0.010207281 / 0.021306151; max MAE 0.010333546 / 0.021306608. Two neighbor pairs, full coverage.
- MAE rises while the output remains meaningful. Absolute native MAE values are not directly comparable across different latent representations.

## Continuation: FLUX preflight

The prepared FLUX YAML validated against cached model configs and the installed
`FlowMatchEulerDiscreteScheduler`, without loading weights on the login node.
Artifact: `outputs/native-controls/flux-preflight.json` (full timestep/sigma arrays).
Raw canvas `[1,16,128,256]`, raw patch `[1,16,128,128]`, RGB patch 1024 square;
actual packed grid 64×64, packed shape `[1,4096,64]`, image IDs `[4096,3]`.
Image ID row/column ranges were asserted against the actual patch geometry.
`image_seq_len=4096`, dynamic shift `mu=1.1500000000000001`; configured static
shift is 3 but the scheduler's dynamic-shift path is enabled. Twenty unshifted
input sigmas are shifted once, producing timesteps 1000 to 142.52936 and a
terminal sigma 0. No scheduler parameters were tuned.

The first two lightweight preflight attempts failed to import `packaging`
because the command overwrote module-provided PYTHONPATH. Preserving PYTHONPATH
resolved it; no packages or model math changed. Successful preflight used:
`module load GCC/13.2.0 OpenMPI/4.1.6 Python/3.11.5 PyTorch/2.7.0`, the existing
venv, and `/tmp/diffpano_flux_preflight.py`. GPU launchers already preserve that path.

Experiment metadata now additionally records full schedules and node; trajectory
metadata explicitly records the shared A/B controls and resolved prompt texts.
These are logging-only additions. Added `slurm/trajectory_a100.slurm` using the
existing GPU launcher environment, and `scripts/plot_overlap_mae.py` using
matplotlib's default colors, separate axes and metrics read from JSON.

FLUX submitted as job `19699367`, running on `g035`:
`sbatch --job-name=native-flux --time=01:00:00 slurm/generate_a100.slurm configs/experiments/native_multidiffusion/flux.yaml`.
Logs: `logs/native-flux.19699367.{out,err}`. Result pending inspection.

### FLUX native completion

Job `19699367` completed on `g035`. Output:
`outputs/native-controls/flux-native_multidiffusion-global_native_canvas-average-uniform/20260907-213330-19699367/result.png`.
Runtime 160.278 s; peak allocated 35.665 GiB, reserved 37.055 GiB.
Actual geometry matches preflight: raw canvas `[1,16,128,256]`, three raw
`[1,16,128,128]` patches, packed `[1,4096,64]`, RGB 1024 square.
First-order FlowMatch Euler, dynamic mu 1.15. Full actual timestep/sigma arrays
in metadata were asserted exactly equal to `flux-preflight.json`.
Mean overlap MAE first/last 0.009314779 / 0.016891429;
maximum 0.010160149 / 0.018564112; two neighbor pairs, full coverage.
Visual inspection: meaningful ruins/valley/mountains, but vertical tonal
boundaries are visible, especially in the sky. Thus this is a functioning native
baseline with visible artifacts, not a seam-free success. The one-patch native
control is still required before attributing those boundaries to patch sampling
versus output decoding.

Reporting check job `19699369`: all 11 focused native/trajectory tests passed
in 10.077 s. Plotting then failed because the CPU environment lacked matplotlib.
Using existing Grace `GCC/13.2.0 matplotlib/3.8.2` modules resolved the plot-only
failure; no packages were installed. Initial SD2/SANA plots generated at
`outputs/native-controls/report/native-overlap.{png,pdf,json}`; these will be
refreshed to include the remaining native runs.

PixelDiT native job `19699380` submitted and running on `g035`:
`sbatch --job-name=native-pixeldit --time=01:00:00 slurm/generate_a100.slurm configs/experiments/native_multidiffusion/pixeldit.yaml`.
Logs: `logs/native-pixeldit.19699380.{out,err}`. Prepared local patches remain
1024 square, canvas 1024×2048, three patches, flow shift 4, 50 first-order steps.

### PixelDiT native completion

Job `19699380` completed on `g035`. Output:
`outputs/native-controls/pixeldit-native_multidiffusion-global_native_canvas-average-uniform/20260907-213819-19699380/result.png`.
Runtime 71.711 s; peak allocated 9.077 GiB, reserved 9.520 GiB.
Raw pixel canvas `[1,3,1024,2048]`, three 1024-square patches, stride 512.
Pinned official commit `41f73006ae532b0b41fee72b181dc22891a5a01a`, flow shift 4.
50 guided first-order steps use official `model_fn` and `dpm_solver_first_update`;
no order-2 reference is used for this control. Full 51-boundary flow schedule
saved in metadata: first 0.9997498393, last model time 0.0754005387, terminal 0.
Mean overlap MAE first/last 0.005885701 / 0.006499934;
maximum 0.006140129 / 0.006644096; two neighbor pairs, full coverage.
Visual inspection at 512-wide preview: detailed meaningful ruins and vegetation,
with visible vertical changes in the sky near patch transitions. Since pixels
are the native state here, this artifact cannot be attributed to VAE decoding.
It remains necessary to check the ordinary one-patch baseline before assigning
all artifacts to MultiDiffusion.

### Single-patch A/B jobs

Submitted the four prepared trajectory configs sequentially in job `19699396`:
`sbatch --job-name=trajectory-controls slurm/trajectory_a100.slurm configs/experiments/trajectory/sd2.yaml configs/experiments/trajectory/sana.yaml configs/experiments/trajectory/flux.yaml configs/experiments/trajectory/pixeldit.yaml`.
Time limit reduced to one hour while queued, since native timings indicate ample
headroom. Logs: `logs/trajectory-controls.19699396.{out,err}`.
Each comparison retains both images, initial epsilon, resolved config and
trajectory JSON with node, schedule, effective prompts, guidance configuration,
bit-exact initialization assertions and equal guided-prediction counts.
Only these four trajectory configs were submitted; later experiments remain gated.

All four native overlap curves are now saved in
`outputs/native-controls/report/native-overlap.png` and `.pdf`, with source paths
and measured first/last values in `.json`. Panels retain each backend's own
native units; values are read from metadata, not hardcoded.

### Full native-curve interpretation

Inspecting the full plotted curves changes the endpoint-only interpretation.
SD2's mean MAE reaches **0.019480073 at step 28**, then drops to 0.000720657
at step 29. It does **not** steadily converge from its initial value.
SANA minimum is 0.007300809 at step 2, FLUX minimum 0.004202454 at step 8,
and PixelDiT minimum 0.000503742 at step 20; all three rise toward the end.
These are differences between proposed next states; their scale depends on the
scheduler update and native representation. Endpoints or curve shapes alone do
not establish trajectory quality or explain final seams.

Larger 512-wide native previews were also inspected while waiting for the A/B
allocation. SANA has a visible central vertical tonal transition in the sky/
mountains, which was less apparent in its first small thumbnail. FLUX shows
vertical transitions at several patch boundaries. SD2 remains recognizable
without a comparably obvious sky seam in this composition. These observations
qualify the initial small-preview impressions; original images remain unchanged.

### SD2 single-patch trajectory (verified)

Job `19699396`, node `g071`, config `configs/experiments/trajectory/sd2.yaml`.
Directory: `outputs/native-controls/sd2-trajectory-global_native_canvas-average-uniform/20260907-214643-19699396/`.
Both `native_final.png` and `x0_renoise_final.png` inspected. Ordinary native
sampling produces a meaningful natural-color ruin/valley image. Fixed-epsilon
x0-renoising produces severely oversaturated, posterized/neon colors with degraded
detail. This is **native good + x0 bad** for this seed/prompt.
All fairness fields pass: same backend/conditioning/guidance/resolution,
bit-identical initialized state, unchanged epsilon/schedule, 30/30 guided
predictions; shared original epsilon and SHA-256 saved. Local RGB 512 square.
Runtime for the pair 128.764 s; peak allocated 2.467 GiB.
Decoded endpoint L1 0.603859, RMSE 0.747165 (differences, not quality scores).
Last native-state std 0.810827 versus x0 predicted-clean std 1.631095;
these are endpoint native-state quantities and support a substantial divergence.
This result already contradicts an assumption that SD2 x0 must work while only
flow models degrade. No sampling formulas or parameters were changed.

### SANA single-patch trajectory (verified)

Same job/node, config `configs/experiments/trajectory/sana.yaml`.
Directory: `outputs/native-controls/sana-trajectory-global_native_canvas-average-uniform/20260907-214912-19699396/`.
Both endpoints inspected: native is a coherent ruin/valley landscape; x0-renoise
collapses to a small distorted, highly saturated scene surrounded by mostly
white space. **Native good + x0 bad** in this control.
All shared-control fields pass; 20/20 guided predictions, RGB 1024 square,
unchanged schedule (999 to 136), same initialized epsilon, prompt, effective
negative prompt, conditioning and guidance 4.5. No tiling in either endpoint.
Runtime 63.995 s; peak allocated 9.571 GiB. Endpoint decoded L1 1.064878,
RMSE 1.251515; last native-state std 0.886436, predicted-clean std 1.547397.
The quality failure occurs with one patch and before any multi-patch fusion,
so noise-consistency/fusion experiments for SANA do not pass the requested gate.

### FLUX single-patch trajectory (verified)

Job `19699396`, node `g071`, config `configs/experiments/trajectory/flux.yaml`.
Directory: `outputs/native-controls/flux-trajectory-global_native_canvas-average-uniform/20260907-215022-19699396/`.
Native endpoint is a coherent natural landscape with ruins. X0-renoise retains
recognizable objects but has strong posterization, unnatural saturation and loss
of natural texture/detail. **Native good + x0 degraded**; the latter is not a
quality-preserving local control. All fairness fields pass; 20/20 guided
predictions, RGB 1024 square, guidance 3.5, true CFG 1.0, mu 1.15. Full actual
schedule/sigmas equal the preflight and native MultiDiffusion schedule exactly.
Runtime 109.187 s; peak allocated 35.664 GiB. Endpoint L1 0.460802,
RMSE 0.613042; native std 0.927637 versus x0 predicted-clean std 1.240468.
Ordinary single-patch FLUX works, so the native MultiDiffusion tonal seams are
not evidence that standalone FLUX generation is broken. Their exact mechanism
remains unisolated (patch context/averaging/output decode are not separately tested).

### PixelDiT single-patch trajectory (verified)

Same job/node, config `configs/experiments/trajectory/pixeldit.yaml`.
Directory: `outputs/native-controls/pixeldit-trajectory-global_native_canvas-average-uniform/20260907-215220-19699396/`.
Native endpoint is a coherent detailed ruin/vegetation landscape. X0-renoise is
severely posterized/saturated with large clipped black/white regions and lost
detail. **Native good + x0 bad**, without any VAE or cross-patch fusion.
All fairness fields pass; 50/50 guided predictions, RGB 1024 square, official
first-order DPM in A, same official clean predictor in B, CFG 2.75 and flow shift 4.
Full 51-boundary flow schedule equals the native MultiDiffusion schedule.
Effective negative prompt is recorded as `low quality, worst quality,
over-saturated, blurry, deformed, watermark` in both paths.
Runtime 66.889 s; peak allocated 9.077 GiB. Endpoint L1 0.577519,
RMSE 0.759862; native std 0.570231 versus predicted-clean std 0.912827.
The PixelDiT independent/global pixel-noise experiment therefore does not pass
the explicit prerequisite of reasonably valid single-patch x0 generation.

### Interpretation and reporting artifacts

The A/B results provide experimental evidence that this repeated clean-prediction/
fixed-initial-epsilon renoising trajectory does not preserve generation quality
under the tested settings. Correct local noising algebra and successful unit tests
do not by themselves establish that the resulting iterative trajectory is viable.
All four ordinary native single-patch controls are meaningful; degradation also
occurs in pixel space, so neither VAE boundaries nor multi-patch fusion are
necessary for the observed x0 failure. SD2 also fails, so this experiment does
not support the proposed “SD2 works, flow models fail” split.

Comparison sheets and CSV are generated by `scripts/summarize_trajectories.py`
from the four saved `trajectory.json` paths above. The script checks shared-control
flags, equal guided-prediction counts, configured step count and exact equality
between stored scheduler timesteps and per-step records before rendering.
Each endpoint file remains unmodified. No selected intermediate tensors were
retained by the existing runner; per-step means/stds and sigma/alpha/time values
are in the trajectory JSON files.

Pair runtimes include model setup; Slurm elapsed additionally includes environment
startup and sequential process overhead. GPU memory is peak allocated memory for
the whole pair; reserved memory is also saved in each trajectory JSON.

### Final verification

All four saved `initial_epsilon.pt` tensors were independently reloaded on CPU.
Their SHA-256 values match the trajectory JSON; shapes match the recorded native
shapes; each tensor is bit-identical to regeneration with its recorded seed and
CPU Gaussian generator. Evidence:
`outputs/native-controls/report/epsilon-verification.json`.
All comparison-sheet metadata assertions passed, and PNG/PDF sheets plus CSV
were generated successfully for all four backends.

Current verification: previous full 125-test gate passed; this continuation's
11 focused native/trajectory tests passed; actual FLUX and PixelDiT generation
and all four actual A/B runs completed. Plot generation and source compilation
succeeded after loading the existing matplotlib module. `git diff --check`
passes. No inference jobs remain in `squeue -u $USER`. Branch and HEAD remain
`no_sphere` / `320ad528a566be371f9f2b7cb3f4d3b26bd9c79c`; all implementation
and reporting work remains uncommitted. No changes were reset or discarded.
