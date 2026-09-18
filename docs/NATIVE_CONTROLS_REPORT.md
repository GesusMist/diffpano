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

## Model-implied endpoint experiment (September 9 continuation)

Starting checkout: clean `no_sphere`, HEAD
`1a37478ca817e7c20fa46ab9c721912e760e4cd1`. Earlier implementation/history was
already committed at this starting point; nothing was reset or discarded.
No Slurm jobs were initially queued.

Added `diffpano/pipelines/endpoints.py`, an endpoint reconstruction interface in
`native_state.py`, and small backend methods consuming existing prediction caches.
SD2 retains DDIM epsilon and clipped/thresholded clean conversion plus the actual
final-alpha convention. SANA/FLUX use exact current/next prepared sigmas; FLUX
unpacks its cached velocity in actual patch geometry without reapplying shift.
PixelDiT retains the official native clean prediction and derives its noise
endpoint using the actual shifted flow time; zero-time termination is explicit.
Existing native and fixed-noise transition code is retained. Backend counters
record guided predictions at prediction entrypoints.

`compare_three_way` in `diffpano/trajectory.py` adds three independent states and
a one-step audit along A, using the SAME cached native prediction for native,
implied, and fixed proposals. It requires one guided prediction per trajectory
per step, identical initialization, unchanged conditioning/schedules/settings,
and finite states. Three decodes occur only after the loop. Existing
`compare_single_patch` and `scripts/single_patch_trajectory.py` remain available.
New runner `scripts/three_way_trajectory.py` uses the prepared trajectory configs,
saving to a separate `outputs/endpoint-controls/` group with per-trajectory
subdirectories, full JSON, CSV, shared epsilon, runtime and memory.
Launcher: `slurm/endpoint_trajectory_a100.slurm`.

Tests added in `tests/test_endpoint_trajectory.py`: DDIM oracle including
clipping/thresholding/final alpha, straight-flow endpoints, both actual shifted
flow schedulers, PixelDiT near-zero/terminal algebra and official first update,
real adapter cache/packing/counter routes, prediction reuse, initialization and
conditioning fairness, extra-forward rejection, and unchanged old A/B outputs.
Full regression command: `python -m unittest discover -s tests -v`, under the
existing Grace GCC/OpenMPI/Python/PyTorch modules and venv.
CPU submission: `sbatch /tmp/diffpano_endpoint_tests.slurm`, job `19716606`.
GPU submissions remain gated on the completed full regression suite.

First full CPU run `19716606` passed 137 tests in 95.071 s. Two additional
actual-adapter integration tests were added after that process loaded its test
module, so a second full run is required before GPU submission. Those tests
exercise all four actual adapter prediction/packing/cache/counter routes with
small deterministic networks. New reporting script:
`scripts/plot_endpoint_controls.py` (default matplotlib colors, image triptychs,
per-step errors and aggregate CSV/JSON; ratio denominator floor explicitly 1e-12).

Final full CPU gate **passed**: job `19716610`, all **139 tests** in 3.456 s
(excludes module/Python startup); logs `logs/endpoint-tests.19716610.{out,err}`.
Both new actual-adapter tests passed, as did the full existing suite. The earlier
137-test run also passed. No numerical test failures required changing formulas.
Independent GPU command per backend, submitted only after this gate:
`sbatch --job-name=endpoint-BACKEND slurm/endpoint_trajectory_a100.slurm configs/experiments/trajectory/BACKEND.yaml`.
No noise/fusion/ERP jobs are included.

GPU jobs: SD2 `19716612`, SANA `19716613`, FLUX `19716614`, PixelDiT `19716615`.
Logs: `logs/endpoint-BACKEND.JOBID.{out,err}`. Each is an independent A100 job.
While queued, their limits were reduced to 20 minutes using
`scontrol update JobId=JOBID TimeLimit=00:20:00` to improve backfill scheduling;
previous comparable real runs took about 1–3 minutes after interpreter startup.
They were submitted separately as requested; all are initially queued.

This experiment should not automatically be called classical Time Travel.
Timestep revisiting and residual/noise endpoint selection are separate choices.
Here the controlled change is fixed original Gaussian versus model-implied
endpoint while using the same scheduled timesteps. No same-timestep Time Travel
or new panorama/global consensus method is implemented in this task.

One-step tables will report arithmetic mean MAE across all scheduled native
states. Ratios use `native_fixed_mae / max(native_implied_mae, 1e-12)`;
measured zeros will remain zeros, with the finite denominator floor disclosed.
Per-step CSV/JSON also retain RMSE, maximum absolute error, current/next
alpha/sigma and mean/std/min/max for each trajectory. B's existing final output
is its last clean prediction; the one-step fixed counterfactual uses the native
scheduler's next-state coefficients even at DDIM's terminal alpha. These are
explicitly different diagnostic quantities, and B's historical semantics remain
unchanged.

SD2 attempt `19716612` failed in the new fairness audit after model evaluation:
`DDIMScheduler.add_noise` legitimately relocates `alphas_cumprod` from CPU to
CUDA. Direct `torch.equal(current, snapshot)` raised a device mismatch rather
than checking the unchanged coefficient values. The audit now compares on the
snapshot device; schedule values, dtype and all sampling formulas are unchanged.
This is a diagnostic bookkeeping fix, not a trajectory-math change. The full CPU
suite is being rerun before replacement SD2 submission. Failed logs are preserved.
SANA `19716613` completed; numerical/visual results are being inspected.

### SANA three-way result

Job `19716613`, node `g013`, output directory:
`outputs/endpoint-controls/sana-trajectory-global_native_canvas-average-uniform-threeway/20260909-041049-19716613/`.
Runtime 215.744 s including model setup; peak allocated 9.572 GiB.
20 guided predictions each for native/fixed/implied, zero extra diagnostic
predictions. Mean one-step native–implied MAE **2.49865538e-8**, native–fixed
MAE **0.0480829512**, ratio **1,924,353**; maximum implied absolute error across
all steps 9.53674316e-7. Complete schedule and per-step statistics are saved.
Visual inspection: native **Good**, fixed **Strong degradation** (distorted
saturated inset with white surround), implied **Good** (coherent landscape
matching native composition). Native and fixed PNGs are byte-identical to the
previous two-way experiment, confirming those controls were preserved.
Full endpoint RGB native–implied MAE 0.0126991, RMSE 0.0299042, maximum 1.80664;
therefore full sampling is not bit-exact even though one-step errors are tiny.
The mean endpoint difference and visual result must be considered alongside the
large localized maximum; no severe implied-trajectory degradation is observed.

Audit-fix CPU run `19716619` passed all 139 tests in 26.276 s. A new targeted
mixed-precision oracle was then added for an observed FLUX numerical difference;
a final full suite will include that test before replacement SD2 submission.

FLUX's installed `FlowMatchEulerDiscreteScheduler.step` computes
`sample + (sigma_next-sigma)*model_output`, then casts the entire result to
`model_output.dtype` (bfloat16 here). The scalar-times-bfloat16 velocity product
also uses bfloat16 arithmetic. The endpoint reconstruction uses float32 native
coordinates, so its mathematically equivalent operation need not match this
quantized transition bit-for-bit. The new oracle separately checks exact native
scheduler precision behavior and float32 endpoint/Euler algebra. Neither native
nor C is altered merely to force a zero diagnostic error.

Final full regression run `19716620` **passed all 140 tests** in 3.512 s,
including the new mixed-bfloat16 FlowMatch Euler oracle and the SD2 audit fix.
The oracle confirms that endpoint reconstruction matches float32 flow algebra,
while the installed native scheduler exactly matches its quantized bfloat16
update. Replacement SD2 was submitted only after this full gate.

### FLUX three-way result

Job `19716614`, node `g051`, output:
`outputs/endpoint-controls/flux-trajectory-global_native_canvas-average-uniform-threeway/20260909-041112-19716614/`.
Runtime 257.574 s; peak allocated 35.665 GiB; 20/20/20 guided predictions.
Mean one-step native–implied MAE **0.000940119688**, native–fixed MAE
**0.0228482181**, ratio **24.3035**; maximum implied absolute error 0.0158534.
This is not float32-roundoff equivalence: see the measured bfloat16 scheduler
rounding distinction above. Prepared shifted sigmas and mu 1.15 are unchanged.
Visual inspection: native **Good**, fixed **Strong degradation** (posterized,
unnatural color and lost detail), implied **Good**, preserving natural landscape
quality and composition. Full RGB native–implied MAE 0.0185235, RMSE 0.0449660,
maximum 1.3828125. Endpoint preservation restores quality in this control even
though full images and native update arithmetic are not bit-identical.

### PixelDiT three-way result

Job `19716615`, node `g013`, output:
`outputs/endpoint-controls/pixeldit-trajectory-global_native_canvas-average-uniform-threeway/20260909-041514-19716615/`.
Runtime 74.951 s; peak allocated 9.077 GiB; 50/50/50 guided predictions.
Mean one-step native–implied MAE **2.56332985e-8**, native–fixed MAE
**0.0142155251**, ratio **554,572.6**; maximum implied absolute error
9.53674316e-7. Official first-order DPM, flow shift 4, 1024-square pixel state,
full shifted flow schedule unchanged. Visual inspection: native **Good**, fixed
**Strong degradation** (clipped/posterized high-contrast colors), implied **Good**
with coherent detailed ruins/vegetation. Full RGB native–implied MAE 0.00733768,
RMSE 0.0100520, maximum 0.550265. No VAE or cross-patch fusion is involved.

Replacement SD2 job: `19716621`, command
`sbatch --job-name=endpoint-sd2 --time=00:20:00 slurm/endpoint_trajectory_a100.slurm configs/experiments/trajectory/sd2.yaml`.
It is queued after the final 140-test pass. Native and fixed final PNGs for both
FLUX and PixelDiT were also verified byte-identical to their prior A/B experiment
outputs, matching the earlier SANA preservation check.

Slurm accounting: initial SD2 diagnostic failure `19716612` elapsed 5:41
(exit 1), SANA `19716613` completed in 5:40, FLUX `19716614` in 5:18,
PixelDiT `19716615` in 1:36. These include cold shared-filesystem startup and
are distinct from generation runtimes above. The final CPU gate `19716620`
completed on `c262`. Replacement SD2's queued limit was further reduced to
10 minutes (`scontrol update JobId=19716621 TimeLimit=00:10:00`) based on the
observed SD2 startup/runtime, to fit shorter backfill gaps without changing any
experimental settings.

One-step and independently accumulated trajectory errors are different measures.
The final native-state A/C MAEs are SANA 0.0106511, FLUX 0.0155538, and PixelDiT
0.00733768. SANA starts at 3.30e-8 and PixelDiT at 3.02e-8 one-step/state error;
those small differences accumulate during independent model evaluations. FLUX
starts at 0.00110835 due to the precision difference already identified.
Thus the completed flow controls reproduce native generation quality and closely
track its transition, but do not establish bit-exact full-trajectory equivalence.
The observed good final images do not exhibit the severe degradation described
in the user's Case C.

### SD2 three-way result (replacement completed)

Job `19716621` completed on `g009`, Slurm elapsed 6:25, exit 0. It started before
the attempted CPU/RAM reduction could apply, so it retained 8 CPUs and 64 GiB
host RAM. The diagnostic-device fix resolved the failed attempt; no sampling
formula changed. Output:
`outputs/endpoint-controls/sd2-trajectory-global_native_canvas-average-uniform-threeway/20260909-043714-19716621/`.
Runtime 244.103 s; peak allocated 2.466 GiB, reserved 2.869 GiB; 30/30/30 guided
predictions. DDIM epsilon prediction, eta 0, clipping/thresholding off,
`set_alpha_to_one=False`, leading timesteps, offset 1, exactly as the native
reference. Actual final-alpha handling is included in C and the diagnostic.
Mean one-step native–implied MAE **1.93732793e-10**, native–fixed MAE
**0.0794849717**, ratio **410,281,453**; maximum implied absolute error
2.38418579e-7. Native **Good**, fixed **Strong degradation**, implied **Good**:
C restores the natural-color ruins/valley composition and detail.
Full native-state A/C MAE 0.00432756; final RGB MAE 0.00307721, RMSE 0.00540950,
maximum 0.140091. These independently sampled endpoints are not bit-identical.

### Final four-backend result

| Backend | Native | Fixed initial noise | Model-implied endpoint |
|---|---|---|---|
| SD2 | Good | Strong degradation: neon/posterized colors | Good: natural detail restored |
| SANA | Good | Strong degradation: distorted inset/white surround | Good: coherent landscape restored |
| FLUX | Good | Strong degradation: posterized color/detail | Good: natural landscape restored |
| PixelDiT | Good | Strong degradation: clipped/posterized detail | Good: detailed pixel-native result restored |

Arithmetic mean one-step MAE across each backend's actual native trajectory:

| Backend | Native–implied MAE | Native–fixed MAE | Fixed/implied ratio |
|---|---:|---:|---:|
| SD2 | 1.93733e-10 | 0.0794850 | 4.10281e8 |
| SANA | 2.49866e-8 | 0.0480830 | 1.92435e6 |
| FLUX | 9.40120e-4 | 0.0228482 | 24.3035 |
| PixelDiT | 2.56333e-8 | 0.0142155 | 554573 |

Ratio denominator is `max(native_implied_mae, 1e-12)`. All measured aggregate
implied errors exceed that floor; no zero or infinite result is fabricated.
Absolute native error scales are backend-specific; ratios compare the two
transitions within the same model and native representation.

**Conclusion:** preserving the model-implied endpoint restores generation quality
for all four tested backends. Extracting clean x0 is not by itself the cause of
the prior severe degradation. Replacing the associated model-implied residual/noise
endpoint with the original fixed Gaussian is strongly supported as the major
error in this tested trajectory. This supports the user's Case A in quality and
near-native transition behavior, with the explicit FLUX precision qualification.
It does not show a diffusion-versus-flow split.

Reproduction is numerical/qualitative, not bit-exact across full trajectories.
SD2/SANA/PixelDiT one-step errors are near floating-point roundoff. FLUX's larger
error reflects the verified native bfloat16 update/product rounding versus C's
float32 endpoint reconstruction. All four independently evaluated C trajectories
remain visually good despite accumulated numerical differences. These results
cover one fixed prompt/seed and the prepared first-order schedules, not a general
benchmark or proof for other solvers, guidance settings or resolutions.

Final evidence:

- [SD2 triptych](../outputs/endpoint-controls/report/sd2-threeway.png), [SANA triptych](../outputs/endpoint-controls/report/sana-threeway.png), [FLUX triptych](../outputs/endpoint-controls/report/flux-threeway.png), [PixelDiT triptych](../outputs/endpoint-controls/report/pixeldit-threeway.png).
- [SD2 errors](../outputs/endpoint-controls/report/sd2-one-step.png), [SANA errors](../outputs/endpoint-controls/report/sana-one-step.png), [FLUX errors](../outputs/endpoint-controls/report/flux-one-step.png), [PixelDiT errors](../outputs/endpoint-controls/report/pixeldit-one-step.png).
- [Numerical CSV](../outputs/endpoint-controls/report/summary.csv) and [JSON](../outputs/endpoint-controls/report/summary.json). All eight figures also have PDF versions.
- Each source run directory contains `trajectory.json`, `steps.csv`, `initial_epsilon.pt`, resolved `config.yaml`, and separate `native/`, `fixed_initial_noise/`, `model_implied_endpoint/` image directories.

Figure command: `python scripts/plot_endpoint_controls.py --output outputs/endpoint-controls/report` followed by the four source `trajectory.json` paths listed above, using Grace `GCC/13.2.0 matplotlib/3.8.2`. Figures read measured data directly, use default colors, and display actual scheduler timesteps. No seaborn is used.

### Implication for later panorama work — documented only

A candidate next method is to predict each local clean endpoint and its associated
model-implied residual/noise endpoint together, decode the clean endpoint, fuse
clean RGB globally, extract each fused clean patch/view, encode it for latent
backends, then reconstruct the next noisy state from that fused clean endpoint
and the retained LOCAL model-implied endpoint. PixelDiT can use the fused clean
pixels directly. This differs from discarding the model-implied endpoint and
reinserting the original Gaussian after every clean prediction.

No such global pipeline is implemented here. Single-patch success does not
validate VAE roundtrips, global fusion, changing view geometry or ERP projection.
Shared-noise, fusion and ERP experiments remain deferred to a separate task.
The fixed-original-noise method should not automatically be equated with classical
Time Travel; timestep revisiting and endpoint choice both matter. Same-timestep
Time Travel is not added in this experiment.

### Final artifact and checkout verification

Independent CPU artifact audit passed for **all four** backends: saved epsilon
matches its SHA-256 and bit-exact regeneration from seed 0; native shapes and
actual model counts match; per-step timesteps equal the frozen schedule. Epsilon,
schedule and effective prompts also match the prior A/B experiment. Every native
and fixed-noise PNG is byte-identical to its corresponding prior A/B PNG.
Evidence: [artifact-verification.json](../outputs/endpoint-controls/report/artifact-verification.json).
C is an independent full trajectory with N guided predictions, not a reused native
image or an extra-evaluation correction pass. All comparison images and error
plots were generated successfully from saved results.

Full regression suite: **140 passed**, final job `19716620`. All four real-model
controls completed; initial failed SD2 audit history is retained above.
`git diff --check` passes. No Slurm jobs remain queued/running for this task.
HEAD remains `1a37478ca817e7c20fa46ab9c721912e760e4cd1` on `no_sphere`; changes
remain local and uncommitted. No previous implementation or outputs were reset,
cleaned, discarded or overwritten. Existing two-way CLI/config semantics remain
available; all new three-way images and metrics use `outputs/endpoint-controls/`.

## Multi-patch RGB implied-endpoint consensus (September 9)

Starting HEAD `1a37478ca817e7c20fa46ab9c721912e760e4cd1`, branch `no_sphere`.
The existing uncommitted native/trajectory work and all previous outputs were
preserved. `squeue -u $USER` was empty before this task.

The additive `implied_endpoint_consensus` mode uses
`diffpano/implied_endpoint_consensus.py`. Persistent diffusion state is a list
of local noisy native patches. There is **no persistent global native diffusion
canvas and no global endpoint canvas**. Global synchronization is uniform
averaging of predicted-clean RGB. A model recomputes each patch's endpoint at
every timestep; that endpoint exists only within the step and is discarded after
all next states have been constructed. Predictions use frozen local states;
canonical-order RGB accumulation makes model processing order irrelevant.

The existing `EndpointPrediction.reconstruct_next` now accepts a replacement
`clean=` argument. This reuses the previously validated SD2/SANA/FLUX/PixelDiT
coefficients and endpoint mathematics without changing its existing call behavior:
`next_alpha * fused_clean_native + next_sigma * original_local_endpoint`.
In flow notation, `(x0_hat, x1_hat)` becomes `(x0_fused, x1_hat)`. It does not
replace the endpoint by `x0_fused + velocity`, fuse endpoints, or reuse the
original Gaussian after initialization. Latent models always decode/encode the
clean patches each step using the existing deterministic VAE mode/scaling/shift
conventions. PixelDiT bypasses VAE operations entirely.

Geometry comes from each successful native config and its loaded backend's
integer native/RGB factor: SD2 RGB 512x1024, 512-square patches, stride 256;
SANA/FLUX/PixelDiT RGB 1024x2048, 1024-square patches, stride 512. All have three
patches and complete coverage. `PLANAR_NATIVE20_GEOMETRY` and old modes are
unchanged. Patch batching is explicitly unsupported and non-1 patch batch sizes
fail validation; generation batch size retains its independent meaning.

Final `result.png` is uniform RGB averaging of decoded terminal local native
states after all scheduled reconstructions. This includes DDIM's actual terminal
alpha convention. The last globally fused clean prediction is also saved as
`final_fused_clean.png`, allowing output decoding/projection effects to be
inspected separately. Neither image changes the sampling trajectory.

New paired configs: `configs/experiments/implied_endpoint_consensus/{sd2,sana,flux,pixeldit}.yaml`.
`python -m scripts.paired_rgb_endpoint --config CONFIG` runs a fresh native
MultiDiffusion reference and the new pipeline with the same backend, prepared
conditioning, frozen schedule/settings, and **same global GPU-generated Gaussian
field**. Every initialized local tensor is asserted bit-identical to the matching
native crop. The global initialization is saved as `initial_native.pt` with hash.
The reference is freed before the new loop, which receives only local states.
Actual guided predictions must equal patches times timesteps for both methods.

Latent one-patch controls use the same CPU-seeded Gaussian convention as the
validated earlier single-patch experiment, unchanged local resolution and full
schedule. They compare independent implied trajectories with and without the
intentional RGB roundtrip; no extra denoiser evaluations are used for diagnostics.
They save both images, epsilon, decoded numerical differences, per-step VAE
projection error, runtime and allocated/reserved memory. No PixelDiT roundtrip
control is needed.

Diagnostics include pre-fusion RGB neighboring-overlap MAE mean/max, mean absolute
RGB consensus correction, and latent `encode(decode(x0)) - x0` MAE. The latter
requires one extra VAE encode of the original clean RGB per patch, explicitly
counted as diagnostic overhead rather than denoiser work. Native overlap metrics
and these RGB metrics are not compared as if they had common units.

`diffpano/seams.py` computes a common final-image diagnostic on identically loaded
PNG RGB [0,1]: average absolute gradient across internal patch start/end lines,
compared with gradient lines within +/-8 pixels, excluding all exact boundaries.
It reports boundary gradient, nearby gradient, difference and ratio with 1e-12
floor. Lines, then active orientations, contribute equally. This is sensitive
to natural scene edges and smoothing, so image quality/visual seams must also be
considered. `scripts/plot_rgb_endpoint_controls.py` creates paired PNG/PDF images,
latent one-patch comparison sheets, diagnostic plots and CSV/JSON summaries.

Tests cover exact global-crop initialization, overlap fusion identity, synchronous
order invariance, a nonzero cross-patch-correction Jacobi oracle, one prediction
per patch/time, endpoint (not velocity) preservation, no fixed-noise helper calls,
identity one-patch projection, supported configuration routing and scope rejection,
and synthetic seam gradients. Initial full CPU regression submission:
`sbatch /tmp/diffpano_endpoint_tests.slurm`, job `19719857`, node `c267`.
The CPU launcher runs compileall then `python -m unittest discover -s tests -v`.
GPU runs remain gated on the complete suite.

First CPU run `19719857` passed all 149 discovered tests in 82.449 s, including
the nonzero consensus-correction oracle. The two seam tests were added after
discovery enumerated files, so a second complete regression is required before
GPU submission. No sampling implementation changes were needed to pass.

Final full CPU gate **passed: 151 tests in 37.205 s**, job `19719905`, node
`c586`. Logs `logs/endpoint-tests.19719905.{out,err}`. All new seam tests were
included. Only after this pass, four independent A100 jobs were submitted using
`sbatch --job-name=rgb-endpoint-BACKEND slurm/rgb_endpoint_a100.slurm configs/experiments/implied_endpoint_consensus/BACKEND.yaml`.
Each requests one A100, 8 CPUs, 64 GiB host RAM and a one-hour limit, with the
existing WebProxy/activate_venv environment and scratch HF caches. Each job runs
a fresh paired native/new comparison; the three latent jobs also run the two
single-patch VAE controls. No ERP or fusion matrix jobs are submitted.

GPU IDs: SD2 `19719936`, SANA `19719937`, FLUX `19719938`, PixelDiT `19719939`.
Logs: `logs/rgb-endpoint-BACKEND.JOBID.{out,err}`. Results pending inspection.

Pending limits were reduced with `scontrol update JobId=JOBID TimeLimit=...`:
SD2/SANA/FLUX 30 minutes, PixelDiT 20 minutes, based on earlier measured model
runtimes with headroom for the intentional VAE work. Initial scheduler estimates
were hours away; these are queue estimates, not generation timings.

A `sbatch --test-only --time=00:20:00` backfill probe found no earlier slot;
its reported hypothetical ID `19719972` is not a submitted job. Existing four
jobs remain the only GPU submissions.

### Grace scheduling adjustment: paired runs on A40

A100 estimates remained roughly two hours away. Grace `gpu-a40` exposed five
idle nodes, and an A40 `sbatch --test-only` probe (hypothetical ID `19719987`,
not a real job) predicted immediate availability. The **existing pending jobs**
were updated in place with
`scontrol update JobId=JOBID Partition=gpu-a40 Gres=gpu:a40:1`.
No duplicate jobs were submitted. Both native and consensus in every pair use
the same A40; model checkpoint, precision, guidance, geometry, initialization,
schedule and evaluation counts are unchanged. Comparison metadata now records
GPU model and total memory. Historical controls were A100; the fresh native
reference and no-roundtrip control will be compared against their saved outputs.
The equivalent direct submission is
`sbatch --partition=gpu-a40 --gres=gpu:a40:1 --job-name=rgb-endpoint-BACKEND --time=00:30:00 slurm/rgb_endpoint_a100.slurm configs/experiments/implied_endpoint_consensus/BACKEND.yaml`
(PixelDiT's current limit is 20 minutes). The script name retains its A100 default;
Slurm overrides select A40 without changing the environment or experimental code.

All four adjusted jobs started: SD2 `19719936` on `g101`; SANA `19719937`
and FLUX `19719938` on `g102` (separate A40 allocations); PixelDiT `19719939`
on `g103`. CPU tests and sampling implementation are unchanged after the gate;
subsequent edits only add GPU-name logging and final-output reporting diagnostics.

### Completed model runs and reporting checks

All model jobs exited successfully: SD2/SANA elapsed 6:26 each, FLUX 9:15,
PixelDiT 6:32. The new method is visually strongly degraded for SD2 (neon/line
artifacts), SANA (blocky grid/banding), and FLUX (large loss of contrast/detail).
All three no-roundtrip one-patch controls are good; all three RGB-roundtrip
controls degrade visibly with **zero RGB fusion correction**. PixelDiT consensus
remains coherent and detailed, with fewer visible tonal boundaries in this scene.
These are preliminary visual classifications pending common seam metrics below.

The fresh A40 native images are not byte-identical to the historical A100 images.
The primary comparison therefore uses the fresh native reference from the SAME
A40 allocation and exact same saved global initial field as consensus. The
historical images are contextual controls, not substituted numerical baselines.
This hardware difference and any historical-image MAE are retained in the audit.

Independent artifact audit job `19720045` was submitted with
`sbatch --nodelist=c586 /tmp/diffpano_rgb_endpoint_audit.slurm`. That previously
warm node was busy; `scontrol update JobId=19720045 ReqNodeList=` removed the
unnecessary constraint without resubmitting or changing the audit.

The first plot command failed because importing `diffpano.seams` also imported
`diffpano.__init__`, whose eager ERP imports require torch. Grace's lightweight
matplotlib module has no torch. The plotting script now loads the same pure
NumPy seam helper via `importlib.util` without package initialization. No metric
formula or sampling code changed, and no package installation was needed.

Artifact audit `19720045` failed on `c262` (exit 2, 33 s) because its Python
file was in the login node's `/tmp`, which is not shared with compute nodes.
Moved the audit into the checkout as `scripts/audit_rgb_endpoint_controls.py`
and changed the Slurm command to `python -m scripts.audit_rgb_endpoint_controls`.
This makes the audit reproducible and accessible on any allocated node. All GPU
outputs are intact; this failure concerns only post-run verification.

### Final paired results

All four fresh native references are **Good**. All latent no-roundtrip controls are **Good**. The quality labels below come from direct inspection of source images and comparison sheets, not from image MAE alone.

| Backend | Native MultiDiffusion | RGB implied-endpoint consensus | Seam change | Quality conclusion |
|---|---|---|---|---|
| SD2 | Good | Strong degradation: neon/line artifacts | Worse | VAE roundtrip already fails without fusion |
| SANA | Good, some tonal boundaries | Strong degradation: blocks/grid/banding | Worse | VAE roundtrip already fails without fusion |
| FLUX | Good, tonal boundaries | Strong degradation: washed-out/blurred detail | Worse relative to nearby detail | VAE roundtrip already loses contrast/detail |
| PixelDiT | Good, visible tonal boundaries | Good, coherent detailed scene | Improved seams | Supports pixel-native clean RGB consensus for this control |

#### One-patch projection controls

Each latent control uses the same full schedule and CPU-seeded initial epsilon as its no-roundtrip partner. RGB fusion correction is **exactly 0 at every step** in these controls. All have N/N guided predictions. Final RGB MAE is measured in raw, unclamped model RGB units and is a difference measure, not a quality score. Self-roundtrip MAE is in each backend's own native latent units.

| Backend | No roundtrip | RGB roundtrip | Guided predictions each | Final RGB MAE / RMSE | Mean native self-roundtrip MAE |
|---|---|---|---:|---:|---:|
| SD2 | Good | Strong degradation | 30 | 0.378362 / 0.513083 | 0.078895 |
| SANA | Good | Strong degradation | 20 | 0.372194 / 0.508684 | 0.308755 |
| FLUX | Good | Strong degradation | 20 | 0.336138 / 0.368682 | 0.158148 |

PixelDiT has no VAE and requires no separate projection control.

#### Per-step consensus diagnostics

Entries are arithmetic means over all scheduled steps. Overlap mean/max are neighboring-pair diagnostics on **pre-fusion predicted-clean RGB**; the max column averages each step's maximum pair MAE (not a global maximum over time). Correction is the mean absolute RGB change per crop. These raw RGB quantities may exceed displayed [-1,1] bounds early in sampling. The last column is the original clean native prediction versus its own decode/encode roundtrip, with backend-specific latent units. No native-baseline overlap metric is equated to these RGB quantities.

| Backend | Pre-fusion overlap mean / max-pair MAE | Consensus correction MAE | Native self-roundtrip MAE |
|---|---:|---:|---:|
| SD2 | 0.203631 / 0.223629 | 0.067877 | 0.085216 |
| SANA | 0.150320 / 0.167101 | 0.050107 | 0.235534 |
| FLUX | 0.077791 / 0.082696 | 0.025930 | 0.161801 |
| PixelDiT | 0.099727 / 0.107269 | 0.033242 | N/A |

Disagreement becomes small late in several runs even when images are severely degraded; agreement alone does not establish quality. For example, FLUX mean overlap falls from 0.285734 to 0.003707 while its output loses contrast and detail.

#### Common final-RGB boundary comparison

All rows use the saved full-resolution PNGs, RGB [0,1], and the same exact patch boundaries within each backend. Internal vertical boundary lines are 256/512/768 for SD2 and 512/1024/1536 for the other backends; there are no internal horizontal boundaries. Excess = boundary gradient minus nearby gradient. Ratios divide by max(nearby gradient, 1e-12). They are scene-dependent diagnostics, not universal perceptual seam scores.

| Backend | Boundary gradient native → RGB | Nearby gradient native → RGB | Excess native → RGB | Ratio native → RGB |
|---|---:|---:|---:|---:|
| SD2 | 0.028935 → 0.156477 | 0.030341 → 0.072135 | -0.001406 → 0.084342 | 0.953653 → 2.169229 |
| SANA | 0.041931 → 0.168073 | 0.040158 → 0.069357 | 0.001773 → 0.098716 | 1.044156 → 2.423306 |
| FLUX | 0.040312 → 0.031208 | 0.039980 → 0.003898 | 0.000333 → 0.027309 | 1.008317 → 8.005294 |
| PixelDiT | 0.059035 → 0.049862 | 0.035891 → 0.040118 | 0.023144 → 0.009745 | 1.644848 → 1.242901 |

PixelDiT's boundary excess decreases **57.90%**, consistent with its visibly improved tonal continuity. SD2/SANA excess increases strongly. FLUX's absolute boundary gradient decreases, but nearby gradients decrease much more because of blur/detail loss; its relative seam excess increases. It would be misleading to call that absolute-gradient reduction a seam improvement.

We also measured the saved final fused-clean image before terminal rendering. Terminal-versus-fused displayed RGB MAE is SD2 0.012214, SANA 0.012335, FLUX 0.007431, PixelDiT exactly 0. The fused-clean boundary/nearby ratios are 2.1424, 2.5072, 7.9165 and 1.2429 respectively: the latent failures and boundary artifacts already exist before terminal output decoding. Detailed alternative metrics are saved in summary.json.

#### Runtime and memory

Timings below isolate each method after backend/conditioning preparation. Native executes first and consensus second in the same process, so warm kernels/caches and VAE diagnostics make these **descriptive timings, not a speed benchmark**. Allocated memory is the peak during that stage. Reserved memory can carry over from earlier stages; these are not whole-process/cold-setup peak measurements. They should not be compared directly with the earlier setup-inclusive A100 peaks. Both methods within each pair ran on NVIDIA A40.

| Backend | Native seconds | RGB seconds | Native allocated / reserved GiB | RGB allocated / reserved GiB | Guided predictions native / RGB |
|---|---:|---:|---:|---:|---:|
| SD2 | 24.413 | 13.920 | 2.399 / 2.873 | 2.415 / 2.873 | 90 / 90 |
| SANA | 12.274 | 39.182 | 6.971 / 10.428 | 5.540 / 10.430 | 60 / 60 |
| FLUX | 61.740 | 89.310 | 24.821 / 27.256 | 24.883 / 27.262 | 60 / 60 |
| PixelDiT | 36.961 | 33.861 | 4.085 / 4.490 | 4.077 / 4.490 | 150 / 150 |

| Backend | Setup seconds | Total measured seconds including controls | Slurm job / node | Slurm elapsed |
|---|---:|---:|---|---|
| SD2 | 184.871 | 232.848 | 19719936 / g101 | 6:26 |
| SANA | 189.912 | 260.534 | 19719937 / g102 | 6:26 |
| FLUX | 219.425 | 425.109 | 19719938 / g102 | 9:15 |
| PixelDiT | 204.421 | 278.447 | 19719939 / g103 | 6:32 |

Slurm elapsed includes interpreter/module/shared-filesystem startup excluded from the measured total. The method loop also includes the optional own-clean VAE re-encode diagnostic for latent models. All jobs exited 0; no GPU numerical or out-of-memory failures occurred.

| Latent control | No-roundtrip seconds | Roundtrip seconds | No-roundtrip allocated / reserved GiB | Roundtrip allocated / reserved GiB |
|---|---:|---:|---:|---:|
| SD2 | 1.100 | 4.485 | 2.396 / 2.873 | 2.408 / 2.873 |
| SANA | 2.344 | 12.777 | 5.340 / 10.430 | 5.512 / 10.430 |
| FLUX | 20.371 | 30.067 | 24.802 / 27.262 | 24.850 / 27.262 |

#### Output artifacts

- **SD2**: `outputs/rgb-endpoint-controls/sd2-implied_endpoint_consensus-global_native_canvas-average-uniform/20260909-153133-19719936/`. [Paired image](../outputs/rgb-endpoint-controls/report/sd2-comparison.png); [diagnostic curves](../outputs/rgb-endpoint-controls/report/sd2-diagnostics.png).
  [One-patch roundtrip comparison](../outputs/rgb-endpoint-controls/report/sd2-roundtrip.png).
- **SANA**: `outputs/rgb-endpoint-controls/sana-implied_endpoint_consensus-global_native_canvas-average-uniform/20260909-153144-19719937/`. [Paired image](../outputs/rgb-endpoint-controls/report/sana-comparison.png); [diagnostic curves](../outputs/rgb-endpoint-controls/report/sana-diagnostics.png).
  [One-patch roundtrip comparison](../outputs/rgb-endpoint-controls/report/sana-roundtrip.png).
- **FLUX**: `outputs/rgb-endpoint-controls/flux-implied_endpoint_consensus-global_native_canvas-average-uniform/20260909-153144-19719938/`. [Paired image](../outputs/rgb-endpoint-controls/report/flux-comparison.png); [diagnostic curves](../outputs/rgb-endpoint-controls/report/flux-diagnostics.png).
  [One-patch roundtrip comparison](../outputs/rgb-endpoint-controls/report/flux-roundtrip.png).
- **PixelDiT**: `outputs/rgb-endpoint-controls/pixeldit-implied_endpoint_consensus-global_native_canvas-average-uniform/20260909-153133-19719939/`. [Paired image](../outputs/rgb-endpoint-controls/report/pixeldit-comparison.png); [diagnostic curves](../outputs/rgb-endpoint-controls/report/pixeldit-diagnostics.png).

Each directory contains the saved initialization, resolved config, comparison.json, and `native_multidiffusion/result.png` / `implied_endpoint_consensus/result.png`, with per-method metadata and per-step CSV. Latent one-patch directories retain both controls, their epsilon and detailed control.json. The report directory contains **11 PNG + 11 PDF figures**, [summary.csv](../outputs/rgb-endpoint-controls/report/summary.csv) and [summary.json](../outputs/rgb-endpoint-controls/report/summary.json).

Plot command: `module purge; module load GCC/13.2.0 matplotlib/3.8.2`, then `python -m scripts.plot_rgb_endpoint_controls --output outputs/rgb-endpoint-controls/report outputs/rgb-endpoint-controls/*/*/comparison.json`. All figures read saved measurements and images; no seaborn or custom color styling is used.

#### Scientific decision

**The architecture preserves native-quality generation for PixelDiT in this control, but does not preserve quality across all four backends.** PixelDiT supports the combination of local native diffusion, global clean RGB synchronization, and local endpoint preservation, with improved seam diagnostics. The latent results support Case C and the VAE-related part of Case D: repeated deterministic decode→encode projection already disrupts the independent one-patch trajectory, before any cross-patch fusion correction. Consequently multi-patch fusion alone cannot explain those failures.

The previously successful no-roundtrip endpoint algebra remains intact and its new controls remain good. These results isolate the intentional RGB/native projection as a major disturbance under the tested full-strength schedule; they do not identify which aspect of VAE projection or early predicted-clean states causes the damage. Model-specific latent MAE values cannot be compared as universal error scales. No diffusion-versus-flow split is supported: SD2 also degrades.

**Do not advance all backends to ERP on this evidence.** PixelDiT alone supports a separately scoped ERP follow-up; latent projection requires further investigation first. No ERP, spherical projection, LPW/DPA/weighted fusion, dynamic layout, time travel, velocity-preserving reinjection, partial consensus, or extra denoiser correction pass was implemented or run. This remains one prompt/seed and the validated first-order schedules, not a general benchmark.

#### Files added or extended in this task

Added `diffpano/implied_endpoint_consensus.py`, `diffpano/seams.py`, `scripts/paired_rgb_endpoint.py`, `scripts/plot_rgb_endpoint_controls.py`, `scripts/audit_rgb_endpoint_controls.py`, `slurm/rgb_endpoint_a100.slurm`, four paired YAMLs, `tests/test_implied_endpoint_consensus.py`, and `tests/test_seams.py`. Extended configuration validation, the generation router and metadata, and added the optional replacement-clean argument to the existing endpoint helper. Both native-control documents were appended. Prior uncommitted trajectory/backend work and all historical results were preserved.

#### Final independent verification

Replacement artifact audit **passed**, job `19720074`, node `c262`, elapsed
2:35, exit 0. Evidence: [artifact-audit.json](../outputs/rgb-endpoint-controls/report/artifact-audit.json).
For all four pairs it independently reloads the saved global initialization and
checks its hash/shape, exact native-to-RGB geometry, complete coverage, schedules,
model metadata and model counts. For each latent control it also checks saved
epsilon against bit-exact CPU regeneration from seed 0 and N/N evaluations.

Historical A100 versus fresh A40 displayed-native RGB MAEs are SD2 0.001057,
SANA 0.007766, FLUX 0.126265, PixelDiT 0.199930. Historical versus fresh
no-roundtrip single-patch MAEs are SD2 0.002723, SANA 0.011714 and FLUX 0.009303.
None of those historical PNG comparisons is byte-identical. The historical
native global Gaussian was not saved, so this audit cannot isolate changes in
the large GPU-generated initial field from arithmetic/hardware effects. It does
**not** claim historical A100 initial tensors were reproduced on A40. Within each
new pair the same saved global field supplies native sampling and bit-identical
local crops, and all requested model/settings controls match. The numerical seam
comparison uses those fresh paired images exclusively.

The complete CPU regression gate passed **151 tests** before GPU submission.
Subsequent changes were reporting/audit-only; both report generation and the
independent artifact audit completed successfully. All 22 figure artifacts and
four source result sets are present. `git diff --check` passes. No Slurm jobs
remain queued or running. HEAD remains
`1a37478ca817e7c20fa46ab9c721912e760e4cd1` on `no_sphere`; all changes remain
local/uncommitted and prior work/history/output artifacts are preserved.

## LookingGlass training-free VAE residual correction (September 10)

Definitions for this section: **A** single-patch native; **B** single-patch
implied endpoint without RGB roundtrip; **C** single-patch implied endpoint with
RGB roundtrip; **D** C plus deterministic VAE residual correction; **E** native
MultiDiffusion; **F** planar RGB implied-endpoint consensus; **G** F plus
synchronized VAE residual correction. Existing A/B/E/F history is retained.

This task uses **no training, learned bridge or correction network**. The preceding
interrupted request had explicitly requested training, and partial unexecuted
learned-bridge files were written before the user replaced that request. They
remain preserved as inactive work, per the instruction not to discard uncommitted
files. No dataset collection, training job or checkpoint was produced. The new
`vae_residual` entrypoint does not import or use those learned-bridge modules.

Starting branch/HEAD: `no_sphere` /
`1a37478ca817e7c20fa46ab9c721912e760e4cd1`; initial queue empty. No prior files
or outputs were reset, cleaned, checked out or discarded.

Read [LookingGlass Section 4.1 and Algorithm 1](https://arxiv.org/html/2504.08902v1#S4.SS1)
before implementation. The 49 MB PDF exceeded the web reader limit, so the arXiv
HTML version was used. In our clean-endpoint notation, Eq. (7) gives
`z_fused_rt_i = E(crop_i(average_j(place_j(D(z0_j)))))`.
Eq. (8) gives `r_i = z0_i - E(D(z0_i))`, then
`r_fused_i = crop_i(average_j(place_j(r_j)))`. Add
`z_corrected_i = z_fused_rt_i + r_fused_i` before the existing implied-endpoint
transition. For partial planar patches, average divides by local coverage.
RGB placement/cropping uses RGB coordinates; residual placement/cropping uses
exact native latent coordinates, with the loaded integer spatial factor.
This deliberately replaces the paper's view transformations and pyramid blending
with exact planar crops and uniform averaging. No LPW or ERP is added.

The VAE reconstruction residual above is **distinct from the model-implied
noise/residual endpoint**. Only the former is synchronized. The original local
predicted epsilon or implied flow endpoint remains unchanged and step-local.
Single identity-patch D computes `E(D(z0)) + [z0-E(D(z0))]` explicitly; it does
not shortcut to z0. A floating-point cancellation bound is checked at every step.

Phase 1 adds `diffpano/vae_residual.py`, `scripts/vae_residual_experiment.py`,
`configs/experiments/vae_residual/{sd2,sana,flux}.json`, and
`slurm/vae_residual.slurm`. C/D perform one clean decode and one encode per step,
without any fusion accumulator. Real VAE preflight checks two random latent
amplitudes before model trajectories. B is rerun only as the matched numerical
oracle for independent D/B trajectory drift. The saved A40 control epsilon,
configuration, conditioning hash, exact timesteps/sigmas and GPU model are checked.
Counts must be N/N/N; preflight and correction introduce zero denoiser calls.

CPU tests check arbitrary-channel identity recovery, device/dtype compatibility,
C decode/encode counts and absence of fusion, D/B equivalence, endpoint bitwise
preservation and rejection of duplicate denoiser calls. Full CPU submission:
`sbatch /tmp/diffpano_endpoint_tests.slurm`, job `19721943`.
GPU image experiments remain gated on the complete regression pass.

Phase 1 full regression **passed: 154 tests in 74.701 s**, job `19721943`,
node `c264`. Logs `logs/endpoint-tests.19721943.{out,err}`.
After that gate, independent C/D jobs were submitted with
`sbatch --job-name=residual-cd-BACKEND --time=00:20:00 slurm/vae_residual.slurm CD configs/experiments/vae_residual/BACKEND.json`.
The launcher requests one A40 on `gpu-a40`, 8 CPUs, 64 GiB host RAM, and uses
the unchanged model environment. The GPU type deliberately matches existing
F/B/C controls. Each job first runs the real-VAE identity oracle, then B/C/D.

Phase 1 GPU IDs: SD2 `19721949`, SANA `19721951`, FLUX `19721952`.
Logs use `logs/residual-cd-BACKEND.JOBID.{out,err}`. New artifacts are isolated
under `outputs/vae-residual-controls/20260910-lookingglass-v1/BACKEND/CD/`.

Phase 1 completed successfully on A40s (`g103` SD2, `g105` SANA/FLUX).
The rerun B PNGs are byte-identical to the saved B images for all three models.
C reproduces the saved SD2/FLUX PNGs exactly; SANA C has a small numerical
trajectory difference but retains the same severe banding/grid degradation.
C has no fusion operation, establishing that the VAE roundtrip alone is
sufficient to produce these isolated failures.

| Backend | B implied | C RGB roundtrip | D training-free correction |
|---|---|---|---|
| SD2 | Detailed natural ruins/valley | Severe neon posterization and edge patterns | Natural colors and detail restored, visually close to B |
| SANA | Detailed ruins/mountains | Severe horizontal/vertical grid and banding | Structure, color and detail restored, visually close to B |
| FLUX | Detailed ruins/path/mountains | Structure survives, but gray and low contrast | Color/contrast and detail restored, visually close to B |

The explicit algebraic D recovery MAE, averaged over steps, is
`2.1834e-10 / 3.4251e-9 / 3.1890e-10` for SD2/SANA/FLUX. Maximum absolute
error across all steps/elements is `2.3842e-7 / 9.5367e-7 / 5.9605e-8`.
The real-VAE random-latent preflights also pass. Thus the required identity
property holds near float32 roundoff. **D is not a bit-identical full-trajectory
reproduction of B**: final raw decoded RGB MAE is
`0.005648 / 0.012452 / 0.008290`, versus C/B `0.378362 / 0.372116 / 0.336138`.
Small per-step cancellation errors can propagate through the fp16/bf16 networks;
these observations are consistent with that sensitivity, but do not separately
measure deterministic-kernel or other low-precision contributions. No shortcut
to the original clean latent was used to force equality. The deterministic mock
D/B oracle passes at `atol=rtol=1e-6`.

Phase 2 adds the optional training-free flag to the existing F pipeline. The
own-proposal VAE encode already used for F's diagnostic now supplies each local
residual, so G still uses two VAE encodes per patch/step and exactly one guided
model prediction. A canonical-order, arbitrary-channel native accumulator
synchronizes residuals, and the existing RGB accumulator and terminal-output
rule remain unchanged. The residual canvas is discarded each timestep; only
local noisy states persist. Focused oracles cover zero residual G=F, one-patch
G=D, C=7 exact placement with horizontal/vertical overlaps, identical residual
crops, reverse patch processing, endpoint bitwise preservation, call counts,
and an independent multi-step oracle that distinguishes synchronized correction
from the incorrect local-residual-only method. Full regression job: `19721963`.

C/D residual magnitudes (raw native units; not directly comparable across
different VAEs). Each first/last/peak is a per-step spatial MAE; mean is over
steps. Normalization is residual MAE divided by clean-latent population std,
with denominator floor `1e-8`. C and D follow different trajectories after the
first update, so their later residual curves need not coincide.

| Backend / control | First MAE | Last MAE | Peak MAE | Mean MAE | Mean RMSE | Mean residual std | Mean MAE / clean std |
|---|---:|---:|---:|---:|---:|---:|---:|
| SD2 C | 0.175692 | 0.057806 | 0.254202 | 0.078895 | 0.121466 | 0.120518 | 0.094488 |
| SD2 D | 0.175692 | 0.081856 | 0.180939 | 0.092288 | 0.154277 | 0.152678 | 0.133818 |
| SANA C | 2.438163 | 0.085561 | 2.438163 | 0.308939 | 0.437809 | 0.437488 | 0.174443 |
| SANA D | 2.438163 | 0.086375 | 2.438163 | 0.367309 | 0.536925 | 0.534505 | 0.234752 |
| FLUX C | 0.382559 | 0.066315 | 0.382559 | 0.158148 | 0.213441 | 0.213074 | 0.167962 |
| FLUX D | 0.382559 | 0.065388 | 0.382559 | 0.181471 | 0.241579 | 0.241247 | 0.202104 |

C/D per-step JSON/CSV also contain actual scheduler timesteps and coefficients,
latent means/std before and after the roundtrip, and D recovery MAE/RMSE/max.
Single-patch figures: `outputs/vae-residual-controls/report/{sd2,sana,flux}-BCD.{png,pdf}`
and `*-residuals.{png,pdf}`. Machine-readable aggregation: `single-summary.{csv,json}`.

Sampling runtimes and peak CUDA memory (allocated / reserved GiB). Timed sampling
includes final decoding and residual diagnostics, but excludes setup, preflight
and writing images. All three phases share a loaded process, so reserved-memory
peaks include the preceding allocator pool.

| Backend | B seconds | C seconds | D seconds | C peak GiB | D peak GiB | Whole Python run seconds |
|---|---:|---:|---:|---:|---:|---:|
| SD2 | 2.384 | 3.595 | 3.608 | 2.400 / 2.879 | 2.403 / 2.879 | 132.122 |
| SANA | 2.696 | 9.946 | 9.923 | 5.492 / 7.057 | 5.504 / 7.057 | 158.010 |
| FLUX | 20.871 | 27.086 | 27.236 | 24.817 / 26.635 | 24.830 / 26.637 | 240.081 |

Actual guided prediction counts per B/C/D: SD2 30/30/30; SANA 20/20/20;
FLUX 20/20/20. Zero extra denoiser calls for residual diagnostics or preflight.
Slurm elapsed times (including module/Python startup): SD2 3:20, SANA 4:08,
FLUX 5:29; all completed with exit code 0.

Phase 2 full regression **passed: 158 tests in 88.981 s**, job `19721963`,
node `c264`; compileall passed. Logs `logs/endpoint-tests.19721963.{out,err}`.
Only after this pass, G jobs were submitted with the phase-1 launcher/resources
and `G` instead of `CD`: SD2 `19721970`, SANA `19721971`, FLUX `19721972`.
Their logs are `logs/residual-g-BACKEND.JOBID.{out,err}`. G writes to
`outputs/vae-residual-controls/20260910-lookingglass-v1/BACKEND/G/`.
The saved global native initial tensor is loaded from F, its tensor SHA-256
is checked, and exact native crops initialize the local states. No new initial
noise draw is used. Each job also repeats the actual VAE identity preflight
before its first model prediction. E/F outputs are reused without modification.

All three G jobs completed with exit code 0 on separate A40s on `g105`.
Slurm elapsed times: SD2 4:17, SANA 4:35, FLUX 5:54. All G preflights passed.
The final artifact audit (`scripts/audit_vae_residual_controls.py`) passed for
all backends: configuration/conditioning/initialization hashes, exact scheduler
timesteps and coefficients, finite per-step metrics, identity recovery bounds,
full coverage, expected image dimensions and unchanged model-evaluation counts.
Audit output: `outputs/vae-residual-controls/report/audit.json`.

| Backend | E Native MultiDiffusion | F uncorrected RGB consensus | G synchronized residual correction |
|---|---|---|---|
| SD2 | Natural detailed coherent ruins/valley | Severe neon posterization and hard patch transitions | Less neon; some natural color returns, but severe central blur/edge patterns and hard boundaries remain; far below E |
| SANA | Detailed ruins and mountains | Severe grid/banding and blocks | Detailed outer quarters, but central overlap regions remain abstract, banded and blocky; boundaries worsen; far below E |
| FLUX | Detailed, colorful ruins panorama | Recognizable structure but gray, washed out and softer | Strong restoration of color/contrast and much detail; approaches E locally, but remains softer and visibly segmented at vertical boundaries |

The SANA outer quarters have coverage one, while the central half has coverage
two. SD2 also shows strong spatially uneven recovery. This pattern is consistent
with a remaining cross-patch incompatibility; it is not evidence that the
identity residual formula failed. No residual was clamped, trained or replaced
by a learned estimate to improve the images.

| Backend | E / F / G sampling seconds | E peak allocated/reserved GiB | F peak allocated/reserved GiB | G peak allocated/reserved GiB | G whole Python seconds | G model calls |
|---|---:|---:|---:|---:|---:|---:|
| SD2 | 24.413 / 13.920 / 14.142 | 2.399 / 2.873 | 2.415 / 2.873 | 2.417 / 2.912 | 145.433 | 90 |
| SANA | 12.274 / 39.182 / 38.861 | 6.971 / 10.428 | 5.540 / 10.430 | 5.552 / 7.561 | 171.411 | 60 |
| FLUX | 61.740 / 89.310 / 89.366 | 24.821 / 27.256 | 24.883 / 27.262 | 24.882 / 26.643 | 250.469 | 60 |

E/F timings are their historical matched A40 pair, G is the new A40 job.
Reserved CUDA memory is allocator-history dependent: E/F shared a process; G
has a fresh process and identity preflight. These are measured sampling costs,
not a controlled speed benchmark. G uses the same 90/60/60 guided evaluations
as F, with zero extra diagnostic denoiser calls.

G diagnostic averages over timesteps (local maximum means the maximum
per-patch spatial MAE at each timestep, then averaged over time):

| Backend | Pre-fusion overlap mean / max | Local residual mean / max | Fused residual MAE / std | Cropped residual correction MAE | RGB consensus correction MAE |
|---|---:|---:|---:|---:|---:|
| SD2 | 0.151473 / 0.174679 | 0.116738 / 0.136528 | 0.104907 / 0.196670 | 0.107771 | 0.050491 |
| SANA | 0.213027 / 0.256688 | 0.208585 / 0.258795 | 0.197880 / 0.276981 | 0.192911 | 0.071009 |
| FLUX | 0.081327 / 0.088120 | 0.172106 / 0.177623 | 0.161083 / 0.216041 | 0.155544 | 0.027109 |

The corresponding F mean RGB overlap disagreements were SD2 `0.203631`,
SANA `0.150320`, FLUX `0.077791`. G decreases this diagnostic for SD2 but
increases it for SANA and slightly for FLUX. A single overlap scalar does not
measure recovered image quality.

Common PNG seam metrics use the same existing internal patch boundaries and
nearby-gradient reference as the previous report, on RGB [0,1].

| Backend | E / F / G boundary excess | E / F / G boundary-to-nearby ratio |
|---|---:|---:|
| SD2 | -0.001406 / 0.084342 / 0.042584 | 0.954 / 2.169 / 2.451 |
| SANA | 0.001773 / 0.098716 / 0.213694 | 1.044 / 2.423 / 4.953 |
| FLUX | 0.000333 / 0.027309 / 0.059204 | 1.008 / 8.005 / 4.819 |

SD2 reduces absolute seam excess while still producing a poor image. SANA
increases both seam excess and ratio. FLUX reduces the ratio from 8.005 to 4.819
as local detail returns, **but absolute seam excess increases from 0.027309 to
0.059204**. Therefore the FLUX color restoration must not be described as seam
removal. Native E remains close to a boundary-to-nearby ratio of one.

The completed evidence supports three conclusions:

1. In this matched prompt/seed/config, the VAE roundtrip alone is sufficient
   to damage all three isolated trajectories. Explicit training-free identity
   correction restores the clean latent near roundoff and the final appearance
   near B, with the measured small accumulated D/B drift stated above.
2. For FLUX, reconstruction mismatch is a major contributor to washed-out
   RGB-consensus output, and residual correction addresses much of that problem.
   It does not establish a complete repair of multi-patch generation.
3. For SD2 and SANA, this correction does not repair the actual multi-patch
   failure. Simple VAE reconstruction mismatch is therefore insufficient as a
   sole explanation or remedy; cross-patch clean-prediction/residual disagreement
   remains implicated. Encoding a fused RGB crop generally differs from fusing
   encodings of the original crops, so the single-patch algebraic identity has
   no corresponding guarantee for overlaps.

These runs do not establish a universal ranking of the VAEs or identify a unique
architectural cause for backend sensitivity. Raw native residual magnitudes are
not comparable across latent scalings/channel conventions, and even normalized
means do not order visual failure severity: C relative means are SD2 0.0945,
SANA 0.1744, FLUX 0.1680, despite SD2 looking much worse than FLUX. The single
prompt/seed is a controlled diagnosis, not a multi-prompt quality benchmark.

Final artifacts (all prior outputs remain unchanged):

- `outputs/vae-residual-controls/20260910-lookingglass-v1/BACKEND/CD/`: B/C/D
  final PNGs, C/D per-step CSV/JSON, exact saved epsilon, preflight, comparison
  and runtime JSON.
- `outputs/vae-residual-controls/20260910-lookingglass-v1/BACKEND/G/`: exact
  loaded native initialization, preflight, comparison/runtime JSON;
  `generation/{result.png,final_fused_clean.png,metadata.json,steps.csv}`.
- `outputs/vae-residual-controls/report/`: 12 PNG and 12 PDF figures: BCD,
  EFG, C/D residual curves, and G/F diagnostics for each backend;
  `summary.{csv,json}`, `single-summary.{csv,json}`, and `audit.json`.
  All comparison images were visually inspected.

Quick visual comparisons: [SD2 BCD](../outputs/vae-residual-controls/report/sd2-BCD.png),
[SANA BCD](../outputs/vae-residual-controls/report/sana-BCD.png),
[FLUX BCD](../outputs/vae-residual-controls/report/flux-BCD.png);
[SD2 EFG](../outputs/vae-residual-controls/report/sd2-EFG.png),
[SANA EFG](../outputs/vae-residual-controls/report/sana-EFG.png),
[FLUX EFG](../outputs/vae-residual-controls/report/flux-EFG.png).

## Stable Diffusion 3.5 A–G Validation

Task started on `no_sphere`, HEAD `1a37478ca817e7c20fa46ab9c721912e760e4cd1`,
with an empty Slurm queue. All existing uncommitted implementations, configurations,
outputs and reports are preserved. Naming: `model.pipeline: sd35`.

No SD3.5 checkpoint/config was present in the inspected project and configured
Hugging Face caches. Existing credentials can access the official
[Stable Diffusion 3.5 Medium checkpoint](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium).
CPU download job `19722944` completed successfully, pinning revision
`b940f670f0eda2d07fbb75229e779da1ad11eb80`. Cache manifest:
`outputs/native-controls/sd35-checkpoint.json`. Only Diffusers components were
downloaded, omitting duplicate standalone checkpoint formats. No credentials
are stored in result metadata. Initial submission without a nodes/tasks field
was rejected by Grace before allocation; the corrected submission above succeeded.

Inspected the installed `pipeline_stable_diffusion_3.py` (especially native
initialization, schedule preparation, transformer/CFG calls, scheduler step and
VAE decode) and `scheduling_flow_match_euler_discrete.py`. The loaded checkpoint
configuration specifies:

- `StableDiffusion3Pipeline`, `SD3Transformer2DModel` (MMDiT-X), 24 layers,
  16 input/output channels and transformer patch size 2. Backend state is raw
  BCHW latents; no FLUX-style external token packing.
- `FlowMatchEulerDiscreteScheduler`, 1,000 training timesteps, static shift 3.0.
  The checkpoint does not enable dynamic shifting. The actual inference sigmas
  are read after `set_timesteps`; no separate sigma reshifting is introduced.
- Ordinary negative/positive classifier-free guidance, no guidance embedding,
  no skip-layer guidance. Three text encoders: two CLIPs and T5-XXL.
- `AutoencoderKL`, 16 latent channels, spatial factor 8, scaling factor 1.5305,
  shift factor 0.0609, no quant/post-quant convolutions. Deterministic roundtrip
  encoding uses the existing posterior-mode helper and unchanged scale/shift.

The installed scheduler integrates `x_next = x + (sigma_next-sigma)*prediction`.
The SD3 pipeline supplies its guided transformer output directly to that step.
Thus its output is flow velocity in this integration convention, and the
candidate decomposition is `z0=x-sigma*v`, `z1=x+(1-sigma)*v`, followed by
`(1-sigma_next)*z0+sigma_next*z1`. Equivalence must be verified numerically
before interpreting B–G. The installed Euler scheduler also casts its result
back to the prediction dtype; this precision behavior must be distinguished
from endpoint algebra, as in the preceding FLUX analysis.

An unmodified official-pipeline sanity image is scheduled as job `19722948`,
only after successful checkpoint download, using 1024×1024, 40 steps, CFG 4.5,
bfloat16, sequence length 256, seed 0 and the existing native-control prompt.
The model card's basic example uses 40 steps/CFG 4.5. No A–G job has been
submitted yet; ordinary-image verification and full regression tests remain
prerequisites. Sanity artifacts go beside existing native controls at
`outputs/native-controls/sd35-official-sanity/JOBID/`.

| Experiment | Patches | Native/RGB sync | RGB round-trip | VAE residual correction |
|------------|---------|-----------------|----------------|-------------------------|
| A | 1 | none | No | No |
| B | 1 | none | No | No |
| C | 1 | none | Yes | No |
| D | 1 | none | Yes | Yes |
| E | multiple | native-state averaging | No | No |
| F | multiple | clean RGB averaging | Yes | No |
| G | multiple | clean RGB averaging | Yes | Yes |

The correction remains exactly the preceding training-free LookingGlass method.

Official sanity attempt `19722948` failed before denoising in CLIP's text
projection (`Half != BFloat16`). The initial loader requested bfloat16 but
`.to('cuda')` alone did not make the loaded CLIP execution homogeneous. The
retry explicitly calls the official pipeline's `.to('cuda', dtype=bfloat16)`
and saves parameter dtype histograms before/after this cast. This is the same
explicit dtype configuration used by the project's existing `_configure_denoiser`.
The failed output directory/logs are preserved, and A–G remain gated.

Official sanity retry **passed**, job `19722965`, A40 `g103`. The image shows
natural, detailed ruins and coherent mountains. Sampling (including prompt
encoding/final decode) took 38.267 s; whole Python work 116.576 s; peak CUDA
allocated/reserved 17.621/19.912 GiB. Installed Diffusers `0.32.2`, Torch
`2.7.0+cu126`. Artifact: `outputs/native-controls/sd35-official-sanity/19722965/`.
`loaded_dtypes.json` confirms the first CLIP had 196 fp16 parameters plus one
bf16 projection, and the second had 516 fp16 parameters plus one bf16 projection.
The explicit pipeline cast makes all floating model parameters bf16.

The SD3.5 adapter now uses the project's SD2/SANA arithmetic convention:
network and VAE bf16; network predictions are promoted to float32 before CFG;
native states, guided velocities and both ordinary scheduler/endpoint transitions
are float32. The scheduler itself is unchanged. This is an explicit precision
choice shared by **all A–G controls**, distinct from the sanity pipeline's native
bf16 CFG/state rounding. A is the ordinary native scheduler through our adapter,
not a claim of bit-identical reproduction of the official sanity image (whose
initial dtype/noise draw also differs). This prevents conflating Euler's
prediction-dtype cast with endpoint algebra. No residual gain or normalization
is introduced.

Implementation: `diffpano/pipelines/sd35.py`, lazy registry/config support,
optional backend-details metadata, and `scripts/controlled_ladder.py`. The new
runner invokes the existing single-patch residual, native MultiDiffusion and RGB
consensus implementations. It runs one scientific stage at a time and rejects
missing preceding results, different configuration/conditioning/sigmas/checkpoint,
different GPU model, and overwritten completed controls. A's cached native states
support an independent B trajectory comparison without rerunning A. E's exact
global initialization is saved once; F/G load it and use bit-identical crops.

E optionally decodes each cached predicted-clean latent only for pre-fusion RGB
overlap diagnostics. Its native state update and fusion are unchanged, with no
extra denoiser calls. This optional flag defaults off for all existing runs;
a focused test compares E with/without it for exact equality. E therefore has
additional diagnostic VAE cost that must be included in timing interpretation.

Seven configs, all validated through the existing configuration system:

| Label | Config |
|---|---|
| A | `configs/experiments/trajectory/sd35.yaml` |
| B | `configs/experiments/trajectory/sd35-implied.yaml` |
| C | `configs/experiments/vae_residual/sd35-c.yaml` |
| D | `configs/experiments/vae_residual/sd35-d.yaml` |
| E | `configs/experiments/native_multidiffusion/sd35.yaml` |
| F | `configs/experiments/implied_endpoint_consensus/sd35.yaml` |
| G | `configs/experiments/vae_residual/sd35-g.yaml` |

`configs/experiments/trajectory/sd35-ladder.json` maps these configs and output
paths into the existing native-controls, endpoint-controls, rgb-endpoint-controls
and vae-residual-controls categories. Single patch: native 128×128 / RGB 1024².
E/F/G: native 128×256 / RGB 1024×2048, three 128²-native / 1024²-RGB patches,
stride 64-native / 512-RGB, complete coverage and 50% overlap. VAE tiling stays
disabled for every SD3.5 run. All controls use the pinned checkpoint, seed 0,
40 steps, CFG 4.5, empty negative prompt and sequence length 256.

Full regression submission with SD3.5 scheduler/VAE/adapter/geometry/CFG/fairness/
residual-order/count oracles: `19722995`. A–G remain gated on this pass.

Regression job `19722995` ran 166 tests and caught one error in the newly added
E RGB diagnostic: stride was read from `PlanarPatchLayout`, which does not expose
it. Fixed to read the already validated native config's stride. The exact-state
preservation test caught this before any A–G GPU experiment. Full-suite retry:
`19723013`. Also canonicalized Diffusers' unordered `_use_default_values` list
in backend metadata so cross-process comparisons check settings rather than
incidental set ordering.

Full regression **passed: 166 tests in 10.812 s**, job `19723013`; compileall
passed. Only after this gate, stage A was submitted. The runner checks the
native scheduler versus reconstructed endpoint update at every A timestep from
one cached prediction (`atol=rtol=2e-6`), without extra denoiser evaluations.

A completed successfully, job `19723014`, A40 `g103`. The image is a coherent,
detailed temple-ruins landscape with natural colors. Same-prediction native vs
implied one-step MAE averaged `2.18335e-8`; maximum absolute error across all
steps/elements `4.76837e-7`. All pointwise tolerance checks passed. Actual guided
predictions: 40. Sampling including final decode and CPU state snapshots:
32.120 s; peak allocated/reserved 7.290/9.707 GiB. Only after inspecting this
image and oracle, B was submitted as an independent 40-step trajectory.

B completed, job `19723033`, with the same A40, checkpoint/configuration,
conditioning hash, initial Gaussian and exact schedule as A. Both images show
the same coherent scene and comparable detail. B uses 40 guided predictions;
no native A rerun was needed. Independent trajectory native-vs-implied latent
MAE averages `0.00233985`, ending at `0.00795768`; maximum pointwise latent
difference across steps is `0.591847`. Final raw decoded RGB MAE/RMSE/max is
`0.0107543 / 0.0175434 / 0.760742`. Thus B approximately reproduces native image
quality, but does not reproduce the full trajectory bit-for-bit. This is distinct
from the same-prediction one-step oracle, which remains near float32 roundoff;
small state differences can propagate through low-precision neural predictions.
Sampling: 18.366 s, peak allocated/reserved 7.288/9.713 GiB. After numerical and
visual A/B validation, C/D were submitted sequentially within one allocation.

Cached VAE configurations provide these cross-model context measurements. SD2's
checkpoint omits scaling; the installed `AutoencoderKL` constructor supplies
its 0.18215 default. These are different native coordinate conventions, not a
quality ranking or causal explanation.

| Backend | VAE | Channels | RGB/native factor | Scale | Shift | Local RGB patch | Steps / guidance |
|---|---|---:|---:|---:|---:|---:|---|
| SD2 | AutoencoderKL | 4 | 8 | 0.18215 | none | 512 | 30 / 7.5 |
| SANA | AutoencoderDC | 32 | 32 | 0.41407 | none | 1024 | 20 / 4.5 |
| FLUX | AutoencoderKL | 16 | 8 | 0.3611 | 0.1159 | 1024 | 20 / 3.5 embedded guidance |
| SD3.5 | AutoencoderKL | 16 | 8 | 1.5305 | 0.0609 | 1024 | 40 / 4.5 CFG |

SD3.5's A–G VAE tiling is off; the existing FLUX controls use tiling. SD2 uses
DDIM epsilon prediction; the other three latent models use their validated
flow-style parameterizations, with different schedulers/sigma distributions.
These differences limit causal claims from the cross-model comparison. Within
SD3.5, every A–G stage holds its model/settings/schedule fixed.

C/D completed successfully in job `19723041`, A40 `g103`, with 40 guided
predictions each and passing real-VAE identity preflights. **C severely degrades**:
washed-out colors, broad horizontal bands, patterned/grid artifacts and loss of
natural detail. C contains no fusion, so this isolates roundtrip-induced
trajectory damage. C/B final raw RGB MAE/RMSE/max: `0.422407 / 0.513250 / 2.92969`.
C VAE residual MAE first/last/peak/mean:
`0.463758 / 0.0412348 / 0.463758 / 0.103995`.

**D restores the natural single-patch appearance near B.** Mean per-step
corrected-clean recovery MAE is `6.98506e-11`, with maximum absolute error
`1.19209e-7`. D/B final raw RGB MAE/RMSE/max is
`0.00996909 / 0.0161135 / 0.663086`: again, clean identity holds near roundoff,
while independent low-precision trajectories are not bit-identical. D VAE
residual MAE first/last/peak/mean:
`0.463758 / 0.0528948 / 0.463758 / 0.109234`. Correction uses the same explicit
`E(D(z0)) + [z0-E(D(z0))]` helper as the other latent models, with no shortcut.

C/D sampling times: 27.741 / 27.567 s. Peak allocated/reserved GiB:
C 7.290/9.322, D 7.296/9.266. After inspecting both images and the recovery
oracle, E was submitted to establish the healthy multi-patch native reference.

E completed successfully, job `19723049`, A40 `g103`: a detailed, meaningful
ruins/grass/mountain panorama with coherent large-scale content. A visible
vertical sky/lighting transition remains in the native reference, so E is not
claimed seam-free. Coverage is 100%; guided predictions 120. Sampling with
cached-prediction clean-RGB overlap diagnostics took 67.111 s; peak allocated/
reserved 9.677/14.135 GiB. Mean pre-fusion clean RGB overlap MAE: `0.109270`.
The global CPU-seeded raw native initialization is saved under the paired
rgb-endpoint-controls root as `initial_native.pt`; F/G must load its exact values
and match its tensor SHA-256. After this native-reference inspection, F was
submitted with the same settings and exact native crops.

F completed, job `19723051`, with 120 guided predictions and matching E's exact
initialization/geometry/settings. The image is **severely degraded**: washed-out
flat areas, ghosted architecture, patterned artifacts, and hard vertical/horizontal
patch transitions. Mean pre-fusion RGB overlap MAE is `0.0835898`; mean RGB
consensus correction `0.0278633`; mean own-patch native VAE roundtrip MAE
`0.0913996`. Lower RGB overlap disagreement than E does not imply higher quality:
F loses structure and contrast. Sampling: 96.751 s; peak allocated/reserved
7.378/9.555 GiB. After inspecting F, G was submitted as the final paired test,
adding only the existing synchronized training-free VAE residual correction.

G completed successfully, job `19723061`, A40 `g104`, with matching settings,
schedule, conditioning hash and exact saved E/F initial native tensor. Real-VAE
identity preflight passed. Actual guided predictions: 120; extra denoiser calls:
zero. **G does not repair the full SD3.5 panorama.** Natural colors and detailed
architecture return in the outer quarters, but the central half (the overlap
regions) remains strongly blurred/blocky, with ghosted geometry and hard patch
boundaries. Some F artifacts disappear or change form, but G remains far below
the detailed native E reference.

**SD3.5 behaves more like SANA/SD2 than FLUX with respect to multi-patch VAE
residual correction in this controlled experiment.** All three single-patch
controls show B good, C degraded, and D restored, but SD3.5 joins SANA/SD2 in
showing that synchronized residual correction is insufficient for multi-patch
quality recovery. No model settings or correction gains were tuned to force
this result.

| Backend | B good? | C degraded? | D restored? | E good? | F good? | G restored? |
|---|---|---|---|---|---|---|
| SD2 | Yes | Severe posterization | Yes, approximately | Yes | No | No; substantial blur/posterization and boundaries remain |
| SANA | Yes | Severe grid/banding | Yes, approximately | Yes | No | No; overlap regions remain broken |
| FLUX | Yes | Gray/washed-out and softer | Yes, approximately | Yes | Structure survives, but washed out | Largely restores color/contrast/detail; seams remain |
| SD3.5 | Yes, close to A | Severe patterned artifacts/banding and color loss | Yes, approximately | Yes; sky lighting transition | No | No; detailed outer regions, blurred/blocky overlaps and strong boundaries |

PixelDiT remains the previously measured pixel-native control: native generation
and direct pixel clean consensus produced meaningful results. VAE C/D/G are not
analogous for that backend. No old-model GPU experiment was rerun for this task.

The observations do not support a diffusion-versus-flow explanation: SD3.5,
FLUX and SANA all use flow-style parameterizations here, yet their G outcomes
differ. SD3.5 and FLUX also share 16-channel/factor-8 AutoencoderKL representations,
so those shared properties alone do not guarantee recovery in these settings.
The result is consistent with remaining cross-patch clean/residual incompatibility,
but no additional ablation identifies a unique architectural cause. VAE weights,
scaling, scheduler distributions, guidance, tiling and perturbation sensitivity
remain potential factors; this single prompt/seed comparison does not isolate
them causally. The correction has an exact identity-view algebraic guarantee,
not a general guarantee that encoding fused RGB commutes with latent fusion.

Completed run accounting (sampling excludes model setup/text encoding; E
includes diagnostic clean decoding; reserved memory depends on allocator history):

| Phase | Guided predictions | Sampling seconds | Peak allocated GiB | Peak reserved GiB | Job | Node | Whole Python seconds |
|---|---:|---:|---:|---:|---|---|---:|
| A | 40 | 32.120 | 7.290 | 9.707 | 19723014 | g103 | 219.027 |
| B | 40 | 18.366 | 7.288 | 9.713 | 19723033 | g103 | 104.140 |
| C | 40 | 27.741 | 7.290 | 9.322 | 19723041 | g103 | 59.066 |
| D | 40 | 27.567 | 7.296 | 9.266 | 19723041 | g103 | 58.006 |
| E | 120 | 67.111 | 9.677 | 14.135 | 19723049 | g103 | 94.696 |
| F | 120 | 96.751 | 7.378 | 9.555 | 19723051 | g103 | 128.806 |
| G | 120 | 99.277 | 7.370 | 9.314 | 19723061 | g104 | 284.576 |

All model jobs exited 0. Slurm elapsed A/B/CD/E/F/G: 5:06 / 1:58 / 2:26 /
1:41 / 2:23 / 6:08, including environment startup; C/D share one allocation.
Sampling-time differences include first-use/kernel/cache effects, so these are
resource measurements rather than a controlled performance ranking.

Per-step C/D JSON/CSV contain roundtrip MAE/RMSE/max, residual standard deviation,
residual MAE relative to clean-latent standard deviation, and D recovery
MAE/RMSE/max. C/D mean normalized residuals are `0.103488 / 0.107946`. G records
local/fused VAE residual and RGB consensus correction diagnostics at all 40 steps.

| Diagnostic (mean across timesteps) | E | F | G |
|---|---:|---:|---:|
| Pre-fusion RGB overlap mean MAE | 0.109270 | 0.083590 | 0.086940 |
| Pre-fusion RGB overlap max MAE | 0.125006 | 0.103559 | 0.105062 |
| RGB consensus correction MAE | n/a (native fusion) | 0.027863 | 0.028980 |

G residual diagnostic time averages:

| Quantity | Native units |
|---|---:|
| local_vae_residual_mae_mean | 0.101342 |
| local_vae_residual_mae_max | 0.117394 |
| fused_residual_mae_mean | 0.099069 |
| fused_residual_std | 0.128775 |
| residual_correction_magnitude | 0.095827 |

Common PNG seam diagnostics use the same internal boundaries (x=512,1024,1536)
and nearby ±8-pixel reference as the other backends, on RGB [0,1]:

| Phase | Boundary gradient | Nearby gradient | Boundary excess | Boundary / nearby |
|---|---:|---:|---:|---:|
| E | 0.031482 | 0.034587 | -0.003105 | 0.910 |
| F | 0.022104 | 0.017678 | 0.004426 | 1.250 |
| G | 0.112597 | 0.026738 | 0.085859 | 4.211 |

G increases boundary excess from F's `0.004426` to `0.085859`, and the ratio
from `1.250` to `4.211`. F's relatively low seam ratio and RGB overlap error do
not make it a good image: washed-out, structureless regions can agree. Likewise,
E's ratio below one does not rule out broad visible sky lighting transitions.
Image inspection remains necessary alongside these local metrics.

Final artifacts, in their corresponding existing categories:

- **A**: [native_final.png](../outputs/native-controls/sd35-trajectory-global_native_canvas-average-uniform/20260910-sd35-v1/native_final.png). Metadata: `outputs/native-controls/sd35-trajectory-global_native_canvas-average-uniform/20260910-sd35-v1/A_control.json`.
- **B**: [implied_endpoint_final.png](../outputs/endpoint-controls/sd35-trajectory-global_native_canvas-average-uniform/20260910-sd35-v1/implied_endpoint_final.png). Metadata: `outputs/endpoint-controls/sd35-trajectory-global_native_canvas-average-uniform/20260910-sd35-v1/B_control.json`.
- **C**: [C_final.png](../outputs/vae-residual-controls/20260910-lookingglass-v1/sd35/CD/C_final.png). Metadata: `outputs/vae-residual-controls/20260910-lookingglass-v1/sd35/CD/C_control.json`.
- **D**: [D_final.png](../outputs/vae-residual-controls/20260910-lookingglass-v1/sd35/CD/D_final.png). Metadata: `outputs/vae-residual-controls/20260910-lookingglass-v1/sd35/CD/D_control.json`.
- **E**: [result.png](../outputs/rgb-endpoint-controls/sd35-implied_endpoint_consensus-global_native_canvas-average-uniform/20260910-sd35-v1/native_multidiffusion/result.png). Metadata: `outputs/rgb-endpoint-controls/sd35-implied_endpoint_consensus-global_native_canvas-average-uniform/20260910-sd35-v1/native_multidiffusion/E_control.json`.
- **F**: [result.png](../outputs/rgb-endpoint-controls/sd35-implied_endpoint_consensus-global_native_canvas-average-uniform/20260910-sd35-v1/implied_endpoint_consensus/result.png). Metadata: `outputs/rgb-endpoint-controls/sd35-implied_endpoint_consensus-global_native_canvas-average-uniform/20260910-sd35-v1/implied_endpoint_consensus/F_control.json`.
- **G**: [result.png](../outputs/vae-residual-controls/20260910-lookingglass-v1/sd35/G/generation/result.png). Metadata: `outputs/vae-residual-controls/20260910-lookingglass-v1/sd35/G/generation/G_control.json`.

A also saves the exact Gaussian, 40 native-state snapshots and same-prediction
one-step errors; B saves independent trajectory errors without rerunning A.
C/D share the existing residual `sd35/CD` folder, each with its own control
metadata, per-step metrics and real-VAE preflight. E/F share the same paired
rgb-endpoint-controls root and initial-native tensor. G follows the existing
residual `sd35/G/generation` layout. Every control metadata record identifies
experiment, checkpoint/revision, config, seed, prompt, job, node, precision,
scheduler/sigmas, runtime/memory and exact initialization/conditioning hashes.

CPU report/audit job `19723065` passed. Saved-result audit verifies all A–G
settings, 40/40/40/40/120/120/120 model counts (**520 total**), finite metrics,
shared initializations, complete coverage, schedule identity, local-state audits
and D/preflight recovery bounds. `git diff --check` passes. Full regression
remains **166 passed** (job `19723013`); no additional core changes after that
gate, and no previous experiment files were overwritten.

[A–G contact sheet](../outputs/native-controls/report/sd35/A-G-contact-sheet.png)
([PDF](../outputs/native-controls/report/sd35/A-G-contact-sheet.pdf)) uses equal
512×512 display panels with letterboxing to preserve the panoramas' aspect ratio.
Source images remain untouched. Also saved: A/B/C/D and E/F/G comparison sheets,
A/B numerical-error curves, C/D residual/recovery curves, E/F/G diagnostic curves,
`summary.json` and `audit.json`, all under `outputs/native-controls/report/sd35/`.
The contact sheet and all seven source images were visually inspected.

Reproduce saved-result verification and figures without model inference:

```bash
python scripts/audit_controlled_ladder.py --protocol configs/experiments/trajectory/sd35-ladder.json
python scripts/plot_controlled_ladder.py --protocol configs/experiments/trajectory/sd35-ladder.json
```

## Experiment H — detail-preserving RGB average, all five models (2026-09-10)

All five H runs completed successfully on NVIDIA A40 GPUs. H changes only
`fusion.mode` from `average` to `detail_preserving_average` in the complete saved
G configuration. PixelDiT uses its existing F pixel-native consensus as the
G-equivalent control: it has no VAE and therefore no VAE residual to correct.

### Exact intervention and fairness

The existing `PlanarFusionAccumulator` computes, independently at every RGB
pixel and channel, with signed unclamped model RGB values `u_i`:

```text
ordinary = sum_i u_i / count
detail   = sum_i [u_i * (abs(u_i) + 1e-6)] / sum_i [abs(u_i) + 1e-6]
fused    = ordinary + 1.0 * (detail - ordinary)
```

Spatial weights remain uniform; alpha=1, power=1, epsilon=1e-6 are unchanged
baseline config defaults. DPA is used both for each timestep’s clean-RGB fusion
and for the final fusion of decoded terminal local states. There is no
clamping or conversion to display RGB inside the trajectory. This operator
weights channel magnitude; its name does not imply semantic detail recovery.

For SD2/SANA/FLUX/SD3.5, G’s training-free residual remains
`r_i = z0_i - E(D(z0_i))`. These residuals are still synchronized with uniform
arithmetic averaging in exact native latent coordinates, cropped, and added
to the encoded fused RGB. No detail weighting is applied to latent residuals.
The same-step local implied endpoint is preserved during reconstruction;
all proposals are computed before the synchronous local-state update.

Every H run loads the exact saved baseline global initialization and verifies
its tensor SHA256, exact local crops, conditioning SHA256, complete config
equality except `fusion.mode`, model source, native geometry, actual scheduler
class/timesteps/sigmas or PixelDiT flow schedule, Python/Torch/CUDA versions,
and A40 GPU type. SD3.5 additionally matches the full saved backend details.
Actual per-step endpoint coefficients match the baseline. All four real-VAE
identity preflights passed (maximum clean recovery error <=2e-6). There were
no A–G generation reruns, learned bridges, training, or extra denoiser calls.

### Results

Visual inspection of the saved paired images gives the following result:

| Model | H versus its average-fusion control |
|---|---|
| SD2 | Central overlaps remain severely distorted, with dark geometric areas and saturated red/neon edges; there is no recovery of coherent temple detail. |
| SANA | Central overlaps remain abstract, with stronger saturated yellow/green contours and hard transitions to the detailed outer regions. Boundary diagnostics worsen substantially. |
| FLUX | Retains detailed ruins and natural scene content. Boundary excess and ratio both decrease, though visible exposure/structure transitions remain. This is a limited seam improvement, not complete consistency. |
| SD3.5 Medium | Central overlaps become flatter and more posterized, with lost structure and false-color bands. Lower boundary excess does not indicate recovery. |
| PixelDiT | The H center becomes much darker and is dominated by a large wall, with abrupt transitions to adjacent regions; the average baseline is visibly more coherent. Boundary diagnostics worsen. |

These are paired observations for the existing prompt and seed, not a
multi-seed model ranking. DPA does not fix the latent overlap failures in
SD2, SANA, or SD3.5; it also degrades this PixelDiT example. FLUX remains the
strongest latent result in this comparison, with an improvement in measured
boundary discontinuity but visible seams still present.

### Common final-PNG boundary metrics and H cost

As in G, boundary metrics use displayed PNG RGB in [0,1] and the same patch
start/end lines, with nearby controls at offsets +/-1..8 excluding boundaries.
The ratio and excess measure boundary gradients, not perceptual quality;
flattening or changed contrast can reduce a value despite a worse image.
PixelDiT’s G column below denotes the saved F pixel-native equivalent.

| Model | G seam ratio | H seam ratio | G boundary excess | H boundary excess | H overlap MAE | H seconds | H peak allocated GiB | Job |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| sd2 | 2.450853 | 2.189067 | 0.042584 | 0.033110 | 0.160086 | 14.699 | 2.414 | 19723544 |
| sana | 4.952902 | 6.864848 | 0.213694 | 0.392679 | 0.198697 | 39.049 | 5.552 | 19723545 |
| flux | 4.819027 | 3.531588 | 0.059204 | 0.040376 | 0.080568 | 89.730 | 24.882 | 19723546 |
| sd35 | 4.211111 | 4.327033 | 0.085859 | 0.063114 | 0.073222 | 97.370 | 7.373 | 19723547 |
| pixeldit | 1.242901 | 3.337354 | 0.009745 | 0.065285 | 0.089943 | 35.624 | 4.077 | 19723548 |

Generation seconds include the consensus loop and terminal output fusion,
excluding model loading and identity preflights. G and H use the same GPU
type but separate jobs/nodes; small runtime differences are not a benchmark.

| Model | G overlap MAE | H overlap MAE | G RGB correction MAE | H RGB correction MAE | Predictions |
|---|---:|---:|---:|---:|---:|
| sd2 | 0.151473 | 0.160086 | 0.050491 | 0.053362 | 90 |
| sana | 0.213027 | 0.198697 | 0.071009 | 0.066232 | 60 |
| flux | 0.081327 | 0.080568 | 0.027109 | 0.026856 | 60 |
| sd35 | 0.086940 | 0.073222 | 0.028980 | 0.024407 | 120 |
| pixeldit | 0.099727 | 0.089943 | 0.033242 | 0.029981 | 150 |

Overlap and correction entries are arithmetic means over timesteps. Native
residual magnitudes and full per-step curves are in the linked summary and
diagnostic plots. Their values can change because H changes the trajectory;
the residual formula and synchronization operator are unchanged.

### Validation and artifacts

- CPU regression job **19723524**: **169 tests passed**, including independent signed-channel DPA and full latent/pixel Jacobi trajectory oracles; compilation passed.
- GPU jobs **19723544–19723548**: all completed with exit code 0, for SD2, SANA, FLUX, SD3.5, and PixelDiT respectively.
- Report/audit job **19723549**: all five paired artifact audits passed; **480 guided predictions total**, **0 extra denoiser calls**.
- [All-model G/H contact sheet](../outputs/vae-residual-controls/report/H/G-H-all-models.png) · [PDF](../outputs/vae-residual-controls/report/H/G-H-all-models.pdf).
- [G/H diagnostic curves](../outputs/vae-residual-controls/report/H/G-H-diagnostics.png).
- [Complete numerical summary](../outputs/vae-residual-controls/report/H/summary.json) · [Artifact audit](../outputs/vae-residual-controls/report/H/audit.json).

| Model | H image | Paired G/H image | Central overlap crops | Run metadata |
|---|---|---|---|---|
| sd2 | [H](../outputs/vae-residual-controls/20260910-lookingglass-v1/sd2/H/generation/result.png) | [G/H](../outputs/vae-residual-controls/report/H/sd2-G-H.png) | [Crops](../outputs/vae-residual-controls/report/H/sd2-overlap-crops.png) | [Comparison](../outputs/vae-residual-controls/20260910-lookingglass-v1/sd2/H/comparison.json) |
| sana | [H](../outputs/vae-residual-controls/20260910-lookingglass-v1/sana/H/generation/result.png) | [G/H](../outputs/vae-residual-controls/report/H/sana-G-H.png) | [Crops](../outputs/vae-residual-controls/report/H/sana-overlap-crops.png) | [Comparison](../outputs/vae-residual-controls/20260910-lookingglass-v1/sana/H/comparison.json) |
| flux | [H](../outputs/vae-residual-controls/20260910-lookingglass-v1/flux/H/generation/result.png) | [G/H](../outputs/vae-residual-controls/report/H/flux-G-H.png) | [Crops](../outputs/vae-residual-controls/report/H/flux-overlap-crops.png) | [Comparison](../outputs/vae-residual-controls/20260910-lookingglass-v1/flux/H/comparison.json) |
| sd35 | [H](../outputs/vae-residual-controls/20260910-lookingglass-v1/sd35/H/generation/result.png) | [G/H](../outputs/vae-residual-controls/report/H/sd35-G-H.png) | [Crops](../outputs/vae-residual-controls/report/H/sd35-overlap-crops.png) | [Comparison](../outputs/vae-residual-controls/20260910-lookingglass-v1/sd35/H/comparison.json) |
| pixeldit | [H](../outputs/vae-residual-controls/20260910-lookingglass-v1/pixeldit/H/generation/result.png) | [G/H](../outputs/vae-residual-controls/report/H/pixeldit-G-H.png) | [Crops](../outputs/vae-residual-controls/report/H/pixeldit-overlap-crops.png) | [Comparison](../outputs/vae-residual-controls/20260910-lookingglass-v1/pixeldit/H/comparison.json) |

The manifest, five H YAML configs, guarded runner, launch commands, and exact
output layout are documented in [NATIVE_CONTROLS.md](NATIVE_CONTROLS.md#experiment-h-g-with-detail-preserving-rgb-average-all-five-backends).

## Experiment I — Current-State-Consistent Transition

Experiment I is paired with G, not H. It changes only the post-consensus
transition: G preserves the step-local **pre-fusion endpoint**, whereas I
preserves the **frozen current noisy state**. The G endpoint is not from a
previous timestep. Ordinary uniform RGB averaging, synchronized native VAE
residual correction, final output fusion, model settings, initialization, and
Jacobi semantics are retained. G/H implementations, configs, and saved results
remain unchanged in behavior; no default transition has been switched.

### Equations and measured identities

Let `x` be a frozen current state, `c` its original predicted clean, `c*` the
corrected fused clean, and `e` the pre-fusion endpoint. The validated
parameterization is `x = alpha*c + sigma*e`. G reconstructs
`next_G = alpha_next*c* + sigma_next*e`.

For straight flow (FLUX, SANA, SD3.5), I uses the actual prepared scheduler's
current and next sigmas:

```text
e*     = [x - (1-sigma)*c*] / sigma
next_I = c* + (sigma_next/sigma)*(x-c*)
```

For SD2's deterministic DDIM, I uses the endpoint helper's exact square-root
alpha/sigma coefficients, including `final_alpha_cumprod`:

```text
epsilon* = (x-alpha*c*) / sigma
next_I   = alpha_next*c* + (sigma_next/sigma)*(x-alpha*c*)
```

No sigma is inferred from a loop index or shifted again. At the flow terminal
transition, sigma_next=0 gives next_I=c*. Current sigma=0 is not divided by:
only an already-consistent clean terminal state with next_sigma=0 is accepted;
other cases raise explicitly. The real runs' prepared coefficients are saved.

Writing `delta = c*-c` and substituting the original decomposition gives:

```text
G current-state mismatch tensor = alpha*delta
next_I - next_G                 = -(sigma_next/sigma)*alpha*delta
```

For flow, alpha=1-sigma, giving the requested negative sign:
`-(sigma_next/sigma)*(1-sigma)*(c*-c)`. For DDIM the same general expression
uses its square-root alpha. These identities hold up to floating-point error
in the original decomposition. When delta=0, G and I agree to that tolerance.

I saves actual tensor measurements for every patch/timestep: clean-change MAE,
G-style current-state mismatch, I current-state mismatch and maximum absolute
error, G/I next-state MAE, analytical magnitudes, and signed-identity errors.
It also saves std(x), std(original_clean), and each main MAE normalized by both
standard deviations (denominator floor 1e-12). Per-step CSV/metadata aggregate
patch means and maxima; per-patch JSON/CSV retain every observation.

**Diagnostic scope:** the G-style mismatch and G/I next-state difference are
counterfactual measurements on the **same I source state, model prediction,
and corrected clean**. They isolate the transition rule; they are not
measurements from the independently evolved historical G trajectory. The
final images and seam comparisons use the actual saved G and new I runs.
Counterfactual G tensors never enter I's state updates and require no extra
model evaluations. Runtime assertions verify I's current-state reconstruction
and the signed analytical next-state identity for every patch and timestep.

### Validation and paired run protocol

The final code snapshot passed **177 regression tests**, job **19728583**
(exit 0; compilation passed). The earlier snapshot also passed all 177 tests
(job 19728581). Focused tests cover flow/DDIM current-state consistency,
interpolation equivalence, unchanged-clean equivalence with prepared real
scheduler schedules, old-endpoint independence, zero-correction G/I trajectory
equivalence, patch-order invariance, an independent full Jacobi oracle, exact
denoiser counts, sigma=0 handling, and unchanged saved G/H config snapshots.

The checkout started clean at `ff42ea4003e91208d33f007453ec715f2cb265b3` on
`no_sphere`. The I runner verifies complete G config equality except the new
`consensus_transition.mode=preserve_current_state` field, exact initial tensor
and conditioning hashes, geometry, prepared timestep/sigma arrays, runtime
model metadata, Python/Torch/CUDA versions, and A40 GPU type. The SD3.5 full
backend configuration is checked as well. The inherited default remains
`preserve_prefusion_endpoint`, omitted from legacy serialized snapshots for
backward compatibility. Four real-VAE identity preflights precede generation.

Run order is enforced with Slurm afterok dependencies: SD3.5 **19728585**,
FLUX **19728586**, SANA **19728587**, SD2 **19728588**, followed by report/audit
**19728589**. No PixelDiT GPU run, parameter tuning, A–H rerun, learned bridge,
LPW, time travel, extra model prediction, or new VAE is part of I.

### Completed results (2026-09-11)

All four GPU jobs and the report/audit job completed with exit code 0. The
four paired artifact audits passed: **330 guided predictions total**, with
**zero additional denoiser calls**. All per-patch numerical assertions passed.

**I preserves the current noisy state numerically as designed.** Across all
patches and timesteps, the largest absolute current-state reconstruction error
was **4.76837158203125e-7**. The largest absolute error in the signed analytical
G/I next-state identity was **1.430511474609375e-6**, consistent with float32
arithmetic. These are measured tensor errors, not values substituted from the
analytical formulas.

| Backend | Clean consensus delta MAE | G-style current mismatch | I current mismatch | G/I next-state MAE |
|---|---:|---:|---:|---:|
| sd2 | 0.08949807 | 0.03652336 | 3.986e-09 | 0.03383794 |
| sana | 0.13767992 | 0.02362913 | 4.067e-09 | 0.02026267 |
| flux | 0.09660864 | 0.01842107 | 4.425e-09 | 0.01506745 |
| sd35 | 0.06179203 | 0.01064827 | 4.460e-09 | 0.00971745 |

Entries above average all patches and timesteps and are in each model’s raw
native latent units, not display RGB units. G-style and G/I entries are the
same-source-state counterfactual diagnostics described above, not independent
G-trajectory measurements. Per-step patch means/maxima and per-patch values
are retained in the run artifacts. Absolute latent MAEs are not directly
comparable across models with different latent scales.

| Backend | Max patch clean MAE | Max patch G mismatch | Max patch I mismatch | Max patch next-state MAE | Max pixel I reconstruction error | Max pixel signed-identity error |
|---|---:|---:|---:|---:|---:|---:|
| sd2 | 0.250298 | 0.0759195 | 7.54315e-09 | 0.0722448 | 4.76837e-07 | 1.43051e-06 |
| sana | 0.608846 | 0.0573387 | 7.88148e-09 | 0.0522172 | 2.38419e-07 | 1.26008e-06 |
| flux | 0.321685 | 0.0430465 | 8.5001e-09 | 0.0360271 | 4.76837e-07 | 7.73929e-07 |
| sd35 | 0.32654 | 0.0241532 | 8.9135e-09 | 0.0233796 | 4.76837e-07 | 1.02178e-06 |

The patch maxima above are maxima of each patch’s MAE over all timesteps;
pixel maxima are the maximum absolute individual native tensor-element error.

| Backend | Mean clean delta / std(original clean) | Mean G mismatch / std(current state) | Mean I mismatch / std(current state) | Mean next-state gap / std(current state) |
|---|---:|---:|---:|---:|
| sd2 | 0.149907 | 0.0390484 | 4.16691e-09 | 0.0360082 |
| sana | 0.162324 | 0.0319831 | 5.22824e-09 | 0.0272297 |
| flux | 0.106163 | 0.0240072 | 5.54488e-09 | 0.0196106 |
| sd35 | 0.0594986 | 0.0131829 | 5.49474e-09 | 0.0121135 |

The SD2 final transition retained its actual DDIM coefficients:
alpha=0.9991476535797119, sigma=0.04127926379442215,
next_alpha=0.9995748996734619, next_sigma=0.029155133292078972.
It was not replaced with a flow-style zero-sigma terminal transition.

### Visual quality and seam comparison

Visual inspection of each actual G/I pair gives:

| Backend | Content quality versus G | Remaining seam quality |
|---|---|---|
| SD2 | Substantial recovery: the neon/geometric central failure is replaced by recognizable ruins, grass, and terrain. | Patch-boundary lighting/texture transitions remain; the ratio improvement is small despite the clear content recovery. |
| SANA | Substantial recovery: detailed temple structures and vegetation replace the abstract/banded central overlaps. | A center lighting/structure discontinuity remains; boundary metrics improve strongly. |
| FLUX | Remains detailed and naturally colored; there is no quality collapse from switching the transition. | The center lighting seam remains visible, but both measured boundary metrics decrease. |
| SD3.5 Medium | Substantial recovery: recognizable architecture and landscape replace the blurred/posterized central regions. | Vertical lighting transitions remain; boundary metrics improve strongly. |

| Model | G seam ratio | I seam ratio | G boundary excess | I boundary excess | I overlap MAE | I seconds | I peak allocated GiB | Job |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| sd35 | 4.211111 | 2.347562 | 0.085859 | 0.024440 | 0.097698 | 98.521 | 7.374 | 19728585 |
| flux | 4.819027 | 2.145042 | 0.059204 | 0.028474 | 0.097220 | 91.477 | 24.881 | 19728586 |
| sana | 4.952902 | 2.640130 | 0.213694 | 0.039949 | 0.188969 | 38.598 | 5.555 | 19728587 |
| sd2 | 2.450853 | 2.359293 | 0.042584 | 0.030489 | 0.089229 | 14.546 | 2.416 | 19728588 |

As in G/H, seam diagnostics operate on saved PNG RGB in [0,1], at exact
internal patch start/end lines versus nearby lines at offsets +/-1..8 excluding
all exact boundaries. Both ratio and excess decrease for all four models.
They are boundary-gradient diagnostics, not a perceptual quality score;
changed scene edges and contrast also affect them. The final images still
show visible seams, so the transition does not solve all panorama consistency.

Generation time above includes the additional tensor diagnostics but no extra
model calls; model loading and identity preflights are excluded. All four I
jobs used NVIDIA A40 on g113. Their Slurm elapsed times were 9:59 (SD3.5),
9:24 (FLUX), 6:33 (SANA), and 5:01 (SD2); startup dominated these elapsed times.
Allocated/reserved memory and complete runtime metadata are in each comparison
JSON. The saved G controls use the same GPU type, in different jobs.

### Interpretation and default-transition recommendation

**This matches Case C for the tested prompt/seed:** FLUX remains good, while
SD3.5, SANA, and SD2 recover substantially. Only the post-fusion transition
changed. The result therefore supports preserving the pre-fusion endpoint
after changing the clean prediction as a major cause of G’s central-overlap
degradation in these controlled runs. It also explains much of the apparent
backend difference without changing the VAE, residual correction, or fusion.
The tensor measurements confirm the intended mechanism: G-style current-state
mismatch is substantial, I’s is at floating-point error, and the measured
next-state change obeys the signed clean-consensus identity.

**The evidence supports adopting `preserve_current_state` for future
post-RGB-consensus transitions**, subject to confirming the result across more
prompts and seeds. This experiment uses one existing prompt and seed per
backend; it is not a population-level benchmark or proof that this is the
only difference from the complete LookingGlass method. Residual seams remain
and need separate investigation. Per the task, the actual default remains
`preserve_prefusion_endpoint`: G/H were not changed and no further technique
or parameter tuning was added.

### Experiment I artifacts

- [Four-model G/I contact sheet](../outputs/vae-residual-controls/report/I/G-I-all-models.png) · [PDF](../outputs/vae-residual-controls/report/I/G-I-all-models.pdf).
- [Current-state and next-state diagnostic curves](../outputs/vae-residual-controls/report/I/I-transition-diagnostics.png) · [PDF](../outputs/vae-residual-controls/report/I/I-transition-diagnostics.pdf).
- [G/I RGB-overlap and residual curves](../outputs/vae-residual-controls/report/I/G-I-diagnostics.png).
- [Complete numerical summary](../outputs/vae-residual-controls/report/I/summary.json) · [Artifact audit](../outputs/vae-residual-controls/report/I/audit.json).

| Backend | I image | Paired G/I | Central overlap crops | Run metadata | Per-patch transition diagnostics |
|---|---|---|---|---|---|
| sd2 | [I](../outputs/vae-residual-controls/20260910-lookingglass-v1/sd2/I/generation/result.png) | [G/I](../outputs/vae-residual-controls/report/I/sd2-G-I.png) | [Crops](../outputs/vae-residual-controls/report/I/sd2-overlap-crops.png) | [Comparison](../outputs/vae-residual-controls/20260910-lookingglass-v1/sd2/I/comparison.json) | [CSV](../outputs/vae-residual-controls/20260910-lookingglass-v1/sd2/I/transition_patches.csv) |
| sana | [I](../outputs/vae-residual-controls/20260910-lookingglass-v1/sana/I/generation/result.png) | [G/I](../outputs/vae-residual-controls/report/I/sana-G-I.png) | [Crops](../outputs/vae-residual-controls/report/I/sana-overlap-crops.png) | [Comparison](../outputs/vae-residual-controls/20260910-lookingglass-v1/sana/I/comparison.json) | [CSV](../outputs/vae-residual-controls/20260910-lookingglass-v1/sana/I/transition_patches.csv) |
| flux | [I](../outputs/vae-residual-controls/20260910-lookingglass-v1/flux/I/generation/result.png) | [G/I](../outputs/vae-residual-controls/report/I/flux-G-I.png) | [Crops](../outputs/vae-residual-controls/report/I/flux-overlap-crops.png) | [Comparison](../outputs/vae-residual-controls/20260910-lookingglass-v1/flux/I/comparison.json) | [CSV](../outputs/vae-residual-controls/20260910-lookingglass-v1/flux/I/transition_patches.csv) |
| sd35 | [I](../outputs/vae-residual-controls/20260910-lookingglass-v1/sd35/I/generation/result.png) | [G/I](../outputs/vae-residual-controls/report/I/sd35-G-I.png) | [Crops](../outputs/vae-residual-controls/report/I/sd35-overlap-crops.png) | [Comparison](../outputs/vae-residual-controls/20260910-lookingglass-v1/sd35/I/comparison.json) | [CSV](../outputs/vae-residual-controls/20260910-lookingglass-v1/sd35/I/transition_patches.csv) |

The scientific label `experiment=I` is explicit in each comparison JSON.
The inherited config experiment name is deliberately unchanged so the config
difference remains exactly the transition-mode addition. Each run retains
the config/prompt, exact initialization, hashes, actual schedule, seed, job,
preflights, image, per-step statistics, and per-patch diagnostics. Reproduction
commands and the manifest are documented in
[NATIVE_CONTROLS.md](NATIVE_CONTROLS.md#experiment-i-current-state-consistent-transition).

## Experiment J — Planar RGB Consensus Without VAE Residual, Current-State Interpolation

J completes the 2x2: F/J have no VAE residual correction and G/I have residual
correction; F/G preserve the pre-fusion endpoint and J/I preserve current x_t.
J still uses the model's original VAE for clean decode and fused-RGB encode.
It does not compute or add `z0-E(D(z0))`. F's diagnostic-only own-RGB encode is
also disabled, leaving one fused-RGB encode per patch/timestep.

J's replacement clean is exactly `c = E(exact_crop(average(decoded_clean)))`.
The validated current-state helper is unchanged: flow uses
`c+(sigma_next/sigma)*(x_t-c)` and SD2 uses
`alpha_next*c+(sigma_next/sigma)*(x_t-alpha*c)` with its actual DDIM coefficients.
The saved F global native initialization and conditioning hashes, complete
model settings, geometry, prepared schedule, guidance, and A40 GPU type are
checked before generation. Predictions are one per patch/timestep; RGB fusion
and local-state commitment retain strict Jacobi behavior.

All **179 regression tests passed** (job **19731955**), including the prior
flow/DDIM identities and new tests that make VAE-residual calls fail, require
only one encode per patch, verify exact saved-F config pairing, and check order
invariance. I's pre-existing uncommitted work and historical serialized settings
were preserved. The task started on `no_sphere` at HEAD
`ff42ea4003e91208d33f007453ec715f2cb265b3`; each new run stores its commit and dirty
status in `repository.json`. The explicit no-residual flag is bookkeeping for
F's existing method, not a second scientific change.

J jobs run in order: SD3.5 **19731976**, FLUX **19731978**, SANA **19731979**,
SD2 **19731980**; the F/J/G/I report/audit job is **19731981**. No K implementation
begins until the J correctness gate and SD3.5 J image inspection pass.

### J results and the F/J/G/I factorial conclusion

All four J jobs and artifact audit **19731981** completed successfully: **330**
guided predictions total, exactly matching F (120 SD3.5, 60 FLUX, 60 SANA,
90 SD2). Maximum current-state reconstruction error was **4.7684e-7 for every
backend**. Mean counterfactual old-transition current-state mismatch on the J
trajectory was 0.02772 / 0.03731 / 0.04103 / 0.05677 respectively. These are
counterfactual transitions evaluated on J's frozen states, not a subtraction
of independently evolved F and J trajectories.

| Backend | J runtime seconds | Peak allocated / reserved GiB | F / J / G / I boundary-to-nearby gradient ratio | Visual factorial conclusion |
|---|---:|---:|---|---|
| SD3.5 | 103.47 | 7.37 / 9.73 | 1.250 / 1.904 / 4.211 / 2.348 | F's central loss of content recovers in J; G's hard central degradation also recovers in I. Current-state transition is the major recovery factor. J retains grid-like texture and exposure changes; I looks cleaner. Residual correction still affects texture. |
| FLUX | 93.54 | 24.88 / 27.29 | 8.005 / 2.527 / 4.819 / 2.145 | Detail survives all four. F→J greatly reduces boundary contrast; G→I improves it further within the residual pair. J retains hazy lighting; I has stronger color/contrast. Both factors affect appearance, with interpolation useful even without residual. |
| SANA | 31.06 | 5.54 / 8.50 | 2.423 / 2.928 / 4.953 / 2.640 | F's oversaturated, fragmented center becomes detailed, connected ruins in J; G→I similarly recovers its degraded overlap. J still has strong vertical exposure boundaries. Interpolation drives the structural recovery; residual is not required for it. |
| SD2 | 15.03 | 2.41 / 2.87 | 2.169 / 2.044 / 2.451 / 2.359 | F's colorful fragmented output becomes coherent ruins/grass in J. G has less extreme corruption but a degraded center; I recovers it. Residual can mitigate the old transition's damage, but current-state interpolation recovers structure without it. |

These timings cover the generation loop and output-state decoding, not model
loading; diagnostic overhead is included. J skips F's diagnostic-only
own-RGB encode, so this is not a pure performance benchmark of the transition.
Boundary metrics use full displayed PNG RGB [0,1], exact planar patch-edge
lines, and eight nearby lines on each side. A low ratio on a corrupt or flat
image does not imply quality: SD3.5 F and SANA F illustrate why the images and
absolute gradients must be considered. J boundary/nearby gradients are
0.04162/0.02186, 0.03988/0.01578, 0.07070/0.02415, and 0.05836/0.02855 in the
same backend order.

**For this fixed prompt/seed, current-state interpolation is sufficient for
major planar content recovery without residual correction. It is not
sufficient for a seamless, artifact-free panorama.** The two factorial
transition comparisons (F→J and G→I) agree on the principal structural result;
residual correction has a backend-dependent interaction with detail, color,
and the older inconsistent transition. This is one paired seed per model,
not evidence that either factor has the same effect across all prompts.

- [Full F/J/G/I contact sheet](../outputs/vae-residual-controls/report/J/F-J-G-I-all-models.png), with uncropped images and the per-model F|J / G|I layout in [SD3.5](../outputs/vae-residual-controls/report/J/sd35-F-J-G-I.png), [FLUX](../outputs/vae-residual-controls/report/J/flux-F-J-G-I.png), [SANA](../outputs/vae-residual-controls/report/J/sana-F-J-G-I.png), [SD2](../outputs/vae-residual-controls/report/J/sd2-F-J-G-I.png).
- [Summary with all per-patch diagnostic aggregates](../outputs/vae-residual-controls/report/J/summary.json), [audit](../outputs/vae-residual-controls/report/J/audit.json), [J gate allowing K](../outputs/vae-residual-controls/report/J/k_gate.json).
- J output folders: `outputs/vae-residual-controls/20260910-lookingglass-v1/{sd35,flux,sana,sd2}/J/`; image `generation/result.png`, per-patch `transition_patches.csv/json`, per-step metadata, comparison, provenance, and exact F initialization copies are retained.

## Experiment K — ERP Standard-Warp RGB Consensus Without VAE Residual, Current-State Interpolation

K was implemented only after J's 179-test gate, numerical checks, and visual
inspection of the meaningful SD3.5 J image. It reuses the unchanged current-state
helper introduced by I and the existing `StandardWarpOperator`; no new projector
or model was added. Its sequence is:

`perspective clean predictions → standard warp to ERP → validity-aware RGB average → standard warp back to perspective → VAE encode → current-x_t transition`.

**ERP is a transient RGB consensus representation only. There is NO spherical
latent representation or ERP latent tensor.** Six persistent local noisy native
tensors belong to six immutable perspective camera slots. Each timestep freezes
all six states, predicts/decodes all six clean views, fuses all RGB contributions,
projects/encodes all six synchronized views, computes all six next states, then
commits them together. There is no residual computation/correction, fixed-noise
bank or renoising, LPW, DPA, dynamic camera motion, learned bridge, time travel,
or extra denoiser prediction. Standard ERP→perspective uses the existing nearest
sampling default; perspective→ERP uses the existing bilinear default, with
existing wrap/pole handling and validity masks. Average/uniform fusion uses
valid contributors only.

### Fixed K geometry, initialization and validation

The first-pass geometry was fixed before seeing any K model output: six camera
slots in order `(yaw,pitch)` degrees `(0,0), (90,0), (180,0), (-90,0), (0,90),
(0,-90)`, all roll 0 and horizontal/vertical FOV **100°**. The four equatorial
and two polar views form an overlapping cube cover. This smaller symmetric
cover provides full-sphere coverage without the cost of the historical 89-view
sampler; no camera count/FOV was tuned by backend quality.

| Models | Perspective RGB | ERP RGB | Coverage | Multiple contributors | Min / max contributors |
|---|---|---|---:|---:|---:|
| SD3.5, FLUX, SANA | 1024×1024 | 1024×2048 | 100% | 14.8460% | 1 / 3 |
| SD2 | 512×512 | 512×1024 | 100% | 14.8376% | 1 / 3 |

The actual-resolution **CPU preflight preceded GPU model runs**; it required
100% coverage and exact preservation of a constant RGB signal within 1e-6.
The GPU preflight must match its camera hashes and coverage statistics.
Saved ordered camera hashes are `2d2fc24456a397c20483aaf2a7e15e29daa76f8e641281750b72d8368162b198`
for the 1024 views; SD2's size-specific hash is in its geometry JSON. All steps
assert the camera definitions are unchanged. Contributor maps and camera
records are under `outputs/vae-residual-controls/report/K/geometry/`.

One CPU `torch.Generator` seeded with the unchanged experiment seed 0 samples
independent native Gaussians in camera order; backend initialization applies
its normal initial-noise scale. The saved initial tensors/checksums are local
only. J and K cannot share bit-identical initialization: J crops one planar
Gaussian, while K samples independent perspective-native tensors. No latent
noise is projected to make them artificially match. The retained
`native_multidiffusion.patch_size` setting prepares the backend at J's native
local resolution; K never allocates that configuration's global native canvas.

The full suite passed **184 tests** in CPU job **19732210**, followed by the
full-resolution coverage preflight. New tests cover fixed camera identity,
deterministic native initialization, real standard projection of a smooth
synthetic wrap-crossing feature, constant-signal coverage and gap rejection,
Jacobi order invariance, exactly one prediction/encode per slot/step, RGB-only
projection inputs, and forbidden residual/fixed-noise calls. Existing flow/DDIM
current-state identity tests remain in the suite. Runtime diagnostics additionally
assert actual-backend current-state reconstruction and the signed counterfactual
next-state identity. The synthetic wrap test is an interpolation-tolerance test,
not evidence that independent generated views will agree semantically.

K model job **19732229** runs SD3.5→FLUX→SANA→SD2 sequentially on one A40;
report/audit job **19732232** follows. Within each backend, the J checkpoint,
conditioning hash, prompt, negative prompt, seed, local resolution, dtype,
schedule, steps, CFG/guidance and software metadata are checked unchanged.
The J/K comparison changes geometry, camera context, view count and the necessary
initialization policy; it is an algorithm-transfer comparison, not the
single-variable bit-identical F/J causal ablation.

### K visual outcome and scientific interpretation

All four models completed without a resource fallback or settings change.
Their full terminal ERPs and local views were visually inspected. SD3.5 retains
recognizable ruins but loses texture and shows broad camera-edge transitions.
FLUX retains the clearest local temple detail, yet also has strong region
boundaries and inconsistent polar content. SANA has recognizable temples but
large exposure/color discontinuities and a heavily saturated central region.
SD2 produces recognizable ruins with blurred/mismatched camera regions. In
all four, polar views contain ordinary horizon-oriented landscape/architecture
that does not fit a coherent full sphere. These are substantial defects, not
merely the expected stretching of an otherwise coherent scene in an ERP display:
SD3.5's saved perspective views themselves show hard rectangular transitions,
and its polar perspective view contains a conventional landscape horizon.

The mean pre-fusion neighboring RGB overlap MAE falls from first to last step:
SD3.5 **0.69523→0.00411**, FLUX **0.65784→0.01163**, SANA **0.85371→0.02570**,
SD2 **0.64790→0.01814**. These diagnostics use raw decoded RGB (normally [-1,1],
without PNG clipping) and average the 12 valid camera-pair MAEs equally.
Current-state reconstruction stays within **4.7684e-7** for the three flow
models and **9.5367e-7** for SD2. Low final overlap disagreement is not proof of
semantic coherence: repeated consensus makes pixels agree while the larger
scene layout remains incompatible. Shared grass, mountain and sky colors and
some local contours connect across regions, but architecture and horizons do
not consistently continue as one scene across the camera boundaries.

**Current-state interpolation transfers numerically to standard perspective ↔
ERP synchronization, but this first K control does not deliver a visually
coherent panorama.** The major visible limitation is cross-view semantic/layout
and exposure consistency, particularly between equatorial and polar views,
with resampling blur and hard validity-boundary transitions also present.
There is no evidence here of a broken horizontal-wrap convention or missing
coverage: the geometric tests pass and all pixels are covered. Conversely,
a passing smooth-feature projection test does not rule out detail loss under
repeated resampling. The ERP angular sampling is coarser than the native views,
and ordinary hard-mask averaging can imprint transitions at contributor changes.
These observations do not quantitatively separate resampling, VAE round-trip,
independent-noise initialization, limited overlap, and camera-context effects.

J→K therefore does **not** establish that “ERP is bad.” It establishes that
this current-state/RGB-consensus implementation operates with local native
states and standard ERP geometry, while exposing a scene-consistency problem
under the fixed six-camera recipe. Each camera receives the same global temple
prompt and an independent initial Gaussian, so coherent sky/ground orientation
at the poles is not supplied by viewpoint-specific conditioning. That is an
interpretive limitation of this controlled run, not a setting silently changed
after viewing results. Only about 14.85% of ERP pixels have multiple contributors.

The next candidate under the requested decision rule is **time travel / repeated
same-timestep consistency**, because the failure is major semantic disagreement,
not a coherent panorama with slight blur. True LPW could later address detail
and resampling, but this result does not support expecting LPW alone to repair
incompatible scene layouts. Neither candidate, stronger/partial consensus,
a new camera recipe, nor any other future method was implemented. K supports
the feasibility of the no-spherical-latent architecture and its transition math;
it does not yet validate its final panorama quality. Further evidence requires
new controlled experiments and more prompts/seeds.

### K measurements and artifacts

K model job **19732229** and report/audit job **19732232** both completed with exit 0. The artifact audit passed all **660** guided predictions, with no extra predictions. No model was skipped.

| Backend | Predictions | Runtime s | Peak allocated / reserved GiB | Horizontal wrap gradient / nearby | Wrap ratio | Terminal pairwise overlap MAE mean / max |
|---|---:|---:|---:|---:|---:|---:|
| sd35 | 240 | 193.80 | 7.63 / 9.79 | 0.002475 / 0.002840 | 0.872 | 0.003823 / 0.005349 |
| flux | 120 | 177.63 | 25.14 / 27.53 | 0.002724 / 0.002768 | 0.984 | 0.003177 / 0.004924 |
| sana | 120 | 74.11 | 5.82 / 7.82 | 0.007897 / 0.007251 | 1.089 | 0.011292 / 0.017834 |
| sd2 | 180 | 29.03 | 2.48 / 3.01 | 0.006906 / 0.006749 | 1.023 | 0.017534 / 0.022495 |

K runtime covers generation, final terminal decoding/fusion, and diagnostics, excluding model loading. Per-stage model/decode/encode/projection totals and curves are in the summary and diagnostic figures. Horizontal seam measurements use displayed PNG RGB [0,1]: the last-to-first-column absolute difference versus the mean of the eight neighboring differences on each side, ratio denominator floored at 1e-12. They measure the wrap edge only, not all internal camera boundaries. Terminal overlap values use raw decoded RGB and are distinct from the pre-fusion clean-prediction trajectory metrics above.

The wrap ratio is near or below 1 for every backend despite visibly poor panorama consistency. This is direct evidence that wrap smoothness is insufficient as a panorama-quality score. Full source images are retained at native output size; comparison sheets uniformly scale full images without cropping or shifting a seam out of view.

- [Full J/K contact sheet](../outputs/vae-residual-controls/report/K/J-K-all-models.png) · [PDF](../outputs/vae-residual-controls/report/K/J-K-all-models.pdf).
- [K numerical summary](../outputs/vae-residual-controls/report/K/summary.json) · [artifact audit](../outputs/vae-residual-controls/report/K/audit.json) · [job record](../outputs/vae-residual-controls/report/K/jobs.json).

| Backend | Terminal ERP | J/K comparison | Final camera views | Diagnostic curves | Complete run record |
|---|---|---|---|---|---|
| sd35 | [ERP](../outputs/vae-residual-controls/20260910-lookingglass-v1/sd35/K/final_erp.png) | [J/K](../outputs/vae-residual-controls/report/K/sd35-J-K.png) | [Six views](../outputs/vae-residual-controls/report/K/sd35-final-views.png) | [Curves](../outputs/vae-residual-controls/report/K/sd35-diagnostics.png) | [Comparison](../outputs/vae-residual-controls/20260910-lookingglass-v1/sd35/K/comparison.json) |
| flux | [ERP](../outputs/vae-residual-controls/20260910-lookingglass-v1/flux/K/final_erp.png) | [J/K](../outputs/vae-residual-controls/report/K/flux-J-K.png) | [Six views](../outputs/vae-residual-controls/report/K/flux-final-views.png) | [Curves](../outputs/vae-residual-controls/report/K/flux-diagnostics.png) | [Comparison](../outputs/vae-residual-controls/20260910-lookingglass-v1/flux/K/comparison.json) |
| sana | [ERP](../outputs/vae-residual-controls/20260910-lookingglass-v1/sana/K/final_erp.png) | [J/K](../outputs/vae-residual-controls/report/K/sana-J-K.png) | [Six views](../outputs/vae-residual-controls/report/K/sana-final-views.png) | [Curves](../outputs/vae-residual-controls/report/K/sana-diagnostics.png) | [Comparison](../outputs/vae-residual-controls/20260910-lookingglass-v1/sana/K/comparison.json) |
| sd2 | [ERP](../outputs/vae-residual-controls/20260910-lookingglass-v1/sd2/K/final_erp.png) | [J/K](../outputs/vae-residual-controls/report/K/sd2-J-K.png) | [Six views](../outputs/vae-residual-controls/report/K/sd2-final-views.png) | [Curves](../outputs/vae-residual-controls/report/K/sd2-diagnostics.png) | [Comparison](../outputs/vae-residual-controls/20260910-lookingglass-v1/sd2/K/comparison.json) |

Each K folder also contains `metadata.json`, `runtime_preflight.json`, `repository.json`, `spec.json`, `initialization.json`, `initial_local_states.pt`, `contributors.pt/png`, `steps.csv`, `transition_patches.csv/json`, `final_consensus_erp.png`, six final views and five snapshots. Provenance records HEAD, branch and dirty status at run start. The source tree remains on `no_sphere` at `ff42ea4003e91208d33f007453ec715f2cb265b3`, with prior uncommitted I work preserved and additive J/K changes uncommitted. No historical A–I output or configuration was overwritten.

## Experiment L — SphereDiff-Style View/Prompt Strategy

The L/M task began on `no_sphere`, HEAD
`7eb0fff246291701901165b71f8485b2c9d102dc`, with a clean working tree and no
queued jobs. All historical A–K code and outputs are preserved. L/M are additive
controls using one shared streaming implementation of K's local-native/current-x_t
algorithm. They introduce no residual correction, fixed-noise renoising, LPW,
DPA, time travel, partial fusion, changed VAE, or spherical/ERP latent field.

### Official SphereDiff verification

Verified official source at commit
[`2c8c68ba088f2803b3dce4b52b7b0d68bc996139`](https://github.com/pmh9960/SphereDiff/tree/2c8c68ba088f2803b3dce4b52b7b0d68bc996139):
[`spherical_functions.py`](https://github.com/pmh9960/SphereDiff/blob/2c8c68ba088f2803b3dce4b52b7b0d68bc996139/pipelines_ours/spherical_functions.py),
[`pipeline_spherical_sana.py`](https://github.com/pmh9960/SphereDiff/blob/2c8c68ba088f2803b3dce4b52b7b0d68bc996139/pipelines_ours/pipeline_spherical_sana.py), and
[`pipeline_spherical_flux.py`](https://github.com/pmh9960/SphereDiff/blob/2c8c68ba088f2803b3dce4b52b7b0d68bc996139/pipelines_ours/pipeline_spherical_flux.py).
The static pipelines call the dense-equator cover with 80° FOV, 60% horizontal
and vertical overlap, and three extra azimuth samples per ring. The existing
`spherediff_camera_cover` matches this construction. Its default is unchanged;
only an optional overlap parameter was added for M's geometry search.

The resulting **89 cameras** occupy pitch rings −90, −67.5, −45, −22.5, 0,
22.5, 45, 67.5, 90 degrees, with **4, 8, 11, 14, 15, 14, 11, 8, 4** cameras
respectively. Yaws are uniformly spaced from −π, excluding the repeated +π
endpoint. Slots are fixed throughout denoising, with no per-step rotation.

Official prompting expands five lines at phi −90, −10, 0, 10, 90 over yaw
anchors 0, 90, 180, 270 and selects maximum cosine similarity. SphereDiff's
rendering uses the inverse camera rotation and a central negative-z ray; negative
phi denotes an upward-looking camera. DiffPano's physical forward convention
has positive pitch upward. The existing prompt helper already converts the
semantic bands to +90, +10, 0, −10, −90. Tests verify top/upper/equator/lower/bottom,
nonpolar yaw anchors, the cosine rule, and the official cover formula. Global
yaw origin/renderer conventions are not claimed to reproduce identical pixels
from the original spherical-latent renderer. L reproduces its view-cover and
semantic prompt strategy in DiffPano's existing standard projector.

The L slot histogram is `{0:14, 4:5, 5:6, 6:6, 7:6, 8:4, 9:4, 10:3, 11:4,
12:5, 13:6, 14:6, 15:6, 16:14}`. Semantic-band totals are **14/23/15/23/14**.
Equivalent pole anchors can tie; selecting a different yaw at a pole does not
change its prompt text. Each backend caches conditioning once per selected slot.

### Prompt and FOV fairness limits

The user explicitly selected **80° FOV for both L and M**. Original K used
100°, so K→M changes FOV as well as camera count/layout and initialization
cardinality. It must not be described as a strictly single-variable redundancy
ablation. Checkpoints, seed 0, model-native view resolution, ERP resolution,
steps, scheduler, guidance, dtype, VAE options, standard warp and average fusion
remain paired to K; PixelDiT inherits the existing F native control because it
has no six-view K result.

The original K prompt source `prompts/native_control.txt` has **five identical
lines**. That exact file is preserved for L/M; no text was rewritten to favor L.
L assigns directional slots correctly, while M uses K's global equatorial slot
8, but all selected slots contain the same prompt text. Consequently this run
can verify directional routing but **cannot estimate a semantic benefit from
different directional prompt texts**. Because M selects the same geometry as L,
L/M are effectively repeat controls with different slot routing, not a
nontrivial directional-text ablation. This limitation was identified before
GPU runs. The existing night-scene `prompts/ruins.txt` was not substituted.

## Experiment M — Dense ≥5-Coverage ERP Consensus

The geometry-only search uses the same parameterized latitude-ring construction
at 80° FOV, with overlap candidates 0.00, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50,
0.55, 0.60, 0.65, 0.70, 0.75, 0.80 and a cap of 300 cameras. It evaluates
candidates in increasing camera count using actual `StandardWarpOperator`
projections of constant perspective RGB, first at ERP 128×256, then verifies
any passing candidate at both production ERP resolutions before accepting it.
The first accepted cover is the smallest **tested** cover, not a proof of a
continuous/global camera-count optimum.

| Overlap | Cameras tested | Coarse minimum contributors | Coarse coverage |
|---:|---:|---:|---:|
| 0.00 | 16 | 0 | 91.260% |
| 0.10 | 16 | 0 | 91.260% |
| 0.20 | 31 | 2 | 100% |
| 0.30 | 34 | 2 | 100% |
| 0.40 | 37 | 2 | 100% |
| 0.45 | 38 | 2 | 100% |
| 0.50 | 58 | 4 | 100% |
| 0.55 | 63 | 4 | 100% |
| **0.60** | **89** | **7** | **100%** |

The selected M geometry is therefore **identical to L**. Higher-density
candidates were unnecessary. Full-resolution geometry job **19754794** passed
before any model submission:

| ERP / view RGB resolution | Coverage | Min | P01 | P05 | Median | Mean | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512×1024 / 512×512 (SD2) | 100% | **7** | 8 | 9 | 12 | 12.907318 | 19 |
| 1024×2048 / 1024×1024 (other four) | 100% | **7** | 8 | 9 | 12 | 12.905493 | 19 |

All pixels satisfy the hard ≥5 requirement. Only two compact shared geometry
JSONs are saved, each containing both resolution checks, camera hashes and
construction information. No coverage maps or tensors are saved. M's prompt
histogram is `{8:89}`.

| Exp | Geometry | Fused quantity | VAE residual | Transition | Prompt policy |
|---|---|---|---|---|---|
| K | six fixed cameras, 100° | clean RGB | no | current x_t | global slot 8 |
| L | SphereDiff 89 cameras, 80° | clean RGB | no | current x_t | directional slots |
| M | 89 cameras, 80°, full-resolution min=7 | clean RGB | no | current x_t | global slot 8 |

### Shared execution and storage

`DenseERPLocalCurrentStatePipeline` extends K's infrastructure with one common
L/M streaming loop. It retains local native noisy states on CPU, moves one view
at a time to the backend, predicts/decodes once, and accumulates its standard ERP
contribution. Temporary CPU clean-native/RGB copies permit exact consensus-delta
diagnostics without keeping ~89 decoded GPU images. After fusion, it projects,
encodes and interpolates each view from the frozen x_t, then commits all next
states together. The shared `interpolate_from_current_state` is unchanged;
pre-fusion endpoints do not enter the transition. PixelDiT retains its identity
pixel-native encode/decode behavior.

The existing bounded `ProjectionCache` retains four device entries per mapping
and CPU overflow, precomputes all fixed maps once, and reuses them. Only one
full-size projected RGB contribution is retained at a time. Overlap diagnostics
use O(N) per-view disagreement with the fused ERP restricted to multiply-covered
pixels; they are **not** the older K pairwise MAE. Means average per-view means;
maxima are maximum absolute pixel/component errors. Compact overall/first/midpoint/
last metrics are stored; no N² pair list or per-step file is written.

Per-run output is exactly `final_result.png` (standard-fused decoded terminal
local states) and `metadata.json` (settings, source/checkpoint/prompt hashes,
schedule, geometry, job, git provenance, timing/memory, metrics and audit).
No intermediate images, camera images, predictions, masks, or .pt files are
saved. One K/L/M contact sheet is produced after all runs. Missing PixelDiT K
is labeled unavailable rather than rerun.

### L/M validation and jobs

Focused tests (6) and the full **190-test regression suite** passed in job
**19754806**, with compileall and whitespace checks passing. The first focused
attempt rejected a test fixture that had omitted `warp.mode=standard`; correcting
that fixture allowed the streaming-equivalence test to run and pass. No model
jobs ran before validation. A refreshed geometry preflight **19754823** normalized
FOV serialization (`80.0`) and explicitly verified all ten config camera hashes
against the shared full-resolution records. Contributor counts were unchanged.

The new tests cover official camera construction, semantic prompt-band/yaw
selection, correct conditioning rows for all five adapters, deterministic search,
forbidden settings, and a streaming trajectory matching original K within
2e-6 on a small case. They also verify processing-order invariance, exact
prediction/encode/decode counts, frozen source states, no residual/fixed-noise
calls, and successful transitions with the old endpoint poisoned by NaNs.

| Backend | L Slurm job | M Slurm job | Guided predictions per run |
|---|---:|---:|---:|
| SD3.5 | 19754844 | 19754848 | 3560 |
| FLUX.1-dev | 19754845 | 19754849 | 1780 |
| SANA | 19754846 | 19754850 | 1780 |
| SD2 | 19754847 | 19754851 | 2670 |
| PixelDiT | 19754834 | 19754839 | 4450 |

Jobs were submitted L first, then M, in the requested backend order. Independent
jobs may execute concurrently according to Grace availability. All ten runs use
A100 resources. The eight initial A40 jobs and their pending report job were
cancelled before execution after Grace estimated a 6–9 hour wait; no duplicate
model runs occurred. No model settings changed. Original K used A40 for the four
latent models, so observed runtime multipliers are **not hardware-matched**.
The submission record, including cancelled pending job IDs, is in the single
shared `validation.json`. The final report/audit job is **19755015**. The initial CPU report job
19754852 stopped because Diffusers serialized its unordered `_use_default_values`
provenance list in different orders. The audit now sorts only that list, retains
exact comparisons for all scheduler values/timesteps/sigmas, and additionally
checks effective conditioning hashes. No experiment output or model run changed.

Exact checkpoints remain SD2 `sd2-community/stable-diffusion-2-base`, SANA
`Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers`, FLUX
`ModelsLab/flux.1-dev`, SD3.5 `stabilityai/stable-diffusion-3.5-medium` at revision
`b940f670f0eda2d07fbb75229e779da1ad11eb80`, and PixelDiT `pixeldit_t2i_v1.pth`
with official implementation commit `41f73006ae532b0b41fee72b181dc22891a5a01a`
and its existing 1024px stage-3 configuration. No suggested alternative model ID
was substituted. Steps remain 40/20/20/30/50 in the listed order; actual prepared
schedules are saved, and the four latent schedules are checked against K before
generation.


### L/M completed results and interpretation

All ten model jobs completed with Slurm exit code 0:0 on NVIDIA A100-PCIE-40GB.
They performed **28,480 guided predictions** in total, exactly one per camera
per timestep, with no additional denoiser evaluations. The maximum current-state
reconstruction error was **1.430511474609375e-6** (SD2); every other backend was
at most 4.76837158203125e-7. All L/M pairs have identical initial-state and
effective-conditioning hashes and **byte-identical final PNGs**.

Runtime below is the measured pipeline run (including its diagnostics and final
fusion), excluding model loading/conditioning/geometry preparation. Memory is
peak CUDA allocated / reserved GiB during that run. Original K used A40, while
L/M use A100: these timings are descriptive, not a controlled speed benchmark.

| Backend | K original | L SphereDiff-style | M dense >=5 |
|---|---|---|---|
| SD2 | Recognizable ruins but disconnected regions and camera seams; 29.0 s | [Success](../../outputs/vae-residual-controls/20260915-dense-lm/L/sd2/final_result.png): Washed-out gray/olive field; faint ruin fragments; no connected architecture; hard seams muted by blur; 279.6 s; 2.46/2.93 GiB | [Success](../../outputs/vae-residual-controls/20260915-dense-lm/M/sd2/final_result.png): Washed-out gray/olive field; faint ruin fragments; no connected architecture; hard seams muted by blur; 267.3 s; 2.46/2.92 GiB |
| SANA | Detailed, colorful temples; exposure/camera seams and weak connectivity; 74.1 s | [Success](../../outputs/vae-residual-controls/20260915-dense-lm/L/sana/final_result.png): Blurred green/gray landscape bands; faint temple silhouettes; poor connectivity; hard seams muted; 616.6 s; 5.79/7.87 GiB | [Success](../../outputs/vae-residual-controls/20260915-dense-lm/M/sana/final_result.png): Blurred green/gray landscape bands; faint temple silhouettes; poor connectivity; hard seams muted; 609.4 s; 5.76/8.43 GiB |
| FLUX.1-dev | Strong local temple detail; incompatible views and seams; 177.6 s | [Success](../../outputs/vae-residual-controls/20260915-dense-lm/L/flux/final_result.png): Severely blurred gray/green landscape; vague temple silhouettes; local detail lost; hard seams muted; 1540.6 s; 25.00/27.08 GiB | [Success](../../outputs/vae-residual-controls/20260915-dense-lm/M/flux/final_result.png): Severely blurred gray/green landscape; vague temple silhouettes; local detail lost; hard seams muted; 1528.2 s; 24.95/27.04 GiB |
| SD3.5 | Recognizable ruins with blur and camera boundaries; 193.8 s | [Success](../../outputs/vae-residual-controls/20260915-dense-lm/L/sd35/final_result.png): Near-flat gray/olive landscape bands; almost no recognizable architecture; hard seams muted by washout; 1838.9 s; 7.48/9.40 GiB | [Success](../../outputs/vae-residual-controls/20260915-dense-lm/M/sd35/final_result.png): Near-flat gray/olive landscape bands; almost no recognizable architecture; hard seams muted by washout; 1840.4 s; 7.42/9.29 GiB |
| PixelDiT | Unavailable; no K rerun | [Success](../../outputs/vae-residual-controls/20260915-dense-lm/L/pixeldit/final_result.png): Heavily blurred olive landscape and vague ruins; poor connectivity; hard seams muted; 769.9 s; 4.17/4.63 GiB | [Success](../../outputs/vae-residual-controls/20260915-dense-lm/M/pixeldit/final_result.png): Heavily blurred olive landscape and vague ruins; poor connectivity; hard seams muted; 763.7 s; 4.14/4.62 GiB |

K peak allocated/reserved memory was SD2 2.48/3.01, SANA 5.82/7.82,
FLUX 25.14/27.53 and SD3.5 7.63/9.79 GiB. L/M memory remains bounded near
these model-dependent levels despite 14.83 times as many camera predictions.
Temporary host copies grow with view count; this is a GPU-memory bound, not a
claim that host storage is independent of the number of cameras.

The single [K | L | M contact sheet](../../outputs/vae-residual-controls/20260915-dense-lm/K-L-M.png)
shows the full uncropped panoramas in the requested model order. PixelDiT's K
cell is explicitly unavailable. Individual run links above point to each final
image; its adjacent `metadata.json` contains the exact settings and metrics.

**Dense coverage did not improve image quality under this recipe.** Relative to
K's six-view results, the four comparable latent backends lose recognizable
structure and local detail. Smoother camera boundaries mainly reflect blur and
contrast collapse; they are not evidence of improved semantic connectivity.
L/M ERP wrap-gradient ratios are SD2 1.069, SANA 0.912, FLUX 0.973, SD3.5 0.995
and PixelDiT 0.993. These ratios measure the horizontal wrap only, and values
near one do not establish a coherent panorama. Original K also often had wrap
ratios near one while showing other camera seams.

**The effect of meaningful directional prompts is not identifiable here.** L
uses verified directional slot selection and M uses global slot 8, but the
required original prompt file repeats exactly the same text on all five lines.
M's minimum-five search also selected exactly L's camera geometry. The resulting
identical conditioning and images are therefore expected; they do not establish
that distinct SphereDiff directional text helps, fails, or matters less than
redundancy. No prompt substitution was made after seeing these outputs.

All full-resolution pixels have at least seven contributors, so uncovered or
weakly covered pixels cannot explain these dense results. The common blur is
consistent with strong repeated averaging/resampling of independent local
predictions and insufficient cross-view semantic trajectory agreement. This is
an interpretation, not a separate causal ablation of averaging versus semantic
conflict. PixelDiT exhibits the same broad failure with identity encode/decode,
so a VAE round trip is not a necessary cause and a VAE-only explanation is
insufficient. The current-state reconstruction checks and matching native
schedules argue against a transition bookkeeping error.

K-to-L/M also changes FOV from 100 to 80 degrees, as explicitly requested, so
this is not a pure view-count ablation. These two geometries do not establish a
diminishing-returns curve or a globally optimal camera count. The present result
supports neither proposed source of the next quality gain: dense coverage at
this setting is unsuccessful, and distinct directional conditioning remains
untested. No time travel, repeated refinement, or other follow-up method was
implemented.

## Experiment N — SphereDiff Views + Detail-Preserving RGB Average (September 16)

## Experiment O — SphereDiff Views + LookingGlass-Style Laplacian-Pyramid Warp/Blend + DPA

These additive controls start from branch `no_sphere`, clean HEAD
`c12873f3582377cd01aabd35669edee7aada3992`. Existing A–M results and all five L
configs are preserved. Both descendants use L's geometry JSON, 89 fixed cameras
at 80 degrees, camera order, directional prompt slots, local native initialization,
checkpoint/precision/VAE settings, guidance, seed, resolutions and prepared
scheduler. The five prompt lines remain identical, as in L; this task tests
spatial consensus methods, not new directional text.

### Controlled spatial differences

| Setting | L | N | O |
|---|---|---|---|
| Warp | standard | standard | pyramid coefficients in both directions |
| Fusion | ordinary average | existing DPA | existing DPA at every pyramid level |
| Weight mode | uniform | uniform | uniform |
| DPA alpha / power / epsilon | inactive | 1 / 1 / 1e-6 | 1 / 1 / 1e-6 |
| Pyramid levels | inactive | inactive | 5 (four detail bands + coarse base) |
| Project Jacobian band-confidence heuristic | inactive | inactive | disabled (`lod_mode=none`) |
| Transition | current x_t | unchanged | unchanged |
| Residual correction | none | none | none |

The exact resolved L→N diff is `dense_consensus.experiment: L→N` and
`fusion.mode: average→detail_preserving_average`. The N→O diff is experiment
label, `warp.mode: standard→lpw`, `warp.lpw.levels: 4→5`, and
`warp.lpw.lod_mode: jacobian→none` (the last two settings are inactive in N).
O additionally enables periodic ERP pyramid reconstruction as part of its
spatial operator. No denoising or conditioning setting changes. The runner
compares each resolved L config with its actual saved metadata and rejects any
other config difference, changed prompt-file bytes, camera digest, prepared
schedule, effective conditioning, model source/revision or initial-state digest.
Per-camera prompt-index arrays are checked against L's fixed cameras and anchors.
Compact diffs are stored in the shared validation record and each run metadata.

For raw RGB values (N) or signed pyramid coefficients (O), the existing
`RGBFusionAccumulator` computes, per color channel,

```
u_i = valid_i * spatial_weight_i
mean = sum(u_i * value_i) / max(sum(u_i), epsilon)
d_i = u_i * (abs(value_i) + epsilon)**power
detail = sum(d_i * value_i) / max(sum(d_i), epsilon)
fused = mean + alpha * (detail - mean)
```

These are the existing tested parameters, chosen before seeing results. RGB
magnitude weighting is not an explicit texture detector; signed coefficient
magnitude in O has a different frequency interpretation. No per-model tuning
or extra model pass is introduced.

### LookingGlass reference verification and ERP adaptation

Inspected the public third-party
[LatentGenerativeAnamorphoses implementation at e5fbb217](https://github.com/Cedric-Perauer/LatentGenerativeAnamorphoses/tree/e5fbb217841c78a7c3e72d09166d28626356fcf6),
not an author-released byte-identical LookingGlass implementation:
[`lwp()`](https://github.com/Cedric-Perauer/LatentGenerativeAnamorphoses/blob/e5fbb217841c78a7c3e72d09166d28626356fcf6/diffusers/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L805)
and [`lod_new.py`](https://github.com/Cedric-Perauer/LatentGenerativeAnamorphoses/blob/e5fbb217841c78a7c3e72d09166d28626356fcf6/diffusers/src/diffusers/pipelines/stable_diffusion_3/lod_new.py#L433).
The public LWP entry uses five levels and combined arithmetic/magnitude-weighted
pooling. Its Gaussian reduction is 2×2 block averaging; reconstruction is
bilinear. Its optional mask branch bypasses pyramid pooling. Its simple forward
warp reconstructs before resampling, while its inverse path uses autograd and
3D LOD sampling. Those shortcuts are not the required O operation.

O reuses the repo's binomial-filter Gaussian/Laplacian utilities, five-level
streaming accumulator, explicit validity masks and normalized masked
reconstruction. It projects coefficients at corresponding camera/ERP level
resolutions and pools them before reconstructing; the return direction also
warps pyramid coefficients. ERP expansion is periodic horizontally with pole
padding, enabled only for O so historical defaults remain unchanged. This is an
adaptation for many perspective views, not byte-for-byte reference equivalence.
The reference inverse LOD is different from the repo's band-confidence heuristic;
that optional heuristic is disabled rather than introduced as another variable.
Reference alpha defaults (0.5 at `lwp`, 0.25 at `blend_pyramids`) are not adopted:
N/O deliberately retain the same repo DPA alpha=1 for causal pairing.

### Shared trajectory, streaming, diagnostics and snapshots

There is one dense Jacobi trajectory. Only local native noisy states persist;
ERP RGB is transient. Each view is predicted once, decoded, accumulated, then
synchronized and encoded (identity for PixelDiT). The unchanged
`interpolate_from_current_state` uses the frozen x_t and actual scheduler
coefficients, never the old endpoint. All next states commit together.

O retains running per-level numerators/denominators and one projected view at a
time. Temporary local tensors live on CPU; the geometry cache has bounded device
entries and CPU overflow. All fixed projection maps, including O's smaller
levels, are prepared once. The diagnostic standard projection in O is only for
comparable O(N) view-to-consensus error and never enters the spatial consensus.
Timing records distinguish model, decode, view-pyramid construction,
view→ERP, per-level fusion, ERP reconstruction, ERP→view and encode. ERP→view
timing includes construction/reconstruction of its return pyramid.

Both experiments save already-computed predicted-clean consensus at completed
steps `ceil(percent * steps / 100)`, for 10% through 90%. Integer arithmetic
avoids floating-point ceil errors; duplicate steps on short schedules are grouped.
No persistent x_next decoding or extra model prediction is used for snapshots.
The callback receives a detached copy. The final image instead decodes terminal
local states and uses the experiment's same spatial fusion method.

| Backend | Steps | Snapshot completed steps (10% through 90%) |
|---|---:|---|
| SD2 | 30 | 3, 6, 9, 12, 15, 18, 21, 24, 27 |
| SANA | 20 | 2, 4, 6, 8, 10, 12, 14, 16, 18 |
| FLUX.1-dev | 20 | 2, 4, 6, 8, 10, 12, 14, 16, 18 |
| SD3.5 | 40 | 4, 8, 12, 16, 20, 24, 28, 32, 36 |
| PixelDiT | 50 | 5, 10, 15, 20, 25, 30, 35, 40, 45 |

Each run saves `consensus_010.png` through `consensus_090.png`,
`final_result.png`, and one `metadata.json`. Snapshot percentages, completed
steps, actual timesteps and current/next alpha/sigma are recorded in that JSON.
No per-camera images, pyramid-level images, tensors, or per-step JSONs are saved.
The only planned extra image is one FLUX N/O progression sheet at
10/30/50/70/90% and final. L has no comparable snapshots and is not rerun.

### N/O validation and submitted jobs

CPU job **19762588** passed compileall, whitespace checks, all **8 focused N/O
tests**, and the full **198-test regression suite**. Its final gate also verified
the ten resolved configs against the actual completed L metadata and prompt-file
hash. The focused tests cover camera/prompt pairing, identical initialization,
strict settings guards, pyramid-level projections and DPA accumulators,
periodic reconstruction and wrap, constant-image fusion, poisoned old endpoints,
Jacobi order invariance, unchanged model/encode/decode counts, and bit-identical
final tensors when snapshots are enabled. Existing regression tests cover masked
reconstruction and the mathematical difference between per-level DPA and DPA
after reconstruction.

| Backend | N job | O job | Guided predictions per run |
|---|---:|---:|---:|
| SD3.5 | 19762606 | 19762611 | 3560 |
| FLUX.1-dev | 19762607 | 19762612 | 1780 |
| SANA | 19762608 | 19762613 | 1780 |
| SD2 | 19762609 | 19762614 | 2670 |
| PixelDiT | 19762610 | 19762615 | 4450 |

N was submitted first in the requested model order, followed by O in that order.
All use A100 resources, matching L's GPU class. The dependent final report/audit
job is **19762616**. The shared validation record retains job IDs, config diffs,
and hashes of all 22 L/M run and geometry artifacts; the final audit checks these
hashes again. No duplicate jobs or existing output directories were present at
submission. Run outputs are under
`outputs/vae-residual-controls/20260916-dense-no/{N,O}/{backend}`.

### Additional pyramid artifact checks

Early O snapshots showed high-frequency ripples and curved outlines, motivating
an additional CPU-only mask audit (job **19762637**, exit 0:0). It used the exact
L-derived cameras at all five actual O level resolutions. Coverage was 100% at
every level: minimum 7 contributors everywhere except the coarsest SD2 grid,
which had minimum 8 (32×64 ERP). A partial-mask constant reconstruction crossing
the ERP wrap had maximum error **0.0** with O's periodic reconstruction enabled.
This rules out missing coverage and simple invalid-zero darkening in these
checks; it does not rule out scale-dependent resampling or discontinuities in
contributing coefficients. No parameters, model calls, or experiment outputs
were changed in response to the images. The audit saved only Slurm text logs.

### Completed L/N/O image comparison

All ten N/O GPU jobs completed with Slurm exit **0:0**. These are matched
single-prompt, seed-0 observations; the five directional prompt slots contain
identical text. Neither spatial change rescues the dense pipeline. N mainly
changes contrast and silhouettes. O preserves more local architectural edges,
especially during denoising, but also preserves or introduces conspicuous
frequency artifacts and does not produce a coherent, detailed final panorama.

Each linked image below is the terminal-state output, not a milestone clean
prediction. Coherence and continuity remain poor in every cell; more visible
edges alone are not evidence of a better reconstructed scene.

| Backend | L Avg | N DPA | O LPW+DPA |
|---|---|---|---|
| SD2 | [L](../outputs/vae-residual-controls/20260915-dense-lm/L/sd2/final_result.png): gray/olive blur with tiny isolated ruins; little fine detail or continuity. | [N](../outputs/vae-residual-controls/20260916-dense-no/N/sd2/final_result.png): brighter orange patches, but still heavy blur and disconnected fragments; early curved boundaries fade. | [O](../outputs/vae-residual-controls/20260916-dense-no/O/sd2/final_result.png): a few faint column edges, subdued bright blotches; washout and disconnected structure remain. |
| SANA | [L](../outputs/vae-residual-controls/20260915-dense-lm/L/sana/final_result.png): dark green ghosted temples, curved outlines and weak columns; poor continuity. | [N](../outputs/vae-residual-controls/20260916-dense-no/N/sana/final_result.png): very dark blurred forms and vague temples; no convincing fine-detail gain. | [O](../outputs/vae-residual-controls/20260916-dense-no/O/sana/final_result.png): clearer local facades, columns and steps, but faint, ghosted and disconnected; curved outlines remain. |
| FLUX.1-dev | [L](../outputs/vae-residual-controls/20260915-dense-lm/L/flux/final_result.png): nearly featureless gray/olive skyline, heavily smoothed. | [N](../outputs/vae-residual-controls/20260916-dense-no/N/flux/final_result.png): stronger temple silhouettes; surface detail still blurred and structures disconnected. | [O](../outputs/vae-residual-controls/20260916-dense-no/O/flux/final_result.png): faint roof/column fragments, but stepped horizontal banding and washout; much less structure than its own 70–90% clean snapshots. |
| SD3.5 | [L](../outputs/vae-residual-controls/20260915-dense-lm/L/sd35/final_result.png): almost flat landscape-colored fields with very faint ruins. | [N](../outputs/vae-residual-controls/20260916-dense-no/N/sd35/final_result.png): modest contrast change, still almost featureless; no useful structural continuity. | [O](../outputs/vae-residual-controls/20260916-dense-no/O/sd35/final_result.png): faint columns, walls and steps survive, but with horizontal streaking, curved outlines and severe washout; intermediate architecture is much clearer. |
| PixelDiT | [L](../outputs/vae-residual-controls/20260915-dense-lm/L/pixeldit/final_result.png): soft ruin-like masses and isolated temple silhouettes; little detail. | [N](../outputs/vae-residual-controls/20260916-dense-no/N/pixeldit/final_result.png): stronger silhouettes and contrast, but very soft edges and disconnected repeated masses. | [O](../outputs/vae-residual-controls/20260916-dense-no/O/pixeldit/final_result.png): faint column edges accompanied by severe horizontal streaking/blocky bands; no convincing overall gain over N. |

### Runtime and device memory

All L/N/O runs used NVIDIA A100-PCIE-40GB. Each cell is **pipeline seconds;
peak allocated / reserved GiB**. Runtime excludes model loading and projection
precomputation and includes diagnostics, final fusion, and N/O snapshot saving.
Consequently L→N timing is not a pure fusion microbenchmark. Small differences
are single-run measurements, not statistically established speedups.

| Backend | L Avg | N DPA | O LPW+DPA |
|---|---:|---:|---:|
| SD2 | 279.6; 2.461 / 2.928 | 296.7; 2.471 / 2.934 | 303.9; 2.457 / 2.875 |
| SANA | 616.6; 5.786 / 7.869 | 626.8; 5.836 / 7.916 | 650.5; 5.857 / 7.680 |
| FLUX.1-dev | 1540.6; 25.005 / 27.084 | 1545.1; 25.055 / 27.127 | 1584.5; 25.010 / 26.896 |
| SD3.5 | 1838.9; 7.483 / 9.402 | 1835.0; 7.535 / 9.473 | 1914.1; 7.484 / 9.215 |
| PixelDiT | 769.9; 4.166 / 4.631 | 760.0; 4.217 / 4.678 | 843.9; 4.168 / 4.713 |

O adds approximately 2.5–11.0% pipeline time over N in these runs. Streaming
keeps peak device allocation close to N without reducing scientific resolution.
Per-stage timings and aggregate consensus/update diagnostics are in each run's
single metadata JSON.

### Intermediate behavior and final-output distinction

The single [FLUX progression sheet](../outputs/vae-residual-controls/20260916-dense-no/flux-progression.png)
shows N/O at 10/30/50/70/90% and terminal final. L has no comparable snapshots.

| Milestone | N, standard warp + RGB DPA | O, pyramid warp + per-level DPA |
|---|---|---|
| 10% | Large curved camera-footprint tiles and overlap boundaries already visible in FLUX and SD2; little coherent structure. | Broad tiles can be smoother, but strong ripples/grain and sharp curved outlines appear. |
| 30% | Overlap boundaries persist while vague temple masses emerge. | FLUX begins to show towers, with strong radial/moire-like texture covering the panorama. |
| 50% | Soft silhouettes dominate; SANA shows ghosted temples and broad arcs. | FLUX towers/columns and SANA steps become clearer; artifacts and overlapping incompatible fragments remain. |
| 70% | FLUX has a soft skyline; SD3.5 is still almost flat. | FLUX and SD3.5 show substantially clearer architecture, but disconnected/duplicated structures and floating fragments remain. |
| 90% | Broad soft forms remain without a strong recovery of surface texture. | FLUX retains visible columns and roof edges; most early ripples attenuate, but ghosting and gray/olive washout persist. |
| Final | Contrast/silhouette changes survive, but heavy blur remains across the five backends. | Local edges remain in some backends; FLUX/SD3.5 lose much of their intermediate clarity, and horizontal bands/streaks are conspicuous in FLUX, SD3.5 and PixelDiT. |

Seams are present **by the first saved 10% milestone**; these snapshots cannot
locate their onset within the first 10%. The final image has different semantics:
it decodes terminal local states and fuses them. The 90% image is an already
computed predicted-clean consensus. Their difference includes the remaining
steps and terminal encode/decode/fusion effects. There is no saved 100% clean
consensus control, so it would be incorrect to attribute all late detail loss
solely to the final fusion operation.

### Scientific interpretation

**L→N: DPA alone does not reliably preserve useful texture.** FLUX and PixelDiT
gain more conspicuous silhouettes/contrast, while SANA and SD3.5 remain badly
smoothed. SD2 gains bright patches rather than connected architecture. Since N
weights RGB magnitude, stronger bright/dark observations can gain influence
without recovering consistent texture. It does not solve seams; early overlap
boundaries and ghosting remain. L has no milestone images, so these runs do not
establish whether N worsens the early seams relative to L.

**N→O: pyramid-domain DPA preserves some local edges, but not reliably better
final panoramas.** SANA and SD3.5 retain more architectural fragments, and FLUX
is visibly clearer mid-trajectory. This is evidence that the spatial mechanism
affects detail retention. It is not evidence that projection artifacts are
uniformly reduced: early ripples and late banding are prominent, structural
continuity remains poor, and FLUX/PixelDiT have no convincing overall terminal
quality improvement over N. A high-frequency statistic also counts these
artifact edges, so it must not be read as a perceptual-quality score.

The current-x_t invariant remains numerically stable: maximum observed error is
**1.430511474609375e-6** for SD2 and **4.76837158203125e-7** for each other backend,
for both N and O. Thus the sharper fragments are not accompanied by a detected
failure of the current-state transition. Coverage is complete at every tested
pyramid level, normalized partial-mask reconstruction preserves constants, and
the project-specific LOD heuristic is disabled. These checks argue against
coverage holes, simple invalid-zero darkening, or that heuristic as explanations
for the observed O artifacts; they do not prove every frequency/resampling
choice is optimal.

The visible final failure is dominated by **over-smoothing and loss of coherent
structure**, with cross-view semantic disagreement a plausible contributor and
additional projection/pyramid artifacts visible in O. This controlled spatial
ablation cannot causally rank semantic disagreement against repeated resampling
or encoding losses. PixelDiT shows similar failures without a VAE, so a VAE is
not a necessary cause. The results do not justify claiming that a VAE-only fix,
DPA alone, or this particular LPW adaptation resolves the problem. No parameters
were tuned and no extra denoising, residual correction, time travel or cleanup
was introduced after viewing outputs. Conclusions remain limited to this one
prompt and seed, and to the documented adaptation rather than full LookingGlass.

### Final artifact audit and storage

CPU report job **19762616** completed with exit **0:0** and all checks passed.
The [shared validation record](../outputs/vae-residual-controls/20260916-dense-no/validation.json)
records the successful audit, compact config diffs, job IDs, baseline hashes,
mask audit and storage totals. The full numerical report is in
[`dense-no-report.19762616.out`](../logs/dense-no-report.19762616.out).

All ten actual runs match L's camera hash, fixed camera order/count (89), 80°
FOV, geometry source, resolutions, prompt hash/assignment, effective conditioning,
seed, initial-local-state digest, model source/revision, guidance and prepared
scheduler. Resolved-config guards also hold precision and VAE options fixed.
N/O per-camera prompt-index arrays match. Every run has exactly one guided
prediction per camera per step: **28,480 total**, with **zero extra denoiser
calls**. Strict Jacobi, fixed cameras and conditioning, current-state transition,
and disabled residual/fixed-noise/spherical-latent/ERP-latent flags all pass.
All 22 saved L/M run and geometry artifacts retain their pre-run SHA-256 hashes.

| Output set | PNG images | Metadata JSONs | Files | Bytes | MiB |
|---|---:|---:|---:|---:|---:|
| N, five models | 50 | 5 | 55 | 65,394,262 | 62.365 |
| O, five models | 50 | 5 | 55 | 103,498,452 | 98.704 |
| Entire N/O root, including shared validation and one progression PNG | 101 | 11 JSONs total | 112 | 169,702,221 | 161.841 |

Every model directory contains exactly nine milestone PNGs, one final PNG and
one metadata JSON, at the expected native ERP resolution. Snapshot percentages,
completed steps and scheduler timesteps pass the mapping audit. There are no
per-view images, pyramid images, tensor dumps or per-step JSONs. Slurm text logs
live outside these output sizes. No new temporary debugging images were needed.

As a limited numerical check, terminal horizontal-wrap RGB differences are
higher in O than N for all five backends (O range 0.000319–0.001510 versus N
0.000140–0.000665 on a 0–1 scale). This is not a perceptual seam score: natural
image gradients contribute, and curved internal camera boundaries are not
measured by the left/right edge difference. The increase does not support a
claim of uniformly improved wrap continuity. The report also records simple
8-bit luminance high-frequency measurements; these increase with O but count
banding and artifact texture as well as meaningful detail.

Final repository state remains branch `no_sphere`, HEAD
`c12873f3582377cd01aabd35669edee7aada3992`, with the additive N/O changes
uncommitted. Earlier experiment data and existing work are preserved. Final
whitespace checks pass; no additional GPU runs or parameter changes were made.


## Experiments P–T — dense consensus causal diagnostics (September 16 continuation)

Work began on clean `no_sphere`, HEAD `8bf6d13257a32db5ee279ca973cde2dadc9beb25`,
with an empty Slurm queue. Historical A–O outputs remain untouched. Durable
execution state and historical L–O file hashes are in
`outputs/vae-residual-controls/20260916-diagnostics-pt/execution.json`.

P observes L at ceil-mapped 10/50/90/100% milestones in four physical camera
directions and measures terminal reassembly separately. Q evaluates the frozen
L/N/O spatial operators on continuous spherical functions and one sampled
generated panorama, at both ERP resolutions. R compares exact planar native
MultiDiffusion with clean/current-state updates from shared and independent
pixel Gaussian initializations. S changes only L's fusion to normalized
SphereDiff center weights at temperature 0.1. T independently doubles only
ERP resolution to 2048×4096, retaining 1024² views.

L/M metadata confirms equal camera, conditioning and initialization hashes;
their final PNGs are identical. They do not provide independent evidence about
directional text. K→L changes both count and FOV. Reconstruction identities
remain algebraic checks, not evidence of image quality. O remains the historical
ERP pyramid adaptation, not an exact LookingGlass reproduction.

FLUX uses the actual L checkpoint `ModelsLab/flux.1-dev`; L stored no resolved
revision. The present cached main resolves to
`fa45a9eb6808ba8fdfc7cc2756f7f1a16e0921f4`; historical revision identity cannot
be established solely from that current cache. PixelDiT uses the actual saved
checkpoint blob `625fd174d6348af3ad1123b281ab817643efa5dc447c2344eeb222e88b4317db`
and official code `41f73006ae532b0b41fee72b181dc22891a5a01a`.

The existing pinned SphereDiff source at
`/tmp/diffpano_spherediff_official/spherical_functions.py` has default temperature
0.1 (line 134) and `exp(-norm(normalized_coordinate)/temperature)` (line 243).
The project retains its own pixel-center/align-corners-false projector. Ordinary
`average` explicitly ignores supplied weights; S uses existing `weighted_average`
with positive-denominator normalization, preserving average/DPA semantics.

Initial estimate per dense run: FLUX 1,780 guided predictions, ~26 min, 25 GiB
allocated; PixelDiT 4,450, ~13 min, 4.2 GiB. P adds at most 16 diagnostic FLUX
decodes and no denoiser calls. T uses a two-entry device geometry cache with
host fallback: approximately 9–12 GiB host projection maps plus states and
temporaries, under a 64 GiB host allocation; device geometry is bounded and
local model tensor resolution is unchanged. R has three patches and 50 steps,
450 guided predictions across all methods. Q has zero model/VAE calls.

Initial status at implementation: no P–T model results had been claimed. Completed results and final status are recorded below.
Validation 19769134 found a camera-selection test failure; selecting the nearest
other equatorial camera fixes the intended adjacent pair. Validation 19769139
includes the corrected test and new R mock oracle/configuration guards.

Validation 19769139 stopped at compileall on an R launcher syntax error,
corrected before execution. Validation 19769143 passed the focused controls and
ran 205 full-suite tests; its only three errors were config tests reading G
sidecars deleted by the prior user-requested cleanup. These tests now compare
the same config fields against retained generation metadata; no assertion was
skipped and no historical result was restored or modified. Validation 19769155
is the replacement full gate. Model launchers also require source hashes to
match that passed gate.

### Q — completed coherent-input spatial controls

Job **19769161** completed on a T4: 890.55 s measured execution, 17m25s Slurm wall time, 13.67 GiB maximum host RSS, zero denoiser/VAE calls. Full-resolution coverage is 100% for all six operator/resolution combinations.

| Operator | ERP | Synthetic RMSE, cycles 1 / 5 / 20 | Band retention at cycle 20, k=8 / 32 / 96 | Sampled-image RMSE at cycle 20 |
|---|---|---|---|---|
| L | 1024x2048 | 0.000187 / 0.000882 / 0.003030 | 1.0000 / 0.9994 / 0.9942 | 0.018306 |
| N | 1024x2048 | 0.000214 / 0.000922 / 0.003006 | 1.0000 / 0.9994 / 0.9946 | 0.018123 |
| O | 1024x2048 | 0.046654 / 0.091543 / 0.108065 | 0.7958 / 0.1716 / 0.0598 | 0.178804 |
| L | 2048x4096 | 0.002045 / 0.008674 / 0.026402 | 0.9988 / 0.9746 / 0.8032 | 0.040734 |
| N | 2048x4096 | 0.002414 / 0.008936 / 0.023401 | 0.9992 / 0.9827 / 0.8530 | 0.040059 |
| O | 2048x4096 | 0.014918 / 0.055023 / 0.103117 | 0.7968 / 0.2540 / 0.0991 | 0.162449 |

Measurements use unclamped floating RGB and cosine-area-weighted global errors. Each grating is fitted jointly with its smooth, marker and DC nuisance components. The k=96 maximum phase increment is approximately 0.93 rad per baseline ERP pixel and 0.50 rad per perspective pixel, both below Nyquist (pi); the doubled ERP halves the former. Higher frequencies near the sampling limit remain untested. Markers report peak shifts within predefined wrap/north caps relative to the sampled analytic reference, with finite-pixel precision.

Maximum constant error: 1.19e-07; maximum whole-observation-set duplication error: 3.58e-07. These invariants show no observed count-normalization failure.
O nevertheless damages directly evaluated consistent views before repeated cycles: at baseline resolution, direct assembly RMSE is 0.052997 and k=32/96 retained amplitudes are 0.6143/0.5556. L direct assembly RMSE is 0.000695 with k=96 retention 0.9928. The immediate loss is already in pyramid assembly/scale transport, not solely incompatible diffusion predictions or a final VAE pass. Passing normalization does not prove every pyramid implementation detail correct; no historical O correction or retuning was made.

Visual inspection of the synthetic overview finds O suppresses fine rings and creates uneven coarse shapes/ripples. L/N mostly preserve the tested coherent patterns. The larger ERP does not improve repeated L/N cycles at fixed perspective sampling; repeated cycles are a stress test, not a denoising trajectory. T remains an independent model test, not an assumed remedy.

No suitable photographic ERP asset was found. The sampled-reference test uses the existing sharp SANA K panorama, with its visible stitching explicitly inherited; it is not geometric ground truth. Both resolutions use this one source image (the larger sampled reference is upsampled), whereas synthetic references are evaluated directly from directions at each resolution.

Artifacts: `Q/metrics.json`, `Q/synthetic.png`, `Q/sampled.png` under the shared P–T output root. They include matched full/central single-camera roundtrips, equatorial/polar summaries, wrap errors, and marker shifts.

The sampled-image overview was also visually inspected: O progressively blurs
stone/vegetation texture and broadens stitch boundaries; L/N retain much more
of the source detail. These are losses added to the same already-stitched
reference, not an assessment of its geometric correctness.

### P/R execution pairing update

A100 scheduling estimates were long, so pending P jobs 19769172/19769173 were
updated in place to available A40 resources. Both stopped at the exact
effective-conditioning hash guard before **any denoiser calls**. No dense
result was produced and the guard was not relaxed. Replacement P jobs
19769228 (FLUX) and 19769229 (PixelDiT) request historical A100 hardware with
offline cached checkpoints. R job 19769230 uses A40; its three methods share
one loaded model and identical conditioning within the controlled comparison.
The saved historical native config/checkpoint/schedule remain checked, but R
is not claimed bit-identical to an older run on different hardware.

### R — paired oracle review before independent initialization

The deterministic mock oracle passed. Real native/shared trajectories (job
19769230, A40) pass all 50 **same-input** first-order checks, maximum absolute
error 1.6913e-6, under the predeclared `32 * float32 epsilon * max(1, |next|)`
bound. Shared next-state overlaps agree bit-for-bit at every step. Initial
model inputs and clean predictions are identical; differing FP32 evaluation
orders first introduce a maximum 1.69e-6 next-state difference. Subsequent
bf16 model evaluations amplify it (step 2 clean-prediction MAE 0.01881).

The freely evolved final outputs are **not bit-identical**: raw RGB MAE
0.0057924, RMS 0.0076167, maximum 0.21482. RMS corresponds to about 0.97
8-bit display code value over the [-1,1] range, but localized differences can
be much larger. Visual inspection shows the same detailed ruins/foliage and
composition in both images. This supports practical image agreement and the
mathematical oracle, not a claim that a strict pixelwise trajectory oracle
passed. No tolerance was loosened. The per-step evidence and this qualification
were recorded before enabling the independent branch.

### R — completed three-way initialization control

| Method | Raw RGB std | HF sigma=1 RMS / std | Guided predictions | Visual observation |
|---|---:|---:|---:|---|
| native | 0.528931 | 0.170153 | 150 | Detailed stonework and vegetation, recognizable temples |
| shared | 0.528725 | 0.170168 | 150 | Same composition/detail as native; small numerical differences |
| independent | 0.509346 | 0.125311 | 150 | Strong blur/ghosting in the overlapping central region; outer regions remain sharper |

Independent source overlap differences shrink from a maximum 3.5760 after the first update to zero at terminal sigma=0, yet the final central region is visibly blurred. Zero terminal disagreement is therefore not a quality certificate. The final schedule coefficients are alpha_next=1 and sigma_next=0, verified from the actual solver.

R-native and R-shared start from exactly the same global CUDA FP32 unit Gaussian; local shared states are exact crops. R-independent uses distinct sequential local draws from a reset CUDA generator with the same seed and unit marginal scale; a common seed does not imply identical realized tensors. No clipping, VAE, noise reinitialization, or warp was added.

The matching detailed native/shared images, tiny same-input oracle errors, and substantially degraded independent overlap region implicate initialization correspondence **in this planar pixel-native setting**. The small propagated native/shared numerical difference, one seed, and different independently realized scenes are limitations; this is not proof of the complete ERP mechanism. High-frequency measurements support the visual observation but are not used alone as a quality score.

Job 19769230 completed with exactly 450 guided predictions and 450 transformer forward invocations (CFG branches batched internally), zero VAE calls. Paired execution took 70.43 s for two trajectories; independent execution 34.08 s. Total process time 339.85 s includes loading and the explicit evidence-review wait. Peak allocated GPU memory was 4.243 GiB; process maximum host RSS 10.263 GiB. No algorithmic speedup is inferred from these diagnostic timings.

Exactly five R artifacts: `native.png`, `shared.png`, `independent.png`, `comparison.png`, `metrics.json`.

### P — PixelDiT stage audit completed

Job **19769229** reproduced historical L's terminal PNG byte-for-byte and matched its exact conditioning, camera, initial-state and schedule hashes. The observation-only path used 4,450 guided predictions / 4,450 transformer forwards, no extra denoiser or VAE calls. Runtime 774.71 s includes 3.97 s measured diagnostics; peak allocated/reserved GPU memory 4.108/4.545 GiB, process host maximum 12.374 GiB.

The selected fixed slots are 7 and 8 (equatorial yaw -12/+12 degrees), 59 (pitch +45 degrees, yaw about -16.3636 degrees), and 85 (north pole, yaw about -180 degrees). Exact FP32-derived poses are embedded in metadata. PixelDiT milestones are completed steps 5/25/45/50.

| Step | Local HF1 RMS | Returned HF1 RMS | Returned/local HF1 | Central-crop returned/local HF1 | Returned-vs-local MAE |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.018315 | 0.004169 | 0.228 | 0.245 | 0.172298 |
| 25 | 0.014253 | 0.002213 | 0.155 | 0.166 | 0.026639 |
| 45 | 0.007854 | 0.001325 | 0.169 | 0.175 | 0.009130 |
| 50 | 0.005709 | 0.001059 | 0.186 | 0.193 | 0.005755 |

Numbers average the four selected cameras; they are not an exhaustive all-camera image-quality score. Raw mean/std, range fractions, HF at sigma 1/2/4 and contrast-normalized HF are all retained in metadata. Early local predictions contain more contrast and differing scene fragments, but also noise. By the middle and late stages the local predictions are themselves already degraded; faint columns/edges are nevertheless further suppressed by returned consensus. Thus the evidence is cumulative collapse plus an immediate additional consensus loss, not pristine local images suddenly ruined only at the final step. Similar center-crop suppression argues against a purely peripheral explanation.

The VAE projection is identity. Actual final coefficients are (alpha, sigma, alpha_next, sigma_next) = (0.92459947, 0.07540054, 1, 0). Terminal assembly therefore equals the measured render/reassemble spatial cycle. Last-clean versus terminal raw spherical MAE is 0.00005520, RMS 0.00012440, contrast ratio 0.99999856. HF1 decreases from 0.00116041 to 0.00112371 (3.16% of an already tiny value); same-camera terminal differences remain small. Final assembly is not the dominant cause of this PixelDiT L collapse.

After Q/R and this backend's P review, separate PixelDiT S/T jobs **19769301/19769302** were submitted on A100. FLUX S/T remain unsubmitted until its P review. No shared-noise policy or terminal-output change was added to S/T.

### P — FLUX stage audit completed

Job **19769228** reproduced historical L's terminal PNG byte-for-byte. Conditioning, initial states, cameras and prepared schedule pass exact historical guards; the historical resolved checkpoint revision remains unrecorded. Milestones are steps 2/10/18/20 and use the same four physical cameras as PixelDiT.

| Step | Local HF1 RMS | Returned HF1 RMS | VAE-roundtrip HF1 RMS | Central returned/local HF1 | VAE/returned contrast ratio |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.048295 | 0.010412 | 0.009356 | 0.177 | 0.9527 |
| 10 | 0.002438 | 0.000797 | 0.000780 | 0.290 | 0.9529 |
| 18 | 0.001063 | 0.000483 | 0.000612 | 0.400 | 0.9529 |
| 20 | 0.000749 | 0.000426 | 0.000595 | 0.495 | 0.9549 |

Visual review finds the early local outputs are strongly differing blurry compositions with grid texture, not sharp ground truth. Their returned consensus has overlapping footprint arcs/tiling. By step 10, faint local architecture is still visible but is suppressed further on consensus return. At steps 18/20 the local predictions themselves are almost featureless. Center crops also lose detail. This supports cumulative trajectory degradation plus immediate consensus suppression; it does not establish a uniformly healthy local trajectory that fails only at final fusion.

The VAE roundtrip adds roughly 4–5% contrast reduction at the inspected stages (mean camera MAE 0.01404/0.01042/0.00947/0.00877). It sometimes raises late HF energy without restoring visible architecture; this is why HF is not treated as a quality score. PixelDiT's collapse without a VAE rules out a VAE-only account.

Actual final coefficients are (0.85747063, 0.14252935, 1, 0). Last-clean to terminal assembly spherical MAE is **0.0106008**, RMS 0.0124015, contrast ratio **0.949126**. Global HF1 is 0.000197122 before versus 0.000157241 after (20.23% decrease in an already tiny signal). Final assembly adds measurable contrast/detail loss but acts on an already collapsed image. This is a possible separate output-stage ablation, not the primary explanation or a change included in S/T.

Runtime 1568.21 s; diagnostics 4.96 s, exactly **16 extra VAE decodes, zero extra encodes or denoiser predictions**. Guided predictions and actual transformer forwards both 1,780. Peak allocated/reserved GPU memory 25.051/26.998 GiB; host process maximum 10.202 GiB. Four P artifacts only.

After numeric and visual review of both P backends, Q and R, FLUX S/T jobs **19769470/19769471** began on A100. These are independent single-variable controls, not a combined intervention.

### S/T — PixelDiT completed

S job **19769301** restores recognizable columns, temple facades, foliage and stone forms compared with the blurred P/L baseline. The ERP remains a patchwork of repeated/incompatible structures; improved local detail is not proof of a coherent scene. T job **19769302** remains severely blurred at common display size and does not visually rescue the scene.

Both pass all exact L pairing guards, retaining the identical initial local-state digest and 4,450 guided/actual transformer predictions each. S changes only normalized geometric influence at temperature 0.1. T changes only ERP size to 2048x4096, with 1024-square native views.

| PixelDiT method | Mean full-view std | Mean view HF1 RMS | Mean view HF1/std | Central HF1/std | Runtime s | GPU allocated/reserved GiB | Host max GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| P / exact L replay | 0.151689 | 0.001028 | 0.006781 | 0.008694 | 774.71 | 4.108 / 4.545 | 12.374 |
| S / center weights | 0.178443 | 0.008879 | 0.049685 | 0.048295 | 773.78 | 4.115 / 4.635 | 10.550 |
| T / double ERP | 0.151770 | 0.000705 | 0.004647 | 0.006137 | 936.08 | 4.518 / 5.186 | 16.310 |

These are raw 1024-square renders into the same four 80-degree cameras; no cross-resolution ERP-gradient comparison is used. Larger S HF is interpreted alongside visible architecture, while its coherence defects remain explicit. T's measured angular detail is lower and its runtime about 21% higher than P; diagnostic and hardware timing caveats apply.

S retains all cameras and 100% geometric coverage, 7–19 geometric contributors per ERP pixel (median 12, mean 12.9055). Accumulated weights range 0.117888–3.983120; p01/median/p99 are 0.129723/0.275361/3.329189. No pixels are uncovered or have accumulated weight below 1e-6. Effective contributor count N_eff has min/median/mean/max 1.0469/2.1783/2.5393/6.6288 (p01 1.0770, p99 6.4232). Geometric count and effective influence are distinctly different. Weighting changes influence without removing geometric coverage.

### P — cross-backend stage conclusion

| Backend | Local predicted clean | Returned consensus | VAE conversion | Last clean versus terminal assembly |
|---|---|---|---|---|
| FLUX | Strongly differing early compositions; faint architecture at halfway; almost featureless late | Suppresses remaining mid-stage architecture; early camera-footprint arcs | Additional ~4–5% contrast reduction in sampled roundtrips; late HF increase does not restore structure | MAE 0.01060; contrast -5.09%; HF1 -20.23% from an already tiny value |
| PixelDiT | Differing/noisy early predictions; weak local structure persists mid/late but is already degraded | Strong added suppression, including central crops; returned/local HF1 0.155–0.228 across milestones | Identity; no VAE exists in this control | MAE 0.0000552; contrast effectively unchanged; HF1 -3.16% |

Both terminal PNGs exactly match historical L. The diagnostics therefore reproduce the failure rather than creating a new failure mode. These snapshots locate immediate stage losses within a cumulatively degraded trajectory; they do not estimate the counterfactual quality of isolated local denoising.

### Q — regional coherent-input check

Synthetic cycle-20 regional errors below are unweighted within each specified region; the earlier global RMSE uses spherical area weights. Equatorial rows cover the middle third of the ERP; north/south caps cover the first/last sixth. Wrap MAE uses the first/last two columns.

| Operator | ERP | Equatorial RMSE | North RMSE | South RMSE | Wrap MAE |
|---|---|---:|---:|---:|---:|
| L | 1024x2048 | 0.003099 | 0.001088 | 0.001087 | 0.001643 |
| N | 1024x2048 | 0.003070 | 0.001076 | 0.001074 | 0.001653 |
| O | 1024x2048 | 0.147541 | 0.030670 | 0.005981 | 0.036351 |
| L | 2048x4096 | 0.036318 | 0.001716 | 0.001292 | 0.007309 |
| N | 2048x4096 | 0.032092 | 0.001716 | 0.001294 | 0.006551 |
| O | 2048x4096 | 0.140723 | 0.013493 | 0.005877 | 0.016906 |

At baseline resolution, O shifts the selected wrap/north composite-signal peaks by about 2.12/2.47 degrees after 20 cycles; L/N peaks remain effectively at the same sampled locations. These peak tests compare against the sampled composite reference in predefined caps, not a subpixel fitted isolated marker. Tiny reported angular differences near 0.02 degrees are below the practical pixel/FP32 angular precision and should not be overinterpreted.

Q peak GPU allocation was not recorded; host MaxRSS and zero model/VAE calls are verified.

### Implementation scope and regression gate

The working branch remains `no_sphere` at starting HEAD `8bf6d13257a32db5ee279ca973cde2dadc9beb25`; no commit, reset, checkout or historical-result rewrite was made. The changed/new files are:

- `diffpano/config.py`
- `diffpano/dense_consensus.py`
- `diffpano/fusion.py`
- `docs/NATIVE_CONTROLS.md`
- `docs/NATIVE_CONTROLS_REPORT.md`
- `tests/test_current_state_transition.py`
- `tests/test_detail_preserving_consensus.py`
- `tests/test_no_residual_planar.py`
- `diffpano/consensus_audit.py`
- `diffpano/planar_initialization_control.py`
- `scripts/consensus_dense_controls.py`
- `scripts/consensus_planar_controls.py`
- `scripts/consensus_spatial_controls.py`
- `scripts/report_consensus_controls.py`
- `slurm/consensus_diagnostics.slurm`
- `slurm/consensus_validate.slurm`
- `tests/historical_configs.py`
- `tests/test_consensus_audit.py`

Core changes are limited to optional audit hooks, explicit S/T configuration guards and positive-weight normalization/effective-count tracking. New launchers embed paired resolved configs and hashes. Three historical config tests now read retained generation metadata when user-cleaned sidecars are absent.

Gate job **19769155** passed compileall, `git diff --check`, **7 focused tests** and the **205-test full suite**. All scientific source/test hashes still match that gate. The display/report-only script was subsequently adjusted for readable matched-camera panels and stable exact byte accounting, then compiled separately; it does not affect any trajectory.

### S/T — FLUX completed and cross-backend comparison

FLUX S job **19769470** restores recognizable temple towers, columns, arches and stonework. Its panorama remains hazy with repeated, disconnected structures. FLUX T job **19769471** remains severely blurred and does not restore architecture at common ERP display size. Both match L conditioning, initial-state, camera and schedule hashes, with 1,780 guided predictions and actual transformer forwards each. S geometric/weight/N_eff statistics exactly match PixelDiT S. T retains 100% coverage and the same 7–19 geometric contributor range.

| Backend / method | Mean matched-view std | HF1 RMS | HF1/std | Central HF1/std | Runtime s | GPU allocated/reserved GiB | Host max GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| flux P | 0.124827 | 0.000397 | 0.003196 | 0.003614 | 1568.21 | 25.051 / 26.998 | 10.202 |
| flux S | 0.134450 | 0.005442 | 0.040221 | 0.047937 | 1562.37 | 24.956 / 27.002 | 10.160 |
| flux T | 0.125333 | 0.000207 | 0.001666 | 0.001836 | 1616.93 | 25.358 / 27.385 | 12.973 |
| pixeldit P | 0.151689 | 0.001028 | 0.006781 | 0.008694 | 774.71 | 4.108 / 4.545 | 12.374 |
| pixeldit S | 0.178443 | 0.008879 | 0.049685 | 0.048295 | 773.78 | 4.115 / 4.635 | 10.550 |
| pixeldit T | 0.151770 | 0.000705 | 0.004647 | 0.006137 | 936.08 | 4.518 / 5.186 | 16.310 |

P is the verified exact L replay. All view metrics use identical 1024-square diagnostic cameras, including T. They are means of four selected cameras, not a complete panorama quality metric. S increases full-view contrast-normalized HF1 about 12.6x for FLUX and 7.3x for PixelDiT, alongside visibly restored structure; coherence defects remain. T decreases the same metric to about 52% and 69% of P, respectively, and remains visually collapsed. No combined S+T run is supported by these results.

Runtime is the dense pipeline interval including terminal assembly and existing diagnostics, excluding loading and preflight; P includes the separately counted optional stage audit. Small P/S timing differences do not establish a speed advantage. T takes about 3.1% more time for FLUX and 20.8% for PixelDiT in these runs. Total process times, stage timings, GPU model and node are embedded in each metadata file. Host maximum includes initialization/loading.

### Final completion, resource counts and storage

All scientific P–T jobs completed successfully and all saved comparisons were visually inspected. CPU summary/artifact job **19769554** completed in 3m03s. The final Slurm queue is empty. Earlier validation failures were corrected before model execution; A40 P attempts 19769172/19769173 stopped at strict conditioning-hash preflight with zero denoiser predictions. No duplicate successful trajectory was submitted.

Output root: `outputs/vae-residual-controls/20260916-diagnostics-pt/`. The single shared figure is [P-S-T.png](../../outputs/vae-residual-controls/20260916-diagnostics-pt/P-S-T.png). Each directory in the table is relative to that root.

| Experiment/backend | Completed job | Output directory | Measured pipeline seconds | Guided / actual forwards | Peak GPU allocated GiB | Host max GiB |
|---|---|---|---:|---:|---:|---:|
| Q, model-free | 19769161 | Q | 890.55 | 0 / 0 | not recorded | 13.666 |
| P flux | 19769228 | P/flux | 1568.21 | 1780 / 1780 | 25.051 | 10.202 |
| P pixeldit | 19769229 | P/pixeldit | 774.71 | 4450 / 4450 | 4.108 | 12.374 |
| S flux | 19769470 | S/flux | 1562.37 | 1780 / 1780 | 24.956 | 10.160 |
| S pixeldit | 19769301 | S/pixeldit | 773.78 | 4450 / 4450 | 4.115 | 10.550 |
| T flux | 19769471 | T/flux | 1616.93 | 1780 / 1780 | 25.358 | 12.973 |
| T pixeldit | 19769302 | T/pixeldit | 936.08 | 4450 / 4450 | 4.518 | 16.310 |
| R pixeldit, all three methods | 19769230 | R | 104.52 | 450 / 450 | 4.243 | 10.263 |

Total: **19,140 guided predictions and 19,140 actual transformer forward invocations**. CFG is internally batched where applicable. Q and failed P preflights add zero denoiser calls. P adds **16 FLUX VAE decodes, zero additional VAE encodes and zero additional denoiser calls**. Measured optional P diagnostics are 4.96 s for FLUX and 3.97 s for PixelDiT, about 0.32% and 0.51% of their pipeline intervals; these are instrumentation timings, not a controlled overhead benchmark. R paired/independent intervals are 70.43/34.08 s; total process 339.85 s includes loading and the required evidence-review wait.

Exactly **28 experiment artifacts: 17 PNGs and 11 JSONs, 30,975,602 bytes (29.541 MiB)**. This includes execution/validation/review manifests and the shared sheet. Separately, 30 Slurm log files use 97,561 bytes. Combined artifacts plus logs: 58 files, 31,073,163 bytes. Source files, ordinary Python caches and model caches are excluded. All per-experiment file budgets pass. All 136 recorded historical L–O artifact hashes and the original report prefix remain unchanged.

The shared sheet was inspected in both backend panels. S restores structure in equatorial, upper and polar views; the latter also expose severe orientation/scene inconsistencies. P and T remain blurred at all four selected camera directions. None of these results establishes global geometric correctness.

### Measured conclusions and remaining hypotheses

| Candidate cause | Evidence from this package | Scope / remaining uncertainty |
|---|---|---|
| Standard projection loss | Q L/N preserve the tested coherent bands well at baseline resolution; T does not rescue dense generation | Sampling damage exists, but these tests do not support ERP bandwidth alone as the dominant L cause; near-Nyquist content remains untested |
| Averaging incompatible predictions | P shows immediate consensus suppression; S restores structure with the same states, cameras and calls | Strong support for excessive influence from disagreeing trajectories; no complete quantitative decomposition of all feedback effects |
| Independent initialization | R-shared/native retain detail; independent overlaps blur | Positive evidence in one exact planar PixelDiT control; shared/native have documented numerical drift and this is not a spherical-noise solution |
| Uniform geometric influence | S improves both FLUX and PixelDiT, with unchanged geometric coverage and lower N_eff | Direct dense evidence; one prompt/seed per backend and residual incoherence limit generalization |
| Final output assembly | Negligible PixelDiT contrast loss; additional 5.09% FLUX contrast reduction | Measurable FLUX penalty, but severe collapse precedes final assembly |
| VAE conversion | FLUX sampled roundtrips lose about 4–5% contrast | Additional backend-specific damage; insufficient as a sole explanation because PixelDiT also collapses |
| Historical O pyramid operator | Q coherent-input assembly and cycles suppress bands and shift markers | Independent spatial defect/loss requiring scale-level characterization; normalization passes, so no count-denominator bug has been established |

### Ranked next actions

1. **Use S as the provisional dense baseline and test its repeatability/coherence on a small fixed seed set.** It is the only tested intervention that restores visible structure in both backends. Keep temperature 0.1 fixed initially and measure duplicated geometry/orientation as well as local detail.
2. **Design a separate initialization-correspondence control for spherical views, starting with PixelDiT.** R gives a healthy planar reference and implicates the joint initialization structure there. Preserve valid marginal noise statistics and explicitly verify overlap correspondence; ordinary bilinear warping of Gaussian noise is not a validated construction.
3. **Isolate O pyramid scale transport with model-free tests before spending more diffusion runs on it.** The coherent-input losses are already large at direct assembly. Any correction needs a new/versioned operator and independent frequency/marker tests; historical O remains unchanged.
4. **Consider a small, separate FLUX last-clean-output ablation.** P measures a modest additional terminal contrast loss, but changing output semantics will not solve the earlier collapse. PixelDiT offers little evidence for prioritizing this change.

Deprioritize larger ERP resolution and a combined S+T run on current evidence. There is no result here that motivates replacing the VAE, training a bridge or adding time travel.


## Experiment V — One-Time ERP Noise Initialization (September 17)

V changes only initialization of the completed S center-weighted pipeline. The direct-local S outputs are reused; no S model replay is planned. V-independent-erp draws one CPU FP32 native-channel ERP Gaussian per canonical camera; V-shared-erp draws one field for all cameras. Each source is sampled once with the exact existing nearest/pixel-center/wrap/pole conventions, scaled once through `NativeStateMixin.initialize_native_state`, then discarded. Persistent denoising states remain local native x_t tensors. For FLUX this is honestly a transient ERP-indexed 16-channel raw-latent field at initialization, not a persistent spherical latent sampler.

Predetermined primary grids use H=ceil(pi*max_camera(max(fx_native,fy_native))), W=2H. The actual adapter shape properties resolve PixelDiT native views to 3x1024x1024 and FLUX to 16x128x128, giving **1917x3834** and **240x480** noise grids. FP32 source fields require approximately 84.1 MiB and 7.03 MiB per batch element. Half-grid checks use ceil(H/2), double-grid checks use 2H; these are model-free diagnostics, not image-quality sweeps. The clean RGB ERP remains **1024x2048** for every run.

| Planned run | Views / steps | Guided predictions | Approximate S runtime / allocated GPU memory |
|---|---:|---:|---|
| PixelDiT independent ERP | 89 / 50 | 4,450 | 774 s / 4.12 GiB |
| PixelDiT shared ERP | 89 / 50 | 4,450 | 774 s / 4.12 GiB |
| FLUX independent ERP | 89 / 20 | 1,780 | 1,562 s / 24.96 GiB |
| FLUX shared ERP | 89 / 20 | 1,780 | 1,562 s / 24.96 GiB |

All four run under A100 Slurm allocations with the same checkpoint/cache, precision, conditioning, schedule, camera order, standard RGB warp, S center weights (temperature 0.1), current-x_t transition, Jacobi update and terminal assembly. The FLUX checkpoint remains ModelsLab/flux.1-dev with S's recorded cache revision; its older historical resolved revision is unavailable. The PixelDiT checkpoint and official-code commit are inherited exactly from S metadata.

Nearest selection preserves scalar Gaussian marginals, **not IID samples when source indices repeat**. V-independent and V-shared use exactly the same sampling matrices and within-view covariance; their intended difference is cross-view field sharing. Initialization and diagnostics use separate CPU generators. No empirical patch normalization, whitening, clipping, variance repair, noise reinjection or VAE initialization is introduced.

The numerical gate adds focused coverage for arbitrary channels, shape, exact nearest/tie/pole/wrap behavior, scaling, no-VAE initialization, source release, covariance, camera-order invariance, RNG isolation, unchanged direct initialization and trajectory/call-count preservation. A model-free preflight streams the full camera maps and Monte Carlo checks before any new model execution. Historical S milestone local predictions were not saved: S remains a valid paired final-output reference, but early/middle/late local agreement is compared between the two V variants only. No S rerun is added solely to fill that diagnostic gap.

Artifacts are kept under `outputs/vae-residual-controls/20260917-noise-v/`, separate from P–T. The generation config remains explicitly identified as the inherited S sampler, with a separate V initialization config and top-level experiment identifier V. This preserves S config validation without weakening historical guards.

### V validation gate

Job 19771144 stopped at one overly strict diagnostic test: constant RGB alignment through four FP32 bilinear weights had MAE 1.02e-9 rather than bit-exact zero. The assertion now uses a four-FP32-epsilon accumulation bound; nearest-noise comparisons remain bit-exact. No model ran under that failed gate.

Replacement job **19771149** passed compilation, diff whitespace checks, **10 focused V tests and all 215 full regression tests**. It then runs the complete half/primary/double map and covariance preflight. Source hashes are frozen at that gate and verified by every model launcher. Existing S and P–T scientific source files are unchanged.

### V initialization preflight completed

Job **19771149** completed the complete preflight with zero models/VAEs: 19.23 s measured geometry/statistics, 1.301 GiB process maximum RSS; its 9m20s Slurm duration also includes environment startup and regression tests. All exact map checks and statistical covariance gates passed. The predetermined primary grids remain unchanged.

| Backend | Noise grid | Mean unique fraction | Mean duplicate fraction | Largest cell reuse | Mean horizontal / vertical neighbor collision | Selected ray-pair cell agreement |
|---|---|---:|---:|---:|---|---:|
| pixeldit half | 959x1918 | 0.232893 | 0.767107 | 16 | 0.462419 / 0.495608 | 0.876448 |
| pixeldit primary | 1917x3834 | 0.722956 | 0.277044 | 5 | 0.124479 / 0.155352 | 0.755598 |
| pixeldit double | 3834x7668 | 0.999905 | 0.000095 | 2 | 0.000001 / 0.000071 | 0.594981 |
| flux half | 120x240 | 0.237844 | 0.762156 | 15 | 0.460388 / 0.494938 | 0.899386 |
| flux primary | 240x480 | 0.730336 | 0.269664 | 4 | 0.120510 / 0.153593 | 0.785714 |
| flux double | 480x960 | 0.999934 | 0.000066 | 2 | 0.000000 / 0.000044 | 0.596006 |

These source-index agreement fractions are potential sharing for the shared-field variant. Independent fields have zero expected cross-view covariance even when their index coordinates match. The statistic covers six selected overlapping camera pairs, not every panorama point or all O(N²) camera pairs; polar matches contribute strongly. At primary resolution, equatorial pairs share 61.4–71.9% of indices for PixelDiT and 73.5% for FLUX; upper/lower pairs share about 45.1–45.4% and 51.9%; polar rotation pairs share 100% and 98.6%.

For nonpolar matched rays, mean angular separation is about 0.030–0.032 degrees for PixelDiT and 0.232–0.270 degrees for FLUX; maximum is 0.0638 and 0.4484 degrees. Polar rotation pairs coincide to FP32 geometric precision. Ray matches are selected by projection followed by a 3x3 nearest-angular candidate search, never by equating array indices. Exact poses, all-camera reuse summaries and pair-specific values are retained in the preflight JSON.

At the primary grids, 8,192-draw Monte Carlo cross-view covariance estimates are 0.78015 versus expected 0.78125 for PixelDiT and 0.81837 versus 0.82031 for FLUX on the bounded diagnostic subsets. Independent-field estimates are 0.00030/-0.00044 versus expected zero; cross-channel covariance is about 0.00157 versus zero; same-cell covariance about 0.99921 versus one. These subset expectations differ from the all-selected-ray aggregate because the Monte Carlo routine subsamples 128 pairs. Within-view horizontal/vertical covariance also passes against exact source-index equality. Dedicated diagnostic RNG does not advance initialization RNG.

The primary sampler therefore introduces substantial within-view correlation, as designed and explicitly measured. Increasing the diagnostic grid nearly removes duplicate indices but also reduces sampled cross-view cell agreement. This tradeoff is recorded; no grid was selected after observing image quality, and only the primary grid enters model runs.

PixelDiT independent/shared model jobs **19771157/19771158** were submitted only after this completed preflight review.

### V execution order and release gate

PixelDiT jobs 19771157/19771158 run independently on matched A100 hardware. Comparison job **19771159** depends on both. The self-contained numerical validity job **19771186** then verifies exact S sampler settings, prepared schedule, conditioning, weighting, source maps, first-camera equality, scaling, source draw counts 89/1, finite results and 4,450 predictions/forwards per variant. It releases FLUX jobs **19771183/19771184** only on success. FLUX comparison job **19771187** depends on both FLUX runs. Image quality is interpreted separately and is not used to suppress a negative PixelDiT result or skip FLUX.

A pending administrative gate 19771167 was canceled before execution and replaced with 19771186 so the batch script is self-contained on compute nodes rather than referring to login-node /tmp. No scientific trajectory was duplicated. PixelDiT pending walltime reservations were reduced from 40 to 30 minutes in place; all hardware and scientific settings stayed fixed. Queue forecasts were several hours, but both PixelDiT jobs subsequently obtained backfill slots. All job states and workflow hashes are recorded in execution.json.

### V PixelDiT completed: detail and agreement are different outcomes

Jobs **19771157/19771158** and comparison/validity jobs **19771159/19771186** completed. Each model run uses exactly 4,450 guided predictions and 4,450 transformer forwards. Exact S config, prepared schedule, camera, conditioning, weight and checkpoint comparisons pass. The direct-local initializer regenerated for provenance matches S's saved digest; the two V modes have identical first-camera tensors, native shapes, maps and scaling, but different complete initial-state digests. Source draw counts are 89 and 1. All source fields are released before diffusion.

The final panoramas, four matched perspective cameras and aligned final local-clean pair were visually inspected. Both V variants retain identifiable temples, columns, doorways and masonry. Independent ERP increases color and contrast relative to S. Shared ERP is much sharper but also strongly saturated, with harsh highlights, fragmented terrain/sky-water boundaries and structures with incompatible apparent scene orientation. These observations do not establish a final coherence improvement. Repeated temples alone are not used as evidence: in the actual aligned camera-7/8 footprint, corresponding doorway, terrace and roof features largely coincide in both V variants, although fine edges differ. Global scene plausibility and local overlap compatibility remain separate issues.

| PixelDiT method | Mean four-view RGB std | HF1 RMS | HF1/std | Central HF1/std | Pipeline s | Initialization s | Diagnostic s | GPU allocated / reserved GiB | Host max GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| S direct local | 0.178443 | 0.008879 | 0.049685 | 0.048295 | 773.78 | historical | historical | 4.115 / 4.635 | 10.550 |
| V independent ERP | 0.392162 | 0.023052 | 0.057392 | 0.053313 | 788.35 | 26.29 | 7.18 | 4.115 / 4.635 | 11.528 |
| V shared ERP | 0.598108 | 0.054091 | 0.089397 | 0.087395 | 779.73 | 14.69 | 2.50 | 4.115 / 4.635 | 12.291 |

These are raw tensor metrics before ordinary display clipping. HF1 is RMS of the image minus its separable Gaussian blur at sigma=1 pixel (radius 3, reflected boundaries), averaged across the same four full-size cameras; normalized columns average each view's ratio, not a ratio of aggregate means. Higher values alone do not establish better detail or geometry. The shared image's visible saturation and edge artifacts make that distinction particularly important. Pipeline timing includes diagnostics; initialization and model loading are outside that interval. Unequal diagnostic timings are instrumentation observations, not a controlled performance benchmark.

| PixelDiT milestone | Independent / shared view-to-consensus MAE | Independent / shared RGB clean modification | Independent / shared aligned local-pair MAE | Independent / shared normalized local-pair MAE |
|---|---|---|---|---|
| 10%, step 5 | 0.48263 / 0.33362 | 0.55194 / 0.38554 | 0.89081 / 0.44732 | 0.99654 / 0.68946 |
| 50%, step 25 | 0.18705 / 0.14721 | 0.24983 / 0.19824 | 0.40330 / 0.22564 | 0.75420 / 0.31914 |
| 90%, step 45 | 0.04591 / 0.03520 | 0.06345 / 0.05762 | 0.09716 / 0.05770 | 0.21723 / 0.08147 |
| Final, step 50 | 0.01094 / 0.01163 | 0.01580 / 0.02276 | 0.02345 / 0.03216 | 0.05246 / 0.04553 |

The early and middle shared-field agreement advantage is clear under these definitions. Its final absolute-error advantage is lost: final local-pair MAE is 37% higher and view-to-consensus MAE 6% higher than independent ERP. Final contrast-normalized pair MAE is 13% lower, partly reflecting the substantially higher target contrast (0.7063 versus 0.4470). This mixed endpoint does not support claiming maintained final coherence improvement. It also does not mean the final trajectories are wholly incompatible; both pair errors are much smaller than at initialization-side milestones. The paired footprint covers 64.07% of camera B and is one selected overlap, not a global correspondence guarantee.

Native clean modification equals RGB modification for PixelDiT. Maximum current-state reconstruction errors are 4.77e-7, 4.77e-7, 2.38e-7 and 5.96e-8 at the four milestones in both variants. This verifies transition algebra, not visual quality. At the middle milestone, local/returned-consensus HF1 is 0.08002/0.02157 for independent and 0.06911/0.04271 for shared: fusion still suppresses disagreement-associated detail, but less strongly in the shared run. The final local/returned values are 0.01978/0.02310 and 0.04563/0.05414, respectively; such resampling/assembly differences are not extra model predictions.

The supported PixelDiT conclusion is an early agreement benefit with no clear final coherence win, alongside a strong change in contrast and texture. This is closest to the requested early-benefit-not-maintained case, with the qualification that normalized final disagreement remains slightly better. S's unsaved local predictions prevent a numerical S-versus-V trajectory comparison; only its paired final views and final detail metrics are available.

While the FLUX pair waited for resources, its pending reservations were adjusted in place to **40 minutes and 32 GiB host memory**, retaining one A100 and eight CPUs. Historical FLUX S used 31m53s of allocation and 10.160 GiB peak process RSS, so this leaves time and memory headroom without changing model settings. Slurm requires numeric MiB for this memory update; an initial `32G` update was rejected and the accepted value was 32768 MiB. No new job or model run was submitted for these administrative changes.

PixelDiT display-range audit: mean matched-view scalar fractions outside [-1,1] are 0% for S, 0.124% for independent ERP and **9.439% for shared ERP**; area-weighted ERP fractions are 0.00015%, 0.167% and **11.123%**, respectively. These are pre-display output values, not clipped initialization noise. Ordinary PNG conversion clips them for display. The large shared-run fraction quantitatively supports the visible harsh highlights/contrast warning and further limits interpretation of increased raw HF1 as useful detail.

### V implementation boundaries and preserved invariants

The V implementation is additive: `diffpano/erp_noise_initialization.py` supplies explicit noise config, native-shape camera construction, exact nearest source-index maps and bounded CPU initialization. `diffpano/noise_v_diagnostics.py` measures already-produced clean RGB predictions and embeds two compact final overlap thumbnails per run in metadata. `scripts/noise_v_common.py`, `scripts/noise_v_preflight.py`, `scripts/noise_v_run.py` and `scripts/report_noise_v.py` provide S pairing, preflight, guarded execution and comparison. Ten focused tests live in `tests/test_erp_noise_initialization.py`; three new Slurm scripts handle validation, comparison and the PixelDiT validity release gate. The existing generic model launcher is reused.

The backend already exposed `NativeStateMixin.initialize_native_state`, so no backend scaling API or historical scientific implementation needed modification. The new initializer invokes that exact helper once per standard Gaussian local tensor. Both tested backends resolve scale to 1.0, while the focused scale test also covers a nonunit 2.5 scale. PixelDiT channels come from its actual adapter; FLUX raw channels and spatial scale come from the loaded transformer/VAE configuration and actual adapter properties. Transformer packing remains inside FLUX's adapter.

The noise projector carries integer row/column coordinate channels separately through the established nearest padded ERP grid convention before forming int64 source indices. This avoids loss of integer precision from encoding a large flattened ERP index in FP32. The geometry tests include pixel centers, odd/even sizes, seams, pole reflection, roll and nearest tie behavior. No historical RGB projection helper was changed. Source sampling occurs directly at native spatial resolution; no RGB-resolution noise resize is used.

Explicit invariants for every V run:

- Source ERP noise is generated and used only during initialization, with one field resident at a time, and released before the evolving local-state loop.
- Noise sampling is nearest, CPU FP32, with independent source channels and dedicated seeded CPU RNG. Camera assignment follows the stable canonical order, independently of inference execution order.
- No VAE generates or decodes initial Gaussian noise. No empirical per-patch normalization, whitening, variance repair, clipping or view-count scaling is applied to that noise.
- No fixed-noise renoising, later source sampling, initial-noise reinjection or persistent ERP noisy/native field is introduced. FLUX's ERP raw-latent source is transient and explicitly recorded.
- S's 89 cameras, 1024-square RGB views, clean ERP 1024x2048, center weighting/temperature 0.1, RGB interpolation, scheduler arrays, current-x_t interpolation, Jacobi synchronization, guidance, conditioning, precision, VAE settings and terminal assembly remain unchanged.
- DPA, LPW, VAE residual correction, time travel and camera motion are untouched. Initialization and diagnostics add zero denoiser calls and zero VAE calls; ordinary S-loop VAE conversions remain where applicable.

Full scientific source hashes accompany each run, in addition to HEAD and dirty-diff provenance. Documentation can evolve while runs execute; the scientific source hash dictionary must remain identical to the passed validation gate. The historical S final outputs are reused because configuration, conditioning, model/cache, schedule, camera/weight and direct-initializer comparisons pass. FLUX's older resolved revision remains unavailable; the preserved ModelsLab cache revision `fa45a9eb6808ba8fdfc7cc2756f7f1a16e0921f4` is the one recorded and checked against S, with offline loading. No checkpoint substitution or S model replay was made.

### V FLUX completed: early overlap benefit, nearly tied endpoint

Jobs **19771183/19771184** and CPU comparison **19771187** completed. Each model run produced exactly 1,780 guided predictions and transformer forwards. The full numerical audit passes: S configuration/conditioning/checkpoint/camera/weight and direct-initializer hashes match; prepared coefficient arrays are identical; independent/shared native maps, shapes, source dimensions, scale and first-camera tensors match. The only literal schedule-metadata difference from S is ordering of the `_use_default_values` bookkeeping list. The existing `canonical_schedule` helper sorts that unordered key list; no coefficient, scheduler parameter or trajectory setting differs.

Both final panoramas, S, all four common diagnostic cameras, and the aligned final local pair were visually inspected. Independent ERP has more contrast than S but soft, poorly resolved facades and terrain. Shared ERP restores more visible masonry, towers, gables and wall edges than independent ERP, with richer foreground structure. S retains some cleaner individual temple/roof forms despite its haze. Shared ERP does not establish a more coherent global scene: terrain/path/water boundaries remain fragmentary, and upper/polar views do not resolve a convincing continuous environment. These are scene-level observations, not a claim that repeated buildings prove mismatched correspondence.

Within the actual camera-7/8 aligned footprint, the central stepped temple, adjacent wall and terrain boundaries in shared ERP largely occupy the same locations in both local predictions. The corresponding doorway/lake boundary also largely matches in independent ERP. Neither pair shows a clear gross displaced duplicate of the same feature in the displayed overlap. Remaining fine differences and smoothing are represented by the raw pair errors below; the near tie does not establish a final shared-field coherence gain. Historical S local predictions are unavailable, so no S pair-error value is invented.

| FLUX method | Mean four-view RGB std | HF1 RMS | HF1/std | Central HF1/std | Pipeline s | Initialization s | Diagnostic s | GPU allocated / reserved GiB | Host max GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| S direct local | 0.134450 | 0.005442 | 0.040221 | 0.047937 | 1562.37 | historical | historical | 24.956 / 27.002 | 10.160 |
| V independent ERP | 0.201431 | 0.005502 | 0.027554 | 0.024011 | 1576.20 | 3.08 | 5.52 | 24.977 / 27.002 | 10.144 |
| V shared ERP | 0.230730 | 0.008423 | 0.036592 | 0.038918 | 1558.09 | 0.57 | 2.25 | 24.975 / 27.006 | 10.239 |

FLUX shared ERP has 53% more absolute HF1 and 33% more normalized HF1 than independent ERP on these cameras. Against S, its absolute HF1 is 55% higher but normalized HF1 is 9% lower (central normalized HF1 19% lower). Independent ERP's normalized HF1 is 31% below S and central normalized HF1 about 50% below S. Thus recognizable recovered detail survives, particularly with sharing, but neither a universal sharpness improvement over S nor unchanged texture quality is established. All three FLUX methods have zero measured final out-of-range fraction in the ERP and four views; PixelDiT's clipping issue does not recur here.

| FLUX milestone | Independent / shared view-to-consensus MAE | Independent / shared RGB clean modification | Independent / shared native clean modification | Independent / shared aligned local-pair MAE | Independent / shared normalized local-pair MAE |
|---|---|---|---|---|---|
| 10%, step 2 | 0.45416 / 0.45130 | 0.49515 / 0.50342 | 0.86866 / 0.85931 | 0.46992 / 0.35843 | 0.87378 / 0.59045 |
| 50%, step 10 | 0.08644 / 0.07732 | 0.09982 / 0.09082 | 0.41771 / 0.39985 | 0.15778 / 0.13506 | 0.56728 / 0.43439 |
| 90%, step 18 | 0.01877 / 0.01907 | 0.02140 / 0.02189 | 0.24182 / 0.24592 | 0.02912 / 0.02356 | 0.11669 / 0.08910 |
| Final, step 20 | 0.00772 / 0.00854 | 0.00908 / 0.01086 | 0.20048 / 0.20382 | 0.01128 / 0.01133 | 0.04670 / 0.04489 |

Shared ERP reduces aligned pair MAE by about 24%, 14% and 19% at early/middle/late milestones. At the final step it is 0.5% higher, effectively tied at the scale of this one-run comparison. The final normalized error is 3.9% lower, while full-camera view-to-consensus error is 10.7% higher. The early global view-to-consensus change is only 0.6%, much weaker than the selected pair's early improvement. These different aggregations must not be conflated. Native modification includes the unchanged FLUX decode/fuse/re-encode path and is not a latent-space geometric correspondence test.

Maximum current-state reconstruction errors are 4.77e-7, 4.77e-7, 2.38e-7 and 2.38e-7 at the milestones in both variants. Middle local/returned HF1 is 0.01600/0.00564 for independent ERP and 0.00909/0.00779 for shared ERP. Final values are 0.00520/0.00584 and 0.00815/0.00889, before terminal output's final HF1 of 0.00550 and 0.00842. Detail is still affected by consensus and terminal assembly; neither was changed to favor V.

### V three-way conclusions

| Backend | S direct local | V independent ERP | V shared ERP |
|---|---|---|---|
| PixelDiT | Recognizable but hazy architecture; weak global scene coherence | More contrast and visible architecture; altered initialization operator strongly affects appearance | Strong early/middle agreement benefit; much sharper but saturated/artifact-prone; final raw agreement advantage lost, no clear final coherence win |
| FLUX | Hazy but recognizable individual roofs/temples | Increased contrast but softer normalized detail; no established coherence gain | More structure than independent ERP; early/middle pair agreement improves; final absolute pair error essentially tied, no clear coherence win over S |

The tested coupling is substantial, not absent: exact source-index checks and Monte Carlo covariance verify real geometric sharing, with about 75.6%/78.6% selected-ray agreement at the primary grids. The independent/shared controls have the same within-view sampling covariance by construction and identical measured maps. That covariance differs substantially from S's direct IID local initializer: about 27.7%/27.0% duplicate-source fractions. Independent ERP's large appearance changes show why comparing only shared ERP against S would confound coupling with the sampling operator.

The data support a limited version of the hypothesis: common ERP-indexed noise helps selected early/middle perspective predictions agree. They do **not** establish a maintained final coherence improvement beyond S's center-weighted fusion in either backend. PixelDiT loses its final raw-error advantage; FLUX's final raw pair error is nearly tied. Slightly lower final contrast-normalized pair errors coexist with higher contrast and slightly worse global view-to-consensus errors. One prompt/seed, one selected local overlap, incomplete global geometry assessment and missing S trajectory diagnostics limit inference. These results do not rule out other shared initialization constructions or establish the primary grid as optimal.

### V final jobs, calls and artifacts

| Run | Job / node | Slurm elapsed | Pipeline s | Total process s | Guided / actual forwards | Peak GPU allocated GiB | Host process max GiB |
|---|---|---|---:|---:|---:|---:|---:|
| PixelDiT independent ERP | 19771157 / g038 | 19m18s | 788.35 | 1026.01 | 4450 / 4450 | 4.115 | 11.528 |
| PixelDiT shared ERP | 19771158 / g047 | 16m23s | 779.73 | 890.90 | 4450 / 4450 | 4.115 | 12.291 |
| FLUX independent ERP | 19771183 / g055 | 32m45s | 1576.20 | 1817.85 | 1780 / 1780 | 24.977 | 10.144 |
| FLUX shared ERP | 19771184 / g025 | 29m52s | 1558.09 | 1696.62 | 1780 / 1780 | 24.975 | 10.239 |

Exactly **four new scientific model runs**, with **12,460 guided predictions and 12,460 actual transformer forward invocations**. Both S references are reused without denoising replays. Initialization/preflight and added diagnostics account for **zero extra denoiser calls and zero extra VAE encodes/decodes**. Host values above are process-recorded peaks; sampled Slurm MaxRSS is retained separately and can miss transient peaks. Pipeline time excludes initialization/loading; total process excludes part of batch environment startup. These timings are not controlled speed benchmarks.

Validation/preflight **19771149** completed in 9m20s with compilation, diff checks, ten focused tests, the full **215-test suite**, and all half/primary/double geometry/statistical gates passing. PixelDiT comparison **19771159** took 2m31s, release audit **19771186** took 1s, and FLUX comparison **19771187** took 1m34s. The earlier test-only failure 19771144 and canceled administrative gate 19771167 are documented above; no scientific model job failed or was duplicated. All four runs and both comparison sheets received numerical and visual review.

Outputs: `outputs/vae-residual-controls/20260917-noise-v/`. Each of `pixeldit/V-independent-erp`, `pixeldit/V-shared-erp`, `flux/V-independent-erp`, `flux/V-shared-erp` contains only `final_result.png` and `metadata.json`. Shared artifacts are `initialization-preflight.json`, `execution.json`, `results.json`, [pixeldit-comparison.png](../../outputs/vae-residual-controls/20260917-noise-v/pixeldit-comparison.png), and [flux-comparison.png](../../outputs/vae-residual-controls/20260917-noise-v/flux-comparison.png). The two aligned final local thumbnails per V run are embedded in its metadata; there are no separate camera images, noise/source tensors, index-map dumps, full covariances or timestep images.

The final audit verifies all **164 historical artifact hashes**, **70 prior scientific-source hashes**, the original report prefix and all **116 validated scientific-source hashes** unchanged. Workflow script hashes also pass. Branch remains `no_sphere`, HEAD remains `8bf6d13257a32db5ee279ca973cde2dadc9beb25`, all P–T uncommitted work is preserved, and no commit was made.

Final exact storage: **13 artifacts (6 PNG, 7 JSON), 14,550,539 bytes (13.876 MiB)**. Separately, **18 Slurm logs use 48,729 bytes**; combined artifacts plus logs are **31 files, 14,599,268 bytes**. These totals include embedded overlap thumbnails and the completed execution/result manifests, and exclude source/model caches and ordinary Python caches. No extra scientific run, seed or model-quality grid sweep was added.

## Bridge + Directional Prompting 2^4 Factorial Study — Ruins

This new study is separate from historical A–V. Starting branch is `no_sphere`, HEAD `8bf6d13257a32db5ee279ca973cde2dadc9beb25`; the dirty P–V work and results are preserved. The output root is `outputs/bridge-factorial-ruins/20260918/`. The design has exactly 80 DiffPano scientific cells (five backends times 16 A/B/C/D combinations), plus two external original SphereDiff references. Four pilot cells per backend are part of those 80 and were reused. No additional seeds, prompts, parameter tuning or backend substitutions were added.

### Prompt and fixed angular geometry

Both `prompts/ruins.txt` and the original SphereDiff `data/prompts/ruins.txt` have SHA-256 **e74ca0410b7f22a43842a41a08ecfe857ac196cbf7379ae2c11585017cf79de0**, are byte-identical, and contain exactly five physical lines:

```text
An upward view of the night sky filled with countless stars and the Milky Way stretching across, creating a breathtaking cosmic scene. The ruins' silhouettes subtly frame the sky, adding a sense of ancient mystery.
An upward view of the night sky filled with countless stars and the Milky Way stretching across, creating a breathtaking cosmic scene. The ruins' silhouettes subtly frame the sky, adding a sense of ancient mystery.
A grand view of ancient ruins under a vast, starry night sky. The weathered stone columns and structures stand as silent witnesses to history, illuminated by the soft glow of moonlight and distant celestial bodies.
A directly downward view of the ancient ruins, showing only the moss-covered stone foundations and weathered ground. Cracked stone pathways and scattered remnants of fallen pillars blend into the rugged terrain, illuminated by the faint glow of flickering torches or lanterns. The interplay of light and shadow highlights the textures of the aged stone and creeping vegetation.
A upside-down view of the moss-covered stone foundations and weathered ground. Cracked stone pathways and scattered remnants of fallen pillars blend into the rugged terrain, illuminated by the faint glow of flickering torches or lanterns. The interplay of light and shadow highlights the textures of the aged stone and creeping vegetation.
```

The source is SphereDiff commit `2c8c68ba088f2803b3dce4b52b7b0d68bc996139`. The user's `/home/shig/SphereDiff` checkout is at a later HEAD with unrelated launcher edits; its pipeline code and ruins prompt have no diff from the requested commit. To avoid changing that checkout, an exact source snapshot is held at `/home/shig/diffpano_reference_sources/spherediff-2c8c68b/`, with recorded source hashes.

The 89 canonical pose records are read directly from V's saved preflight and checked against L's existing geometry artifact and camera digests. No new cover is generated. The rings at pitches -90/-67.5/-45/-22.5/0/+22.5/+45/+67.5/+90 degrees contain 4/8/11/14/15/14/11/8/4 cameras, matching the saved artifact. Every camera retains its exact yaw, pitch, roll and **80 by 80 degree FOV**. A common `camera_geometry_sha256` hashes only these angular properties and their order, and must match across all 80 cells. The existing raster-inclusive camera digest is also retained.

As clarified by the user, raster resolution is separate from perspective footprint. SD2 retains 512x512 local RGB and a 512x1024 clean ERP; SANA, FLUX, SD3.5 and PixelDiT retain 1024x1024 local RGB and a 1024x2048 clean ERP. These are fixed within each backend. The spherical patch is always an 80-degree frustum, with fx=W/(2*tan(40 degrees)) and fy=H/(2*tan(40 degrees)); planar patch-size/stride fields do not control this study's coverage. Standard and LPW share the exact poses/FOV; pyramid levels change sampling density, not angular extent. Native initialization constructs its maps directly at raw native resolution using the same angular camera records.

Directional routing retains five vertical semantic bands and four yaw anchors per band, with maximum-cosine selection. Historical L–V's repeated global text is not silently reused: this new study uses the exact original night-sky ruins file above. Seed 0 is verified from all five existing dense configs and fixed throughout.

### Common bridge and transition; factors

All four latent backends use the **local identity-preserving VAE bridge**: decode model clean z0 once to I; encode that exact I to zrt; retain r=z0-zrt in the same camera's native coordinates; fuse/project clean RGB to Isync; encode Isync and add local r. The resulting bridged clean is passed to the existing `interpolate_from_current_state`. Residuals are never warped, spatially fused or assembled into an ERP latent canvas. PixelDiT uses `vae_bridge=not_applicable_identity` and synchronized clean RGB directly.

The existing dense Jacobi loop is reused with optional local-residual hooks whose defaults preserve historical behavior. The new `erp_bridge_factorial` configuration has its own strict validation; historical L–V guards remain. All first-pass predictions see frozen local x_t, and next states are committed only after every view transition. Actual prepared flow sigmas or SD2 DDIM alpha/sigma coefficients drive current-state interpolation. No old endpoint, `reconstruct_next(clean=...)`, fixed-noise renoising or original-noise reinjection enters the factorial trajectory.

A0/A1 reuse V's exact CPU FP32 nearest ERP initializer and predetermined center-density noise-grid rule. They differ only in 89 separate source draws versus one shared source. Sources are released before diffusion. C0 uses the existing `weighted_average` reducer for both D levels, so `average` cannot erase D1 weights. C1 uses existing DPA with alpha=1, power=1 and epsilon=1e-6. D0 is uniform valid support; D1 is the existing exp(-norm(u)/0.1) center map. All four C/D combinations run in standard RGB and independently at the current LPW coefficient levels. B1 is the existing O/Q-tested **current DiffPano LPW adaptation**, with five levels, no Jacobian LOD heuristic, periodic ERP reconstruction and unchanged nearest ERP-to-view/bilinear view-to-ERP interpolation; it is not claimed to be exact LookingGlass.

### Validation and launch gates

Preliminary CPU job **19784411** passed nine focused tests, including all A/B/C/D paths, local bridge/no-op behavior, current-state reconstruction, endpoint poisoning, Jacobi order and unchanged model counts. Additional contrast and camera-boundary diagnostic tests are included in the final gate. No scientific generation was released by the preliminary gate.

The final CPU gate runs compilation, diff whitespace checks, all focused factorial tests and the complete existing regression suite, then freezes the manifest and source hashes. Five real-backend GPU preflights will verify actual native shapes, A0/A1 digests, coverage, conditioning/schedules and real VAE bridge identity before any pilot cell. Identity uses a per-element finite-precision bound `8*eps(native arithmetic dtype)*(abs(z0)+abs(E(D(z0)))+1)`, recording each backend's VAE dtype, actual error and magnitude-dependent bound rather than one blind absolute tolerance. Preflights perform zero denoiser predictions.

The pilot is A0B0C0D0, A1B0C0D1, A0B1C1D0 and A1B1C1D1 for every backend. Numerical and visual review must pass before releasing the remaining 60 cells, without tuning based on pilot appearance. The launcher checks completed outputs, the live Slurm queue and prior accounting before submitting or retrying a missing/failed cell.

The two external references use unchanged original `SphericalFluxPipeline` and `SphericalSanaPipeline`, their original checkpoint IDs and cached revisions, no model CPU offload and bf16 precision. FLUX retains 28 steps, guidance 3.5, true CFG 1, 26,500 points and temperature 0.1. SANA retains 20 steps, guidance 4.5, 1024-square call dimensions, 2,600 points, bf16 variant and temperature 0.1. Both retain native 2048x4096 ERP outputs. A CUDA generator seeded to zero is passed through the normal original API; initialization is not claimed to match DiffPano. The wrapper follows the original launcher's solver-order handling and otherwise only records timing, provenance and outputs.

### CPU validation

Full gate job `19784419` passed compilation and `git diff --check`, all **11 focused factorial tests** (15.326 s), and all **226 regression tests** (101.867 s). The earlier nine-test smoke gate `19784411` also passed; it was superseded by the expanded final gate. The focused suite exercises all sixteen factor combinations with flow, DDIM and pixel mocks, poisoned endpoint independence, bridged-clean transition inputs, Jacobi ordering, model/VAE call counts, orthogonal C/D reducers at LPW levels, saved angular geometry, exact prompt routing, matched initialization, and contrast arithmetic. Real-backend identity and initialization checks follow separately before the matrix.

### Real-backend preflight results

All five preflights passed with zero denoiser predictions. The four real VAE encoders were deterministic under repeated posterior-mode encoding. The maximum elementwise recovery-error/bound ratio was below 0.056 for every latent backend; the bound is defined above. PixelDiT has no VAE and its identity bridge is not assigned a synthetic VAE error. All A0/A1 pairs had identical sampling-map hashes, grid dimensions, initialization scales, native shapes and first-camera state hashes. Coverage was complete for every backend.

| Backend | Native channels × H × W | Noise ERP H × W | Mean duplicate-source fraction | Maximum bridge identity error | Maximum error/bound ratio |
|---|---|---|---:|---:|---:|
| sd2 | 4 × 64 × 64 | 120 × 240 | 0.265537 | 2.38419e-07 | 0.049566 |
| sana | 32 × 32 × 32 | 60 × 120 | 0.255267 | 4.76837e-07 | 0.05486 |
| flux | 16 × 128 × 128 | 240 × 480 | 0.269664 | 4.76837e-07 | 0.0554617 |
| sd35 | 16 × 128 × 128 | 240 × 480 | 0.269664 | 4.76837e-07 | 0.0553872 |
| pixeldit | 3 × 1024 × 1024 | 1917 × 3834 | 0.277044 | N/A | N/A |

The frozen angular geometry SHA-256 is `b55b4eb5051cdfb8c579e81d4f51ece1f15aeb0e8d7e1fde6b7b74a49154435c` across all 80 cells. `provenance.json` records all 1,647 preexisting output hashes. FLUX component configuration files match byte-for-byte; the seven Diffusers weight paths have identical cached content addresses and sizes. The official cache additionally contains `ae.safetensors` and `flux1-dev.safetensors`, which makes the whole-cache inventory comparison unequal. Weight bytes were not independently rehashed, so this is strong cache provenance evidence rather than independently established checkpoint-byte equivalence. Model IDs and revisions remain explicitly distinct.

### External reference execution

Original SphereDiff SANA completed under source commit `2c8c68ba088f2803b3dce4b52b7b0d68bc996139`, retaining the original pipeline and scheduler behavior. The native 2048 × 4096 ERP is in `outputs/bridge-factorial-ruins/20260918/references/sana/`. Generation took 212.441 s with 1780 transformer forwards and 10.241 GiB peak allocated GPU memory. It used the original SANA checkpoint/revision, BF16 variant and precision, 20 steps, guidance 4.5, 1024 × 1024 requested local dimensions, 2,600 spherical points, temperature 0.1, seed-0 CUDA generator, and no model CPU offload or VAE tiling. Runtime environment: Torch 2.7.0+cu126, Diffusers 0.32.2, Transformers 4.49.0. Final comparison is separate from factorial contrasts and uses matched perspective rasters.

The original SANA scheduler executes with **solver order 1**, exactly as assigned by the original static launcher. Diffusers 0.32.2 `FrozenDict` retains a mapping value of 2 when its `solver_order` attribute is set to 1; the scheduler reads the attribute. A source-derived microcheck confirmed this distinction. The initial wrapper metadata used mapping `.get()`; its effective-order field was corrected to 1, retaining the originally captured mapping value and the verification hashes. No generation or algorithm was changed. The original spherical local transformer inputs were `[2,32,20,20]` for 1,760 forwards and `[2,32,21,21]` for 20 forwards, despite nominal height/width arguments of 1024. This differs from DiffPano’s fixed 32 × 32 SANA native raster and is another reason the reference is a whole-method comparison.

Original SphereDiff FLUX also completed, preserving its native 2048 × 4096 output at `outputs/bridge-factorial-ruins/20260918/references/flux/`. It used the official pinned FLUX.1-dev checkpoint, BF16, variant None, 28 steps, guidance 3.5, true CFG 1.0, 26,500 spherical points, temperature 0.1, original local defaults, seed-0 CUDA generator, and no CPU offload or VAE tiling. Generation took 1890.135 s, with 2492 transformer calls, 35.665 GiB peak allocated and 37.061 GiB peak reserved GPU memory. Actual transformer-input shapes are recorded in metadata. Both original reference source snapshots remained unchanged.

### Pilot completion and release of the remaining matrix

All **20 pilot cells completed on their first attempts**, were visually reviewed, and are retained as final matrix cells. Numerical audit job `19784481` passed: all source hashes, scientific settings, camera geometry, initialization digests, transition flags, artifact contents and model-call counts match the frozen design. The pilot used exactly **56,960 guided model calls**. Each latent cell used two VAE encodes per camera/timestep, one clean decode per camera/timestep, plus 89 terminal decodes; PixelDiT used zero VAE calls. The remaining 60 cells were authorized for launch without scientific changes.

The pilot images contain substantial negative outcomes. Uniform pilots are often blurred, noisy or nearly structureless. Shared/standard/arithmetic/center pilots show more recognizable structures in all five backends, but these four-cell comparisons change multiple factors and do **not** identify isolated main effects. LPW/DPA center pilots are visibly softer than their standard/arithmetic counterparts. PixelDiT’s standard center pilot has harsh silhouettes and 12.27% out-of-range diagnostic RGB values, while its uniform pilot has high-frequency grain without recognizable ruins. These observations motivate cautious interpretation of HF and agreement metrics, not retuning. All factor settings remain fixed.

Only final PNGs and numerical metadata are retained per cell. Compact local-view PNG payloads inherited inside the reused V diagnostic metadata were removed as storage-only postprocessing; all numerical diagnostic values and scientific provenance remain unchanged.

### Observed factorial contrasts: SD2

These are descriptive contrasts over all 16 fixed seed-0 cells. Main effects are the factor-1 mean minus the factor-0 mean; two-factor interactions use the positive-product minus negative-product mean of the ±1 factor codes. An interaction is half the corresponding difference of simple effects. No p-values or population-level significance are claimed. Detail metrics below average the four fixed diagnostic views; normalized pair disagreement measures the selected aligned equatorial pair, not global semantic coherence.

| Term | Contrast change | HF1/std change | Normalized aligned-pair MAE change |
|---|---:|---:|---:|
| A | +0.066777 | +0.009235 | -0.037819 |
| B | -0.026361 | -0.015693 | +0.034694 |
| C | +0.026539 | +0.008768 | +0.035681 |
| D | +0.083159 | +0.006105 | -0.001550 |
| A×B | -0.014236 | -0.004161 | -0.025755 |
| A×C | +0.006133 | +0.000861 | -0.027130 |
| A×D | -0.023432 | +0.000792 | +0.001138 |
| B×C | -0.006804 | +0.001991 | +0.024165 |
| B×D | -0.014949 | -0.003452 | +0.009223 |
| C×D | -0.013999 | -0.005726 | -0.005063 |

Shared initialization improves contrast and normalized HF while reducing normalized pair disagreement on average. Center weighting has the largest contrast gain, but is not the largest normalized-HF main effect. LPW decreases normalized HF in the full matrix, and matched image pairs are visibly softer or ghosted. DPA raises HF but also average pair disagreement; turquoise/colored blobs are conspicuous in several independent-initialization cells. Its positive HF effect must not be read as uniformly better detail. The negative C×D HF interaction means DPA’s incremental HF gain is smaller with center weighting; the positive B×C disagreement interaction indicates a larger DPA disagreement cost under LPW.

All 16 native ERP previews were reviewed as matched D0/D1 pairs within each A/B/C setting. `A1B0C0D1` offers a cleaner-looking observed detail/agreement tradeoff; `A1B0C1D1` has higher HF1/std (0.051272 versus 0.047709) but worse normalized pair MAE (0.063326 versus 0.047230). Neither is a universal quality optimum.

### Scheduling adjustment after measured pilots

The longest pilot allocations, including import/model startup, were 35.42 minutes for FLUX, 42.67 for SD3.5, and 21.00 for PixelDiT. To improve backfill eligibility, 27 still-pending remaining-wave jobs had only their Slurm time limits reduced: FLUX standard/LPW 55/60 minutes, SD3.5 65/70 minutes, and PixelDiT 35/40 minutes. Every limit retained at least 14 minutes beyond the corresponding backend’s longest pilot allocation. No job was canceled or resubmitted, and no prompt, seed, model, precision, geometry, step count, bridge, transition, factor, or output setting changed. Per-job old/new limits are recorded in `execution.json`.

### Observed factorial contrasts: SANA

| Term | Contrast change | HF1/std change | Normalized aligned-pair MAE change |
|---|---:|---:|---:|
| A | +0.001442 | -0.000116 | -0.028112 |
| B | -0.021591 | +0.008004 | +0.066821 |
| C | -0.030938 | +0.017645 | +0.066383 |
| D | +0.321709 | +0.003903 | -0.045893 |
| A×B | +0.001189 | -0.003885 | -0.013644 |
| A×C | +0.005128 | -0.002426 | -0.018123 |
| A×D | -0.015431 | -0.000878 | +0.013383 |
| B×C | -0.011889 | +0.020591 | +0.066868 |
| B×D | -0.023226 | -0.016429 | -0.049705 |
| C×D | +0.070884 | -0.012983 | -0.052650 |

Center weighting is the dominant contrast and visual-structure factor. Shared initialization reduces normalized pair disagreement, while its average normalized-HF effect is nearly zero; the previews nevertheless show more recognizable structures under shared standard/uniform fusion. The low-level average does not capture that semantic difference.

The positive average B effect on normalized HF must **not** be interpreted as an LPW detail improvement. The B×C interaction is +0.020591: LPW’s simple HF effect is -0.012586 with arithmetic fusion but +0.028595 with DPA. All 16 previews show that LPW/arithmetic smooths detail, whereas LPW/DPA with uniform weights introduces grain and streaked texture. `A0B1C1D0` has the highest HF1/std (0.073956) but very poor normalized pair MAE (0.352526) and an indistinct dark scene. It is a useful metric-extreme diagnostic, not a best-quality configuration.

DPA also has a strong C×D contrast interaction: it darkens uniform cells while increasing contrast with center weighting. Standard center-weighted cells produce recognizable mossy terrain and star fields; DPA makes their colors more saturated, and LPW counterparts are softer. Shared/standard/center arithmetic remains a useful conservative visual baseline; the standard DPA/center cell has lower normalized pair MAE in this seed but is more saturated. Original SphereDiff remains a separate whole-method reference with different local spherical state dimensions and aggregation.

### Observed factorial contrasts: FLUX

| Term | Contrast change | HF1/std change | Normalized aligned-pair MAE change |
|---|---:|---:|---:|
| A | +0.055516 | +0.008633 | -0.021383 |
| B | -0.050161 | +0.000267 | +0.030216 |
| C | +0.002607 | +0.019258 | +0.036895 |
| D | +0.207137 | -0.021793 | -0.055222 |
| A×B | -0.043077 | -0.019120 | -0.010388 |
| A×C | -0.002825 | -0.008421 | -0.027865 |
| A×D | -0.049362 | -0.003058 | +0.025515 |
| B×C | +0.006533 | +0.017509 | +0.031076 |
| B×D | +0.013400 | -0.008824 | -0.018995 |
| C×D | +0.048766 | -0.014488 | -0.033064 |

Center weighting gives the largest contrast gain and reduces normalized pair disagreement, while its negative HF effect reflects suppression of artifact-heavy uniform outputs. Shared initialization raises average contrast and HF, but the complete previews show dense colored speckles in shared standard/uniform cells. These are not convincing ruin detail. With center weighting, the simple A effect on normalized pair MAE is slightly adverse (+0.004132), even though the average A effect is negative. Shared initialization is therefore not a universal visual improvement for FLUX in this study.

The nearly zero average B HF effect (+0.000267) masks the strong B×C interaction (+0.017509). LPW’s simple HF effect is -0.017242 with arithmetic fusion and +0.017777 with DPA; the latter includes pronounced grain in uniform cells. Center-weighted LPW counterparts are softer or ghosted. The negative C×D HF interaction and positive C×D contrast interaction again distinguish artifact HF from useful structure.

All 16 native previews were reviewed. Independent standard center-weighted `A0B0C1D1` is a useful observed detail/agreement candidate (normalized pair MAE 0.034463); `A0B0C0D1` is a less saturated alternative. Shared standard center cells have somewhat sharper-looking features but retain conspicuous bright sky artifacts. Original SphereDiff’s native preview has more naturally resolved stone structures and ground; the final quantitative comparison uses matched perspective rasters. The highest-HF cell, `A0B1C1D0` (HF1/std 0.100759), is dark and artifact-dominated and must not be presented as a perceptual optimum.

### Frozen backend settings and metric interpretation

| Backend | Steps | Guidance | Model/VAE precision | Prepared scheduler | VAE tiling | Local RGB / clean ERP |
|---|---:|---:|---|---|---|---|
| SD2 | 30 | 7.5 | FP16 | DDIMScheduler | on | 512² / 512×1024 |
| SANA | 20 | 4.5 | BF16 | DPMSolverMultistepScheduler, order 1, flow sigmas | off | 1024² / 1024×2048 |
| FLUX | 20 | 3.5, true CFG 1 | BF16 | FlowMatchEulerDiscreteScheduler | on | 1024² / 1024×2048 |
| SD3.5 | 40 | 4.5 | BF16 | FlowMatchEulerDiscreteScheduler | off | 1024² / 1024×2048 |
| PixelDiT | 50 | official CFG 2.75 | BF16 model, no VAE | PixelDiTFirstOrderSolver | N/A | 1024² / 1024×2048 |

All local native states and bridge arithmetic use FP32. All initialization scales are 1.0. PixelDiT uses the existing official solver configuration, full [0,1] guidance interval and negative prompt `low quality, worst quality, over-saturated, blurry, deformed, watermark`; the generic generation guidance field of 1.0 is not its actual CFG scale. Exact prepared timestep/sigma arrays, SD2 coefficients, negative prompts, source revisions and common-setting hashes are retained in the manifest and per-cell metadata. Scientific parameters were fixed before pilot images were inspected.

Pinned model revisions are SD2 `f5bc1bd97485577aa0b946fa8a9004e2ec147402`, SANA `e2b3c0cbffebcd09d83805e88b9f5f106afc74ac`, DiffPano FLUX `fa45a9eb6808ba8fdfc7cc2756f7f1a16e0921f4`, and SD3.5 `b940f670f0eda2d07fbb75229e779da1ad11eb80`. PixelDiT retains official source commit `41f73006ae532b0b41fee72b181dc22891a5a01a` and the cached `pixeldit_t2i_v1.pth` checkpoint. Original SphereDiff FLUX uses official revision `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`; original SANA uses the same SANA revision as DiffPano. Both environments have Diffusers 0.32.2, Transformers 4.49.0, Accelerate 1.4.0 and Safetensors 0.8.0.

The contrast tables above use HF1 divided by view standard deviation. Absolute HF1 is also retained and must be examined separately: LPW's absolute-HF main effect is negative in SD2, SANA and FLUX (-0.004874, -0.001314 and -0.003855), despite positive normalized-HF effects in SANA and FLUX. Center weighting increases absolute HF in SD2 and SANA (+0.003820 and +0.009633) but decreases it in FLUX (-0.001506), where uniform outputs contain substantial artificial texture. Therefore center weighting's strong visual benefit does not imply that it maximizes every HF proxy.

Normalized aligned-pair agreement is also distinct from raw agreement. D's raw pair-MAE effect is +0.004620 in SD2 and +0.015259 in SANA, despite negative normalized effects; contrast increases explain part of that difference. In FLUX, shared initialization's normalized-pair effect is negative but its raw-pair effect is +0.004161. These results support qualified statements about the measured normalized proxy, not blanket improvements in global scene coherence. Full-camera view-to-consensus error, fixed camera-boundary gradient ratio, and ERP-wrap ratio remain separately reported; none is a semantic scene-coherence oracle.

The bridge and exact directional prompt are common to every factorial cell, so this design does not estimate a bridge-on/off or directional-prompt-on/off effect. In particular, a comparison with historical L–V changes more than one condition and cannot identify a causal bridge improvement. The supported question is how A/B/C/D behave **with** the local bridge and current-state transition held fixed. PixelDiT supplies a VAE-free within-backend factorial, but differences between its effects and latent-model effects cannot be attributed solely to the VAE. Its model, solver, native grid and learned image distribution also differ.

### Observed factorial contrasts: PixelDiT (VAE-free)

| Term | Contrast change | HF1/std change | Normalized aligned-pair MAE change |
|---|---:|---:|---:|
| A | +0.204283 | +0.062540 | -0.043820 |
| B | -0.163880 | -0.009010 | +0.030946 |
| C | +0.036885 | +0.034750 | +0.016609 |
| D | +0.260512 | -0.142000 | -0.146918 |
| A×B | -0.111191 | -0.018191 | +0.009858 |
| A×C | +0.007180 | -0.020459 | -0.019711 |
| A×D | -0.142660 | -0.039155 | +0.051964 |
| B×C | -0.005997 | +0.043335 | +0.017010 |
| B×D | +0.045097 | -0.013799 | -0.004812 |
| C×D | +0.020130 | -0.028777 | -0.016101 |

All 16 native ERP previews were reviewed. Center weighting restores recognizable ruins in both initialization conditions and reduces artifact-heavy high-frequency energy. Shared standard/uniform outputs contain severe black/white speckling and clipping: diagnostic out-of-range fractions are 18.45% for arithmetic and 24.67% for DPA. Shared standard/center outputs show clearer structures but retain harsh black outlines and about 12.27–12.42% out-of-range values. Independent standard/center outputs are less extreme, at 0.78% and 1.34% for arithmetic and DPA. These fractions are measured on diagnostic RGB before PNG clipping, not the proportion of clipped pixels in the saved PNG.

A's positive contrast/HF means therefore do not establish better perceptual detail. Its mean normalized-pair effect is favorable, but A×D is +0.051964; with center weighting, the simple shared-initialization effect is adverse (+0.008145). Its full-camera view-to-consensus main effect is also adverse (+0.012938). The strong negative A×D contrast interaction means shared initialization contributes much less additional contrast when center weighting is already active.

LPW's average normalized-HF effect is negative, and center-weighted counterparts are visibly blurry/ghosted. The positive B×C HF interaction (+0.043335) reverses LPW's simple HF effect from -0.052345 with arithmetic to +0.034325 with DPA, including grain in uniform outputs. DPA's mean normalized-pair cost is +0.016609; C×D is -0.016101, so that cost is largely concentrated in uniform-weight cells. The negative C×D HF interaction likewise reduces the extra artifact HF under center weighting.

`A0B0C0D1` is a conservative observed PixelDiT candidate: recognizable but imperfect ruins, normalized pair MAE 0.035405, and 0.78% out-of-range diagnostic RGB. `A0B0C1D1` lowers that pair metric to 0.028922 but increases saturation/out-of-range values to 1.34%. Shared standard center cells have more aggressive fine edges, but their severe clipping prevents interpreting those edges as an unqualified detail improvement. LPW trades much of the clipping for blur. These effects occur without a VAE, demonstrating that blur, grain, clipping and factor interactions in this study need not originate in VAE reconstruction.

### Observed factorial contrasts: SD3.5

| Term | Contrast change | HF1/std change | Normalized aligned-pair MAE change |
|---|---:|---:|---:|
| A | -0.000119 | +0.005956 | +0.001369 |
| B | -0.026349 | +0.010241 | +0.030294 |
| C | -0.032673 | +0.027044 | +0.024936 |
| D | +0.259027 | -0.012373 | -0.015210 |
| A×B | +0.002109 | -0.008452 | -0.009925 |
| A×C | +0.009400 | -0.000489 | -0.006203 |
| A×D | -0.016835 | -0.004708 | +0.000272 |
| B×C | -0.011752 | +0.023934 | +0.023488 |
| B×D | -0.010870 | -0.014995 | -0.013531 |
| C×D | +0.069392 | -0.017242 | -0.016268 |

All 16 native ERP previews were reviewed. Center weighting produces recognizable columns/temples and terrain in both A conditions, while uniform independent outputs are murky and shared uniform outputs retain sky speckles and fragmented architecture. A has essentially zero average contrast effect, positive normalized HF, and slightly worse normalized pair disagreement (+0.001369); its raw pair and full-camera disagreement effects are also adverse. Shared initialization is not an overall agreement improvement for SD3.5 in this seed.

D remains the strongest contrast main effect (+0.259027), but decreases normalized HF (-0.012373) while increasing absolute HF (+0.003044). LPW center counterparts are visibly soft. The positive B normalized-HF main effect again hides an interaction: B×C is +0.023934, with B's simple effect -0.013694 under arithmetic and +0.034175 under DPA. LPW/DPA uniform images contain grain, and their average disagreement is worse. LPW's absolute-HF main effect is effectively zero (+0.000018), not evidence of retained recognizable detail.

DPA darkens uniform outputs and saturates center outputs, reflected in C×D contrast +0.069392 and HF -0.017242. Independent standard DPA/center `A0B0C1D1` has 13.63% out-of-range diagnostic RGB versus 0.96% for its arithmetic counterpart. The DPA cell's lower normalized pair MAE (0.014683 versus 0.017756) therefore comes with substantial clipping. `A0B0C0D1` is a conservative detail/agreement candidate. `A1B0C0D1` is an alternative with lower out-of-range fraction (0.32%) but worse normalized pair MAE (0.023044). Neither establishes globally consistent 3D architecture.

### Runtime, memory and measured operation costs

All 80 cells completed on NVIDIA A100 GPUs. These are observed run costs, not controlled throughput benchmarks; node startup, file-system load, VAE conventions and native dimensions differ. `runtime_seconds` covers the dense pipeline including terminal assembly/diagnostics, while `total_seconds` additionally includes model preparation and initialization after imports. Slurm allocation time includes batch/import startup.

| Backend | Mean pipeline min (range) | Mean total min | Max GPU allocated / reserved GiB | Max host RSS GiB | Guided calls, 16 cells |
|---|---:|---:|---:|---:|---:|
| sd2 | 6.00 (5.46–6.63) | 9.26 | 2.459 / 2.994 | 6.690 | 42720 |
| sana | 13.59 (13.08–14.25) | 16.47 | 5.807 / 8.469 | 7.197 | 28480 |
| flux | 28.76 (28.12–29.48) | 32.17 | 25.016 / 26.951 | 10.392 | 28480 |
| sd35 | 36.53 (35.67–37.49) | 39.89 | 7.496 / 9.328 | 17.594 | 56960 |
| pixeldit | 13.47 (12.86–14.32) | 17.20 | 4.160 / 4.721 | 13.105 | 71200 |

Mean denoising-loop stage seconds per cell follow. These timers omit some preparation, terminal work and diagnostics, so their sum is not the pipeline total.

| Backend | Model | Decode | Local round-trip encode | Synchronized encode | Warp/fusion |
|---|---:|---:|---:|---:|---:|
| sd2 | 83.75 | 106.11 | 57.60 | 56.35 | 23.18 |
| sana | 128.33 | 245.86 | 184.01 | 174.90 | 27.89 |
| flux | 1010.31 | 307.63 | 157.29 | 157.90 | 29.00 |
| sd35 | 799.76 | 604.20 | 313.86 | 314.84 | 55.44 |
| pixeldit | 607.70 | N/A | N/A | N/A | 60.29 |

The 80 cells used **227,840 guided predictions and the same number of actual transformer forwards**, with no repeated predictions for diagnostics. Their summed pipeline duration is 26.224 hours; summed scientific Slurm allocation time is **121,587 seconds (33.774 hours)**. The two original references add **4,272 forwards** and **2,837 seconds (0.788 hours)** of GPU allocation, for **34.562 GPU allocation-hours across all 82 scientific runs**. Validation, five zero-denoiser preflights and CPU reports are accounted separately in `execution.json`.

The local bridge requires one clean decode and two encodes per view/timestep, plus 89 terminal decodes per latent cell. These costs are substantial, especially in SANA; PixelDiT has zero VAE calls and all VAE timing fields are not applicable. Mean independent/shared initialization times are SD2 1.886/0.333 s, SANA 0.362/0.260 s, FLUX 1.555/1.069 s, SD3.5 1.516/0.962 s, and PixelDiT 25.576/15.944 s. No quality setting was reduced for runtime.

### Resolution-matched original SphereDiff comparison

The native outputs are untouched. The reference sheet projects both methods into the same saved cameras 7 and 59, with **1024×1024 local rasters and 80°×80° FOV**. Metrics in the following table are means over those two full perspective images after loading the saved PNGs into RGB [-1,1]. They are distinct from the raw four-view factorial diagnostics.

Representative selection was made after the full factorial analysis: the `detail` row maximizes HF1/std among cells with at most 1% out-of-range diagnostic RGB; the `agreement` row minimizes normalized aligned-pair MAE among cells at or above backend-median contrast. For both FLUX and SANA, the `detail` row is **A0B1C1D0, an artifact-dominated metric extreme, not a perceptual winner**. This deliberately exposes the failure of a scalar HF ranking. Agreement selects FLUX A0B0C1D1 and SANA A1B0C1D1. Those are descriptive tradeoff choices, not universal optima.

| Backend / method | Mean contrast | Mean absolute HF1 | Mean HF1/std |
|---|---:|---:|---:|
| flux / detail A0B1C1D0 | 0.124095 | 0.013222 | 0.110736 |
| flux / agreement A0B0C1D1 | 0.443090 | 0.007043 | 0.016225 |
| flux / original SphereDiff | 0.442428 | 0.032780 | 0.074089 |
| sana / detail A0B1C1D0 | 0.089371 | 0.006867 | 0.077100 |
| sana / agreement A1B0C1D1 | 0.536881 | 0.014328 | 0.026775 |
| sana / original SphereDiff | 0.448643 | 0.017321 | 0.038601 |

The reviewed FLUX matched views show more naturally resolved stone blocks, arches, ground debris and fine star texture in original SphereDiff. DiffPano A0B0C1D1 is recognizable and avoids the shared cells' conspicuous confetti, but its masonry and upper-sky detail remain soft. A0B0C0D1 is a less saturated alternative, and shared standard/center cells show sharper-looking architecture at the cost of bright sky artifacts. None of the reviewed DiffPano FLUX cells convincingly matches the original reference's combined local detail and scene appearance.

For SANA, A1B0C1D1 is the selected agreement candidate; A1B0C0D1 is the more conservative saturation alternative. Original SphereDiff shows finer sky/ground texture and fuller column structures, with some haze or smeared edges. DiffPano has stronger blue/green saturation and ridged, simplified ground. The original looks more naturally detailed overall in these views, but neither this visual assessment nor two view metrics establish globally correct 3D geometry.

These are whole-method comparisons. The original uses persistent spherical native state, point-based sampling/aggregation and different initialization. FLUX also uses 28 rather than 20 steps; original ERP resolution is twice the DiffPano height/width; SANA's actual spherical local grids differ from the nominal call dimensions. Matching diagnostic raster/FOV removes the direct raw-pixel-resolution comparison error, but does not equalize the information present in native outputs. FLUX source IDs/revisions differ: matching component configs and seven cached weight content addresses/sizes provide strong evidence, but no independent full weight-byte rehash was performed. Seed 0 is deterministic within each implementation, not paired initialization across methods. Original references are excluded from every factorial contrast.

Artifacts: [FLUX/SANA matched comparison](../../outputs/bridge-factorial-ruins/20260918/spherediff-comparison.png); original [FLUX ERP](../../outputs/bridge-factorial-ruins/20260918/references/flux/final_result.png) and [SANA ERP](../../outputs/bridge-factorial-ruins/20260918/references/sana/final_result.png).

### Cross-backend answers and next research configuration

The findings below are **observed factorial contrasts for this controlled seed-0 study**. They do not supply p-values, population-level generalization, or a universal best configuration.

1. **Matched shared initialization:** normalized selected-pair disagreement improves on average in SD2, SANA, FLUX and PixelDiT, but slightly worsens in SD3.5. Raw pair error decreases only in SD2/SANA; it increases in FLUX/SD3.5/PixelDiT. Shared standard PixelDiT and FLUX cells demonstrate that positive HF effects can reflect artifacts.
2. **LPW with the bridge active:** center-weighted images are visibly softer across all five backends. Absolute-HF main effects are negative in four backends and effectively zero in SD3.5. Positive normalized-HF means in SANA/FLUX/SD3.5 are driven partly by LPW×DPA grain, not established detail preservation.
3. **DPA with the bridge active:** normalized HF rises in every backend, but so does normalized selected-pair disagreement. Saturation, clipping and coefficient-level grain qualify the nominal detail benefit. DPA is an optional tradeoff, not the default recommendation.
4. **Center weighting:** D has the largest positive contrast main effect in every backend and the clearest visual benefit for recognizable ruins. It does not maximize HF: removing uniform-cell grain decreases normalized HF in FLUX/SD3.5/PixelDiT. Thus it remains the strongest observed structural/contrast factor, not a universal scalar detail optimum.
5. **A×D:** normalized-pair interactions are positive in all five backends, especially FLUX (+0.025515) and PixelDiT (+0.051964). Shared initialization's normalized agreement benefit weakens when center weighting is active; the D1 simple effect is adverse in FLUX and PixelDiT.
6. **C×D:** normalized-HF and normalized-pair interactions are negative in every backend. Center weighting reduces DPA's incremental artifact HF and disagreement cost. In SANA/FLUX/SD3.5/PixelDiT, positive contrast interactions also show stronger DPA contrast changes under center weighting. SD2's contrast interaction is negative.
7. **B×C:** normalized-HF interactions are positive in every backend, particularly SANA, FLUX, SD3.5 and PixelDiT. In those four, LPW lowers normalized HF with arithmetic but raises it with DPA, accompanied by visible grain in uniform cells. These interactions explain why marginal B effects alone mislead.
8. **VAE-free evidence:** PixelDiT reproduces the weighting benefit, LPW blur and LPW×DPA artifact interaction, as well as sharpness/clipping tradeoffs. Those behaviors do not require a VAE. Cross-model differences still do not isolate VAE causation.
9. **Detail versus coherence:** shared standard uniform FLUX/PixelDiT and LPW/DPA uniform cells are examples of high HF with poor recognizable scene structure. DPA center cells can improve normalized pair error while worsening clipping. Low disagreement in dark or blurred cells is not evidence of useful global coherence.
10. **Closest original-reference candidates:** FLUX A0B0C1D1, with A0B0C0D1 as a less saturated alternative; SANA A1B0C1D1, with A1B0C0D1 as a less saturated alternative. Original references remain more naturally detailed overall in the reviewed matched views; no single DiffPano factor explains the difference.

Seam proxies reinforce the need for separate outcomes. LPW increases the ERP-wrap gradient ratio on average in all five backends. Center weighting lowers the fixed camera-boundary gradient ratio in all five, but its ERP-wrap ratio effect is adverse in PixelDiT. These ratios describe gradient concentration, not semantic continuity, and can change with texture or blur. Full-camera disagreement and absolute/normalized aligned-pair errors are retained separately in the machine-readable summary.

**Recommended next baseline:** retain standard warp B0, arithmetic fusion C0 and center weighting D1, with the same local bridge/current-state framework and fixed angular cover. Use A1 for SD2/SANA as a useful observed baseline, and A0 for FLUX/SD3.5/PixelDiT as the conservative artifact/agreement choice. These are starting configurations for future independently authorized research, not population-level winners. Preserve C1 center variants as explicit saturation/detail tradeoffs for FLUX/SANA, rather than enabling DPA universally. No additional generations, seeds, prompts, parameter tuning or bridge-off experiments were run in this task.

### Final completion and preservation audit

**Completed: 80/80 DiffPano cells plus 2/2 original SphereDiff references. Failed: 0; skipped: 0; pending: 0.** All 82 scientific generations succeeded on their first attempts. The 20 pilot results were reused, with exactly 60 additional matrix generations and no scientific retries or tuning. Every study Slurm job, including validation/preflight/report jobs, finished with `COMPLETED` and exit code `0:0`.

Final report job **19784607** passed the 80-cell audit: resolved configs and non-factor scientific settings, source hashes, exact prompt, common angular camera digest, schedules, bridge/transition flags, A0/A1 matched maps, source release, artifact budgets, finite diagnostics and **227,840** guided/actual-forward counts. All 1,647 historical artifact sizes and SHA-256 hashes were verified unchanged. The original report prefix, pinned manifest, validated scientific sources and original SphereDiff snapshot hashes also passed preservation checks. Branch remains `no_sphere`, HEAD remains `8bf6d13257a32db5ee279ca973cde2dadc9beb25`; dirty preexisting work is preserved and no commit was made.

Maximum recorded current-state reconstruction errors across each full matrix are SD2 2.8610e-6, SANA 2.3842e-7, FLUX 3.0547e-7, SD3.5 1.1921e-7 and PixelDiT 5.9605e-8. These are separate from the real-VAE bridge identity oracle above. The tests and runtime audit retain the current-state transition; stored pre-fusion endpoints do not determine the next state.

All 80 native images, all five 4×4 contact sheets, and both original-reference matched-camera comparisons received visual review. The two inherited metadata corrections are transparent: embedded PNG preview payloads were removed without changing numerical diagnostics, and original SANA's effective solver-order metadata was corrected from mapping value 2 to the actually used attribute value 1, retaining the evidence. No scientific source or generation was changed by either correction.

Study artifacts total **183 files, 157,585,143 bytes (150.285 MiB)**: 88 PNGs, 93 JSON files, one CSV and the launcher lock. Each of the 82 scientific output folders contains exactly `final_result.png` and `metadata.json`; no noise, native-state, residual, per-step or pyramid tensors/images are retained. Associated Slurm logs add **182 files and 1,534,022 bytes**. Combined artifacts plus logs are **159,119,165 bytes**; source/model caches and temporary display helpers are excluded.

The main artifacts are [manifest.json](../../outputs/bridge-factorial-ruins/20260918/manifest.json), [factor-summary.csv](../../outputs/bridge-factorial-ruins/20260918/factor-summary.csv), [factor-summary.json](../../outputs/bridge-factorial-ruins/20260918/factor-summary.json), and [execution.json](../../outputs/bridge-factorial-ruins/20260918/execution.json). The summary includes all four main effects and all six two-factor interactions for every scalar metric in the summary table, separately by backend.

Contact sheets: [SD2](../../outputs/bridge-factorial-ruins/20260918/sd2-factorial.png), [SANA](../../outputs/bridge-factorial-ruins/20260918/sana-factorial.png), [FLUX](../../outputs/bridge-factorial-ruins/20260918/flux-factorial.png), [SD3.5](../../outputs/bridge-factorial-ruins/20260918/sd35-factorial.png), [PixelDiT](../../outputs/bridge-factorial-ruins/20260918/pixeldit-factorial.png), and [original SphereDiff comparison](../../outputs/bridge-factorial-ruins/20260918/spherediff-comparison.png).

## Selected Ruins Cells at 2048×4096 ERP

This follow-up changes only the clean RGB ERP raster to **height 2048, width 4096**, including the projection/fusion/coverage/diagnostic grids derived from that raster. It reuses 22 requested cells from the completed bridge factorial. Every resolved scientific config is compared with its completed baseline and must differ at exactly `erp.height` and `erp.width`.

| Backend | Requested ABCD codes | Runs |
|---|---|---:|
| FLUX | 0001, 0101, 1001, 1011, 1101, 1111 | 6 |
| PixelDiT | 1001, 1011, 1111, 0001 | 4 |
| SANA | 0001, 0011, 1001, 1011, 1111 | 5 |
| SD2 | 1001, 1011, 1111 | 3 |
| SD3.5 | 1000, 1001, 1011, 1010 | 4 |

The common saved 89-camera cover, camera ordering and every **80° horizontal × 80° vertical frustum** remain unchanged. SD2 keeps 512×512 local RGB; the other backends keep 1024×1024. Native dimensions, seed 0, exact five-line ruins prompt, checkpoint revisions, precision, guidance, prepared schedules, step counts, local bridge, current-state transition, A/B/C/D meanings, DPA parameters and five-level LPW settings remain identical to each corresponding baseline cell. The native noise ERP dimensions depend on the native raster/FOV, not clean RGB ERP size, and therefore also remain unchanged. Initial local state hashes must match each paired baseline exactly.

The isolated harness lives in `studies/bridge_erp4k/`, with output root `outputs/bridge-erp4k-ruins/20260918/`. It imports the unchanged validated generation runner into a private module and supplies only study I/O, resolution-config and saved-camera verification hooks; it does not modify the original module or pipeline algorithms. Existing scientific code and all completed study artifacts are preserved. The original L geometry's raster-specific lookup is verified at its original raster, and full coverage is checked again at the new ERP raster using those exact saved cameras. No camera cover is regenerated.

Validation includes four focused resolution/subset/invariance tests, the existing 226-test regression suite, and five real-backend preflights. Each preflight probes full-size standard and LPW DPA/center fusion with all 89 cameras, checks returned local dimensions, repeats the existing bridge/noise checks without denoising, and requires the schedule, conditioning, noise-map and initial-state metadata to match the completed baseline. All 22 actual generations retain their original steps and model-call budgets, totaling 59,630 guided predictions.

This is a selected-cell paired resolution comparison, not another complete 2^4 matrix. No new factorial main effects/interactions will be inferred from the incomplete subset. Comparisons use the unchanged local diagnostic camera rasters/FOV, retaining native output PNGs. Raw ERP pixel-frequency and fixed-pixel seam metrics at differing ERP resolutions are not directly comparable. No additional original SphereDiff runs are needed; both existing original references already have 2048×4096 ERP outputs.

CPU validation job **19786132** passed compilation, whitespace checks, all **4 focused tests** (19.817 s), and all **226 regression tests** (117.707 s). Total allocation duration, including environment/import startup, was 10m48s. The frozen 22-cell manifest contains only the two allowed config changes; the angular geometry SHA remains `b55b4eb5051cdfb8c579e81d4f51ece1f15aeb0e8d7e1fde6b7b74a49154435c`. GPU preflight jobs are 19786158–19786162.

All **22 requested generation jobs** are submitted with Slurm `afterok` dependencies on all five preflights. The runtime independently enforces every passed preflight and matching source hashes before generation. Paired analysis/preservation job **19786195** depends on all 22 generations. No outputs are yet claimed complete. Pending preflight walltimes were reduced from 35 to 20 minutes for backfill eligibility, leaving over 11 minutes beyond the longest historical preflight; GPU type, memory and scientific settings were unchanged.
