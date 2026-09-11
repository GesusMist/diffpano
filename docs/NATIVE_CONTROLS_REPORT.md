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
