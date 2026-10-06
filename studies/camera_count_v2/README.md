# ERP camera-count study v2

The authorized matrix is 108 DiffPano + 18 reduced-ring SphereDiff = **126 runs**.
The later user instruction fixes **SphereDiff FLUX to 20 steps**, overriding the
28-step attachment preset. SANA also uses 20. DiffPano uses FLUX20, SANA20,
PixelDiT50, SD3.540 with its historical checkpoints and schedules.

Prompts: firework, underwater, ruins. Counts: 70, 50, 30. Output: 4096×2048 ERP.
Generation seed 0, batch 1, 80°×80° fixed cameras. No CEA, SD2, LPW, DPA,
time travel, training, or metric-model evaluation.

| Strategy | DiffPano | SphereDiff | Total |
|---|---:|---:|---:|
| old | 36 | 18 | 54 |
| fibonacci | 36 | 0 | 36 |
| random | 36 | 0 | 36 |
| Total | 108 | 18 | 126 |

Each camera count has 36 DiffPano + 6 SphereDiff = 42 rows.
SphereDiff Fibonacci/random rows are not present in the manifest or dependencies.

## Layouts

North-to-south latitudes: +90,+67.5,+45,+22.5,0,-22.5,-45,-67.5,-90.
N70 counts: 3,7,8,11,12,11,8,7,3.
N50 counts: 2,5,6,7,10,7,6,5,2.
N30 counts: 1,3,4,4,6,4,4,3,1.
Each ring uses yaw=-pi+2*pi*j/m, roll=0; order is north to south then yaw.
These are regenerated user-specified rings, not old89 subsets or official
published reduced-count presets. The pinned old89 counts are 4,8,11,14,15,14,11,8,4.
Polar yaw is retained as full pose identity.

Fibonacci reuses the generic complete-N generator at phase 0.
Random candidates reuse the CPU FP32 row-wise uniform-area sampler, seeds
0,1,2,... independently for each N. The first candidate passing frozen geometry
and actual DiffPano fusion support is accepted. The resulting distribution is
**coverage-conditioned**, not unrestricted iid. It is not selected using images.
Accepted seeds, sequential rejection counts, poses, and hashes are recorded once
in outputs/camera-count-erp-seed0-v2/layouts. Interrupted search resumes at next_seed.

Coverage uses 8,192 then 65,536 early-rejection probes, 1,000,000 deterministic
spherical probes, exact poles, 8,196 wrap-boundary directions, all 2048×4096 ERP
texel centers, and actual production FP32 RGB denominators. These are finite
numerical guarantees, not a proof over all continuous directions. Fixed-layout
failure blocks only affected configurations. There is no gray-hole fallback.
Only old/rings undergo SphereDiff adapter and final RGB assembly checks.

## Methods

DiffPano preserves GWTFlow source grid/density/noise seed, Gaussian transport,
FP32 native states, BF16 models, standard bilinear view-to-ERP and nearest
ERP-to-view, center-weighted arithmetic RGB averaging at temperature0.1, local
VAE residual bridge and current-state-preserving transitions. PixelDiT stays
VAE-free. Shared initialization hashes must reproduce across all three prompts.

SphereDiff is labeled **SphereDiff — reduced ring cameras**. A process-local
context manager changes only its camera factory and restores it in finally.
The canonical pose converts to theta=wrap(pi-yaw),phi=-pitch. The conversion is
derived from native extraction and final ERP pasting, including their sign
conventions. Polar yaw survives FP32/BF16 via positive FP64 cos(pi/2).
The original dynamic sampling, spherical aggregation, initialization, packing,
scheduler and final decoder remain intact. SANA keeps center_first=True and
2600 points; FLUX keeps center_first=False and 26500 points. All N views reach
prompt lookup, each denoising interval, and the terminal decoder. Unused latent
points are reported independently from RGB coverage; every decoded point must
have been updated. No GWTFlow or RGB residual bridge is added to SphereDiff.

Both FLUX methods now use20 steps, but their model sources remain separately
pinned (ModelsLab versus Black Forest Labs), and native raster workloads differ.
Numerically equal seeds do not imply paired cross-method noise. Equal camera
counts or steps do not imply equal FLOPs. N89 references remain contextual only;
no N89 scientific images are generated.

## Validation and execution

Entry points: audit, validate, random_search, preflight, run, launch, report.
The CPU audit verifies actual completed N89 configs, source hashes, models,
prompts, original geometry and SCRATCH symlink. Geometry checks precede image
generation. A full-schedule single-interval real-backend preflight is required
for each authorized layout/backend. The first required firework run is the full
pilot; remaining prompts require its numerical completion, never visual approval.

The dispatcher limits **all** GPU jobs, including geometry, random search,
preflight and both method families, to five in total. SphereDiff old jobs never
depend on random search. Each sample runs in a fresh process in its original
environment. Bounded random batches resume without repeating completed seeds.
Infrastructure retries preserve scientific parameters. There are no quality retries.

Only final.png and config.json are saved per scientific sample. Shared layout,
gates, statuses, and durable submission records are compact JSON. The matching
PNG/config and successful completion record are required together; two renames
are not treated as an atomic pair. Existing results and historical sources are
preserved. Outputs are written through the current SCRATCH symlink. No commit,
push, shared environment upgrade, or large intermediate artifacts.
