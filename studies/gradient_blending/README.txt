ERP gradient-domain blending, SANA and FLUX pilot

New operator: diffpano/gradient_fusion.py. Study-only canvas and pipeline
subclasses are in operator.py. No historical file is modified. The original
ordinary_run, interval implementation, bridge, noise initialization, checkpoints,
weights, schedules, prompts, and frozen evaluator remain active.

Modes: rgb (exact original reducer); poisson_mean (weighted gradient control);
poisson_select (maximum geometric edge-confidence source, stable camera IDs).
Objective and boundary conventions are documented in gradient_fusion.py.
Default lambda is 0.1 for both variants; PCG 200 iterations, rtol 1e-5, atol 1e-7.
Production arithmetic is FP32 outside autocast. This is an ERP pixel-grid
objective, not a spherical-gradient objective. All finite/coverage checks remain.

Outputs: outputs/10.6gradient-blending/seed0-v1
Pilot: exact cached SANA and FLUX, Old89, ERP 4096x2048, local RGB 1024x1024,
20 steps, seed 0, original ruins/underwater/firework prompts. No wider sweep.
At most two gradient generation jobs execute concurrently.

Run each stage only after the previous gate is successful, from repo root:
  sbatch studies/gradient_blending/validate.slurm
  sbatch --job-name=grad-smoke --time=00:20:00 studies/gradient_blending/gpu.slurm studies.gradient_blending.gpu_smoke
  sbatch --job-name=grad-ruins-rgb studies/gradient_blending/gpu.slurm studies.gradient_blending.run --prompt ruins --mode rgb --instrument
  sbatch --job-name=grad-replay --time=00:30:00 studies/gradient_blending/gpu.slurm studies.gradient_blending.replay
  python3 -m studies.gradient_blending.launch --pilot
  sbatch studies/gradient_blending/evaluate.slurm
  sbatch studies/gradient_blending/figures.slurm

CPU and GPU numerical work runs in Slurm. The launch command only validates
file hashes, copies cached controls with provenance, and submits an array.
The validated SANA underwater/firework and all three FLUX RGB images are reused. SANA ruins is
regenerated once to collect replay statistics and counts as the sixth control.
The latest user request explicitly adds FLUX: 18 logical cases in total, with
12 new gradient trajectories and one combined array capped at two GPUs.
SANA supplies the required offline replay of the model-independent operator.
FLUX has an additional checkpoint/configuration/cached-control audit:
  sbatch studies/gradient_blending/prepare_flux.slurm
Run this before the combined launch. FLUX outputs are in seed0-v1/flux.
To evaluate or render FLUX after generation, submit its script with:
  sbatch --export=ALL,DIFFPANO_GRADIENT_BACKEND=flux studies/gradient_blending/evaluate.slurm
  sbatch --export=ALL,DIFFPANO_GRADIENT_BACKEND=flux studies/gradient_blending/figures.slurm
Raw snapshots cover steps 1/10/20, retaining consolidated reference/guidance/
support/owner fields plus four selected decoded views and small warped crops.
Estimated snapshot budget is below 2.5 GiB; no all-camera ERP stack is stored.

Terminal RGB reference images are assembly of that run's terminal states, not
independent baseline trajectories. Fixed report crops/views and common [-1,1]
display conversion are used; evolving generation is never clamped.

All new generation cases check original call counts and provenance. Captured
statistics and previews make no additional denoiser or VAE calls. The instrumented
RGB control excludes diagnostic callback/statistics time from generation timing;
peak allocation includes those diagnostics. Missing historical generation-only
timing remains unavailable rather than substituted with a broader runtime.

Validation logs, source preservation, baseline hashes, synthetic target errors,
replay diagnostics, case configs/metadata/status, evaluator results, figures,
and report.txt are retained under the isolated output root. Historical benchmark
folders and frozen evaluator caches are not written.

Explicit PixelDiT and SD3.5 extension (2026-10-06)
------------------------------------------------
The user additionally authorized these same three prompts and fusion modes for
PixelDiT and SD3.5. Preserve their actual benchmark schedules: PixelDiT 50 steps
with zero VAE calls; SD3.5 40 steps with the existing local VAE residual bridge.
All geometry, GWTF initialization, 4096x2048 ERP, checkpoints, guidance and
lambda0.1 remain unchanged. This adds 18 logical cases, reusing six RGB controls
and generating twelve trajectories, at most two generation GPUs concurrently.

  sbatch studies/gradient_blending/prepare_extension.slurm
  # After extension-validation.json passes:
  python3 -m studies.gradient_blending.launch_extension

The launcher submits the generation array plus dependent evaluation, figures
and reports. It refuses duplicate active submissions. The extension CPU gate
executes full 50/40-interval tiny-raster mock trajectories through the actual
runner schedule helper, unchanged ordinary_run and study pipeline; checks RGB
identity, bridge/current-state invariants, call counts and all fusion stages.
The original 293-test gate, GPU smoke and offline replay apply to the unchanged
operator. Original reports and source are retained under history/.

Outputs: seed0-v1/pixeldit and seed0-v1/sd35, each with cases/, evaluation/,
figures/ and report.txt. extension-submission.json records Slurm dependencies.
The old both-models-report.txt and completion.json remain the SANA/FLUX snapshot.
New four-models-report.txt and extension-completion.json summarize all 36 cases.
Use python3 -m studies.gradient_blending.extension_status for current progress.
Intermediate previews remain bounded to executed steps1/10/20, with full final
ERP and fixed-view/detail comparisons for all three prompts. No new full RGB
trajectory is generated solely to provide intermediate baseline diagnostics.
