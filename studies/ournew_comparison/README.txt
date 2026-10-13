Exact four-prompt FLUX28 comparison
==================================
See REQUEST.txt for authorized scope. Outputs: outputs/10.9ournew-flux28/seed0-v1.
No historical case fills this manifest. Shared native trajectory and numerical
solvers are reused; official.py changes only the official loader's validation gate.
No SphereDiff numerical source is edited.

Default GradientSettings(reference_mode='single_pixel') now uses fused_rgb_erp,
semantics version 2. Explicit legacy_source_patch reproduces patch-color anchoring.
GradientSettings.from_record() treats missing source as historical version 1.
The new suite CLI exposes --single-pixel-source but requires fused_rgb_erp for its
fixed manifest; the legacy setting does not add a control trajectory to this suite.

Preparation and gates (submit only after earlier recovery/evaluation finish):
  python3 -m studies.ournew_comparison.apply_anchor
  sbatch studies/ournew_comparison/cpu.slurm studies.ournew_comparison.prepare
  sbatch studies/ournew_comparison/cpu.slurm studies.ournew_comparison.validate tests
  sbatch studies/ournew_comparison/cpu.slurm studies.ournew_comparison.validate weights
  sbatch studies/ournew_comparison/metrics_cpu.slurm
  sbatch studies/ournew_comparison/smoke.slurm diffpano studies.ournew_comparison.gpu_smoke
  sbatch studies/ournew_comparison/smoke.slurm spherediff studies.ournew_comparison.official_smoke
Run prepare once. Tests and weight audit may run independently after preparation;
both GPU smokes require those gates. At most two GPU smokes concurrently, and
account for unrelated active allocations. Never overwrite a frozen production
manifest or historical gate. A failed development gate requires inspecting the
error and preparing fresh gates before any production submission.

Status / submission / resume from /home/shig/diffpano:
  python3 -m studies.ournew_comparison.manage status
  python3 -m studies.ournew_comparison.manage launch --dry-run
  python3 -m studies.ournew_comparison.manage launch
  python3 -m studies.ournew_comparison.manage launch --retry --dry-run
  python3 -m studies.ournew_comparison.manage launch --retry
The manager verifies gates, case identities and hashes, refuses duplicate active
suite jobs, submits only missing rows with array throttle %2, then real dependent
DS/CS/IS and report jobs. --retry does not alter numerical tolerances or geometry.

Evaluation can also be explicitly refreshed after all dependencies finish:
  sbatch studies/ournew_comparison/evaluate.slurm
  sbatch studies/ournew_comparison/report.slurm
Use --allow-partial only for provisional evaluation; incomplete collections have
null aggregates. Per-image CSV always retains all 28 rows. Metric caches include
image, prompt, full configuration/source and protocol identity and stay local.

Wall limits: generation 3 hours (prior 20-step FLUX runs approximately 35 minutes;
new bounded 4K solver and one-interval model profiles are retained in validation).
GPU smoke 90 minutes; evaluation 2 hours. No reduced dimensions or steps.
Figures use seven fixed columns with unchanged PNG exposure, fixed native-detail
crops, lower-pitch views and explicit longitude-seam crops for every method.
The final audit requires 28 generation + 28 DS + 28 CS + 7 matched-collection IS.
