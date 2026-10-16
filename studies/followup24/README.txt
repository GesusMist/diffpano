FLUX FOLLOW-UP24: two separate output types

A: outputs/10.15perspective24-flux/seed0-v1
120 standalone1024x1024 images: 24 exact prompt files x 5 unchanged directional lines.
Seed0 is reset freshly for every image. Exactly20 native intervals, zero VAE image encodes,
one terminal decode. No panorama construction, camera processing, GWTF or metrics.

B: outputs/10.15layout24-flux216/seed0-v1
216 logical ERP4096x2048 slots: 24 prompts x9 variants. Reuse is decided by actual bytes,
archived provenance and semantic compatibility, never current whole-repo source equality.
See reuse_audit.json/csv and generation_plan.json for all candidate decisions.
All new generations retain fixed80-degree views and GWTF/native bridge semantics;
only Old89/Uniform89 have screened+select, lambda.1, rtol1e-5, atol1e-7, cap200.
No refinement, other gradient modes, new seeds, rotations, or extra configurations.

Preparation commands (refuse to overwrite frozen manifests):
  python3 -m studies.perspective24.prepare
  sbatch studies/perspective24/cpu.slurm studies.perspective24.validate all
  sbatch --dependency=afterok:<CPU_ID> studies/perspective24/generate.slurm --indices 0 --profile
  sbatch studies/layout24/prepare.slurm

Before generation, both manifests and their source archives must exist, and A's real slot0
profile must pass. The shared launcher runs A groups first, then four actual missing B
profile slots covering RGB counts89/70/50 and screened89, then the remaining B slots.
All generation arrays use at most2 jobs and successive stages are dependency ordered;
never submit the standalone and panorama generate arrays independently with %2.
The first generation stage waits for existing GPU allocations; unrelated jobs are preserved.

  python3 -m studies.followup24.manage status
  python3 -m studies.followup24.manage launch --dry-run
  python3 -m studies.followup24.manage launch
  python3 -m studies.followup24.manage launch --retry

The launcher rejects duplicate active follow-up jobs. --retry explicitly permits archiving
failed/interrupted case files before retrying only missing slots. Valid results are skipped.
Source freeze mismatches must be investigated, never bypassed by rewriting historical gates.

Evaluation and figures:
  sbatch studies/layout24/metrics_cpu.slurm
  sbatch studies/layout24/evaluate.slurm --allow-partial
  sbatch studies/layout24/report.slurm

The launcher schedules evaluation/reporting after the generation arrays. An initial partial
evaluation should run alongside the one-GPU perspective profile once B preparation passes.
Per-case immutable snapshots and an exclusive evaluator lock avoid concurrent writers.
Cached Inception/CLIP features are reused only after matching image, prompt and protocol
identities and verifying their hashes. DS/CS use the unchanged display PNG/evaluator.
IS is recomputed separately for each24-panorama variant; original20/new4 subsets are
separate160/32-view collections. Partial groups have null aggregate scores.
Results are evaluation/{per_image_scores,summary_scores,subset_scores}.csv,
evaluation/metric_protocol.json, results.tex, report.md and figures/<prompt>.png.
Timing excludes setup where recorded; missing historical timing remains missing.

Submission is not completion. Consult status and metrics before reporting a complete suite.

Allocation-limited launch on 2026-10-15:
The full-array attempt was rejected by Slurm: project132684267659 had77.841030SUs
and the original conservative A array requested108.0SUs. Measured A job limits were
reduced to45minutes without changing image settings. Array20153291 schedules119
remaining A outputs with2 concurrentjobs. Panorama array20153349 schedules only
actual missing indices2,3,4,5,6,8 after A, with1-hour limits and2 concurrentjobs.
These six complete the air_balloons nine-variant group. Evaluation20153350 and
report20153351 follow. Another115panorama slots remain unsubmitted pending funding.
Full B estimated generation cost45.46GPU-hours excludes model setup and queues.
See orchestration/{allocation-status,execution-status,workload-estimate}.json.
Do not mistake the prepared121-row missing plan for submitted or completed work.
