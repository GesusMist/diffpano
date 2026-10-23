FLUX 28-step refinement study

Outputs: outputs/refinement28-flux48/seed0-v1
Scope: 24 existing prompts (original 20 plus followup24 NEW4), two runs each.
Each run uses 28 total scheduler steps, including the independent refinement
tail. The two exact fractions are 5/28 (23 coupled + 5 independent steps) and
8/28 (20 coupled + 8 independent steps). Never round these fractions: the
pipeline uses ceil when converting a fraction to a number of intervals.

Shared settings follow the previous gradient_refinement study: saved Old89
cameras, FLUX.1-dev, seed 0, 4096x2048 ERP, poisson_select, screened reference
with lambda_color=0.1. No camera rotation, time travel, or gradient mixing.
Existing model weights, prompt bytes, and historical experiment outputs are
reused as inputs without modification. All 48 output trajectories are new.

Preparation runs in a CPU Slurm allocation through the established wrapper:
  bash studies/all_prompts/environment.sh diffpano -m studies.refinement28.prepare
It verifies the real 28-step schedule, coverage, configuration, and CPU tests,
then freezes the 48 cases and source hashes in manifest.json. The frozen code
is also archived in implementation.tar.gz. Do not edit its dependencies while
jobs are running. No per-view or per-step image/tensor dumps are generated.

Inspect the submission plan:
  bash studies/all_prompts/environment.sh diffpano -m studies.refinement28.launch --dry-run
Submit the authorized study:
  bash studies/all_prompts/environment.sh diffpano -m studies.refinement28.launch --execute

The two air_balloons cases are full output-producing profile runs. They count
toward the 48 images. The remaining 46 cases depend on both succeeding, with
at most two A100 jobs running concurrently. Each GPU case has a 90-minute cap,
4 CPU cores, and 64 GiB host memory. At verified FASTER rates the generation
reservation is at most 10,512 SUs, plus 5.33 SUs for the CPU completion report.
The approximate 5,829 SU runtime estimate is an extrapolation; actual elapsed
charges and remaining allocation are recorded by the cluster.

Each case stores final.png, unclamped FP32 output, terminal weighted-RGB
reference products, configuration, progress, and verified completion metadata.
The automatic completion report creates completion.json, completion.csv, and
24 original-resolution paired figures when both corresponding runs validate.
The terminal RGB references are diagnostics, not additional generated cases.

Submission intents are journaled before sbatch. If submission is interrupted,
inspect launch.json and the scheduler before any retry; never blindly resubmit.
Case locks and artifact hashes prevent duplicate execution and image overwrite.
