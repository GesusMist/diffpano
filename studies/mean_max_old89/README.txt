Run from /home/shig/diffpano on Grace. All output is suite-local under
outputs/mean-max-old89-flux90/seed0-v1 (the existing outputs scratch symlink).

Development and freeze (never overwrite an existing manifest):
  sbatch studies/mean_max_old89/cpu.slurm studies.mean_max_old89.validate tests
  sbatch studies/mean_max_old89/cpu.slurm studies.mean_max_old89.prepare
Fresh gates after prepare:
  sbatch studies/mean_max_old89/cpu.slurm studies.mean_max_old89.validate tests
  sbatch studies/mean_max_old89/cpu.slurm studies.mean_max_old89.validate weights
  sbatch studies/mean_max_old89/metrics_cpu.slurm
  sbatch studies/mean_max_old89/smoke.slurm diffpano studies.mean_max_old89.gpu_smoke

After all four gates have passed, run one real matrix row (index0) and its
partial evaluation/report, then validate its actual profile before launching:
  python3 -m studies.mean_max_old89.manage profile --dry-run
  python3 -m studies.mean_max_old89.manage profile
  python3 -m studies.mean_max_old89.manage status
  python3 -m studies.mean_max_old89.manage launch --dry-run
  python3 -m studies.mean_max_old89.manage launch

The launcher requires the first row to complete with exact hashes/counts and
uses conservative 3h walltime, prior actual runtimes, and the measured mixed row.
The array has at most two generation GPUs and waits for unrelated active GPU jobs.
The evaluator and report depend on generation/evaluation, respectively.

Explicit failed/interrupted retries (wait for active suite jobs to finish):
  python3 -m studies.mean_max_old89.manage launch --retry
Standalone refresh with completed or partial results:
  sbatch studies/mean_max_old89/evaluate.slurm --allow-partial
  sbatch studies/mean_max_old89/report.slurm

No legacy hashes or source freezes are rewritten. Do not modify shared source
while this frozen suite has active jobs. Existing legacy modes keep their old
serialization shape via GradientSettings.to_record(); missing alpha means a
legacy mode, except explicit poisson_mean_max defaults to max coefficient 0.5.
