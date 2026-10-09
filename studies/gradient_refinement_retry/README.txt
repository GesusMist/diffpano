Authorized retries of five FLUX coarse-color failures: indices 12,14,28,29,30.
Relative tolerance changes from 1e-5 to 1e-4; absolute tolerance stays 1e-7.
All other numerical/generation settings and original source files stay frozen.
The retry runner is a separately fingerprinted copy of the original runner,
reading GradientSettings from each validated retry row rather than defaults.
Both PCG convergence and final FP32 output-residual/constraint checks remain.

Plan, original manifest, failed attempts, and new validation are under:
outputs/10.8gradient-refinement-flux/seed0-v1/retries/rtol-1e-4
Preparing the plan does not change the active manifest. CPU validation must pass
before --activate updates only the five failed rows and adds a retry note.
All 59 other rows and top-level original source hashes remain untouched.

python3 -m studies.gradient_refinement_retry.prepare
sbatch --job-name=gradref-retrycheck studies/gradient_refinement/check.slurm -m studies.gradient_refinement_retry.validate
python3 -m studies.gradient_refinement_retry.prepare --activate
sbatch --job-name=gradref-retry --array=12,14,28,29,30%2 --dependency=afterany:<original-array> studies/gradient_refinement/gpu.slurm studies.gradient_refinement_retry.run --retry

Only submit once and record the array under the original OUT/submissions for
normal status discovery. Wait for the original array to finish, irrespective of
its five previous failures. This preserves the two-generation-GPU limit.
Replace blocked original reporting dependencies with evaluation after successful
retries and factual reporting after those jobs. The retry report wrapper explicitly
records the tolerance exceptions in both the numerical summary and text report.
