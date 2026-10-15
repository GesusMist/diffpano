Authorized Firefly tolerance retry, 2026-10-15

Exactly original mean-max case 27 is generated again: Firefly, Old89, FLUX,
coarse_color, poisson_mean_max alpha .25, seed 0, 20 intervals, refine000.
Only solver.relative_tolerance changes from 1e-4 to 2e-4. Absolute tolerance
1e-7 and constrained iteration cap 400 remain unchanged. No automatic further
relaxation is allowed. All 21 Poisson fusions must converge under this setting.

The original failed case, original frozen manifest and original source files
are preserved. New outputs live in outputs/10.15mean-max-firefly-retry/seed0-v1.
The retry retains the logical case_id, with new semantic and source digests.
Metadata records the user's explicit authorization and both tolerance values.

The runtime bookkeeping wrapper is byte-identical to the original wrapper
except its common/configuration imports. The existing ordinary_run loop,
GWTF initializer, denoiser, geometry, prompt, guidance and image saving paths
are reused. New CPU checks validate the one-field change and wrapper identity.
Original immutable CPU/GPU/weight certificates are reused only for unchanged
components; the manifest states this reuse explicitly.

Submit from a clean login shell, after creating logs/validation and
logs/generation below the output directory:
  sbatch --parsable studies/mean_max_firefly_retry/cpu.slurm
  sbatch --parsable --dependency=afterok:CPU_JOB studies/mean_max_firefly_retry/generate.slurm

Preparation freezes sources and writes a source archive. Do not edit frozen
sources afterward. Render the report using the retry only after complete(row)
passes, while retaining source and tolerance provenance for that panel.
