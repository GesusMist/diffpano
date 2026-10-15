Missing Old89 controls for the 15-prompt mean-max report
======================================================

This isolated suite changes no frozen implementation or historical artifact.
It accepts only explicitly missing prompt/reference pairs, not all 30 controls.
It uses the exact validated mean-max Old89 configuration, geometry, 80-degree
cameras, GWTF native initialization, seed 0, FLUX 20-step schedule, ERP 4096x2048,
1024x1024 view raster, and refinement last_fraction=0. The only solver changes
relative to the corresponding mean-max row are mode='poisson_select' and the
absence of gradient_mix_alpha. The historical select method uses geometric
confidence (the report display alias is select_closest), not magnitude max.

Production solver policies are unchanged: screened rtol=1e-5 and cap=200;
coarse_color rtol=1e-4, 16x32 blocks, eta=.1 and constrained cap=400; atol=1e-7
and screened lambda=.1. No retries change those settings.

Prepare input format, either a list or an object with this key:
  {"missing": [{"prompt": "aurora", "reference_mode": "coarse_color"}]}

From a clean login environment (do not submit from an activated model venv):
  mkdir -p outputs/10.14mean-max-old89-controls/seed0-v1/logs/{validation,generation}
  sbatch studies/mean_max_controls/prepare.slurm --missing /absolute/input.json

After preparation succeeds, submit the manifest row indices as an array with
at most two concurrent tasks, waiting for the existing mean-max generation,
exact-policy retry, evaluation and reporting chain. No helper auto-submits.
Example with 15 missing rows and a final dependency job ID:
  sbatch --dependency=afterany:JOB --array=0-14%2 studies/mean_max_controls/generate.slurm

The preparation freeze verifies all old source hashes and validation gates,
then compares each new configuration/geometry/solver against its frozen
mean-max counterpart. It records the exact missing input, source archive and
15-group GPU evidence identities. Production checks actual conditioning,
initialization and schedule against that evidence; each successful control
requires 21 converged solves and 1780 denoiser / 3560 encode / 1869 decode calls.

Artifacts and statuses live under:
  outputs/10.14mean-max-old89-controls/seed0-v1/cases/
    prompts/<prompt>/<reference>/poisson_select/refine000/
  (new prompt identities retain prompts/ournew/<prompt>).

Run the login-safe inventory with:
  python -m studies.mean_max_controls.status [--verify]

An existing incomplete/failed case requires an explicit --retry, which archives
its previous status, metadata, progress and config without changing policy.
Never modify this source directory after preparation freezes it.
