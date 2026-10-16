120 ordinary independent FLUX20 perspective images
================================================

Twenty original scene files (native_control excluded), four prompts/ournew files,
five unchanged lines each. Each line is a separate 1024x1024 native trajectory
with batch_size=1 and a fresh seed-zero CUDA generator. Labels identify intended
directions only; generation creates no camera. Exact assigned texts are encoded
before the shared model's text encoders are released. The scheduler is rebuilt
from its own immutable config and prepared at 1024x1024 before every image.

Only ordinary native Gaussian initialization, twenty native denoise steps and
one terminal VAE decode occur. Physical calls are instrumented; VAE encode fails
immediately. No panorama loop, GWTF, warp, bridge, fusion, gradient selection,
late refinement, or perspective metrics are used.

Audited settings come from the actual completed gwtf-erp4k FLUX20 metadata,
restricted to local model/guidance/precision/decoder and native scheduler data.
This does not assert the same initial state as GWTF or SphereDiff.

Prepare after all source is final:
  /usr/bin/python3 -m studies.perspective24.prepare
  sbatch studies/perspective24/cpu.slurm studies.perspective24.validate all
Then one real target slot as a bounded measured GPU profile:
  sbatch studies/perspective24/generate.slurm --indices 0 --profile
Once profile succeeds, controller may group remaining indices in allocations:
  sbatch studies/perspective24/generate.slurm --indices 1-30
  /usr/bin/python3 -m studies.perspective24.status

The parent followup24 controller owns the shared maximum TWO generation GPU
allocations across A and B, accounts for existing jobs, and submits explicit
index groups. Do not independently submit A and B arrays with separate %2 caps.
Use --retry explicitly only for failed/interrupted cases; valid results are
content-verified and reused. Per-case nonblocking file locks prevent duplicates.
No old scientific source or artifacts are modified. Source and input artifacts
are frozen, archived, and verified for every case. One model load is shared by
all unfinished cases assigned to a job. Logs and status stay suite-local.
