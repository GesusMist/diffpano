Task B: 24 scenes x exactly nine FLUX panorama variants (216 logical slots).

Output: outputs/10.15layout24-flux216/seed0-v1

Preparation (CPU allocation, clean login environment):
  sbatch --parsable studies/layout24/prepare.slurm --validate-only
  sbatch --parsable studies/layout24/prepare.slurm

The full prepare performs the read-only historical artifact audit before freezing
and adopting accepted original images. It does not submit GPU jobs. The shared
followup24 manager owns the total two-generation-GPU budget across both tasks;
never launch this suite with its own independent %2 array throttle alongside A.

A single missing index, dispatched under the shared budget:
  sbatch --parsable studies/layout24/generate.slurm --index INDEX
An interrupted/failed index requires the explicit --retry flag. Completed outputs
are content-verified and reused. Reused cases retain immutable original metadata
and a suite-local image symlink; they are not attributed to the new revision.

Scientific invariants:
  FLUX pinned existing checkpoint, BF16, seed0, 20 intervals, ERP4096x2048.
  DiffPano local1024x1024/native128x128, FOV80x80, current-state VAE bridge,
  standard warp, spherediff_center weighted_average reference, temperature0.1.
  Fixed Old89, historical regenerated Old70/50, separate phase0 Uniform89/70/50.
  Canonical noise-slot labels and separately mapped stable integer gradient owners.
  Refinement last_fraction0; no time travel or camera rotation.
  RGB at all six banks; screened+poisson_select only Old89/Uniform89.
  Screened lambda0.1, rtol1e-5, atol1e-7, cap200; gradient_mix_alpha absent.
  SphereDiff uses the unchanged original matched20 loader, spherical noise,
  native fusion, original cameras, and terminal decode.

Six focused CPU tests cover exact scope, geometry/routing, confidence selection,
RGB/select shared geometry, physical call counts, and actual GWTF native draws
for all six camera banks matched to historical local/source-noise hashes.
Archived full-resolution coverage gates are reused only for their exact saved
camera banks, with projection/warp/fusion/noise implementation hashes checked.
Generation asserts positive support on every interval; screened solves verify
connected supported-edge graphs and convergence on all21 stages.

A production profile occupies an actual missing logical slot and is never rerun
as an extra experiment. Every new DiffPano generation verifies counts20*N model,
40*N encode,21*N decode,21 fusion, and0 RGB/21 screened Poisson solves. Newly
produced DiffPano images retain the standard unclamped FP32 final ERP artifact.
