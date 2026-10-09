DiffPano gradient-reference / selection / final-refinement ablation

Public config: global_pipeline.refinement.last_fraction (default 0.0).
CLI: python scripts/generate.py --config <existing.yaml> --refinement-last-fraction 0.1
All original schedule intervals execute. ceil(f*N) final intervals retain native
local states, with no intermediate VAE/warp/fusion. Dynamic layouts freeze once.
Terminal representation stays pipeline-specific (native MultiDiffusion averages
native patches before one global decode; RGB modes fuse decoded terminal views).
Previously unsupported backend/pipeline combinations remain unsupported.

Study: FLUX; seed0; saved Old89; 80x80 degree frusta; local RGB1024/native128;
ERP H2048xW4096; GWTF; original FLUX20 schedule/checkpoint/prompt routing/bridge.
4 prompts x 4 references x 2 gradients x 2 fractions = 64 logical cases.
Four audited screened/select/off cases are copied with historical provenance;
60 missing cases are generated. Historical outputs and gates are immutable.

D is pixel-unit forward difference, periodic longitude and open latitude.
Egrad = sum_e M(e)||D_e I-g_e||^2, using same-source valid endpoints.
select: largest geometric confidence. max: largest RGB vector L2 magnitude,
without confidence weighting; copy whole signed vector; stable ID breaks ties.
screened: Egrad + .1||I-R||^2, exact old Jacobi-PCG200 path.
global_mean: Egrad, mean(I)=mean(R), zero-mean nullspace projection, no ridge.
single_pixel: Egrad, I(p0)=current source c0 at the same ERP ray.
coarse_color: Egrad + .1 sum_b n_b ||B_b(I-R)||^2, mean(I)=mean(R).
B uses 16x32 exact nonoverlapping blocks; these are fixed initial choices.
New references: projected PCG <=400, exact even-vertical-extension FFT
preconditioner, rtol1e-5/atol1e-7, FP32; constraints and actual output normal
equations checked. A shift in the coarse PRECONDITIONER is not an objective
ridge. Full pixel coverage and connected supported-edge graph are mandatory.

Commands (from repository root):
  sbatch studies/gradient_refinement/validate.slurm
  sbatch --time=00:20:00 studies/gradient_refinement/gpu.slurm studies.gradient_refinement.gpu_smoke
  sbatch studies/gradient_refinement/check.slurm -m studies.gradient_refinement.prepare
  python3 -m studies.gradient_refinement.launch                  # dry run
  python3 -m studies.gradient_refinement.launch --submit --representative
  python3 -m studies.gradient_refinement.launch --submit --after <representative-job>
  python3 -m studies.gradient_refinement.status
  python3 -m studies.gradient_refinement.launch --submit --retry # explicit failed/incomplete retry
  sbatch --dependency=afterok:<array-job> studies/gradient_refinement/evaluate.slurm
  sbatch --dependency=afterany:<evaluation-job> studies/gradient_refinement/finish.slurm

Maximum two concurrent generation GPUs. Wall limit 3h per case includes margin:
historical FLUX ~30m; measured 4K coarse solver <1s after warmup on saved real
proposals (49 iterations), pure solver 2 iterations. No resolution/accuracy cut.

outputs/10.8gradient-refinement-flux/seed0-v1 contains manifest, reuse audit,
validation logs, source archive, per-case raw FP32 images/PNG/config/provenance,
operation counts/solver diagnostics, fixed-view/detail/wrap figures, frozen
panorama metrics, paired deltas and four-prompt descriptive aggregate/report.
The four historical cases have no unclamped FP32 artifact: this is explicit.

Historical root test test_historical_initializer_and_source_unchanged pins the
September dense-loop source bytes. Requested loop changes invalidate that pin.
The old TT metadata certificate also correctly rejects changed generation code.
Both historical tests/gates are untouched. The new runner replaces these two
source-certificate tests with initializer source+behavior, archived disabled-path
bitwise tests, and a check that the historical metadata certificate still rejects
our changed source. All behavioral regressions remain enabled.
