# Original SphereDiff references

This isolated study audits four requested seed-0 original SphereDiff entries:
SANA and FLUX, each on the unchanged ruins and underwater five-line prompts.
It uses the pinned official classes from
/home/shig/diffpano_reference_sources/spherediff-2c8c68b
(commit 2c8c68ba088f2803b3dce4b52b7b0d68bc996139).
There is no DiffPano generation import or method reimplementation.

config.yaml is JSON-compatible YAML. Both original ruins references are reused
only after strict matching; underwater runs use the verified local prompt file.
The immutable manifest, audit, durable submission ledger and outputs live in
outputs/original-spherediff/ruins-underwater-seed0-v1/.

Use the existing spherediff environment. The Slurm scripts preserve its proven
activation/cache pattern; no shared package installation or upgrade is needed.

Commands:
    python -m studies.original_spherediff.audit
    python -m studies.original_spherediff.launch validate --submit
    python -m studies.original_spherediff.launch generate --submit
    python -m studies.original_spherediff.launch status
    python -m studies.original_spherediff.launch report --submit

The launch command inspects the ledger, statuses, squeue and sacct, then prints
the dry-run plan before submission. At most two missing GPU cases are allowed.
Ambiguous or failed recorded submissions are never blindly duplicated.
Each new generation uses a fresh process, an exclusive per-case lock and atomic
completion status. Existing attempt directories are never overwritten.

The four requested CLI cases are supported:
    python -m studies.original_spherediff.run --backend sana --prompt ruins
    python -m studies.original_spherediff.run --backend sana --prompt underwater
    python -m studies.original_spherediff.run --backend flux --prompt ruins
    python -m studies.original_spherediff.run --backend flux --prompt underwater

Reused entries perform no generation. New entries require a GPU Slurm allocation
and the passed focused validation gate. Validation runs compileall, diff --check,
the CPU provenance audit and study-local unit tests; it loads no model weights
and performs no scientific generation.

SANA retains 20 steps, guidance 4.5, nominal 1024-square arguments, 2600 points.
FLUX retains 28 steps, guidance 3.5, true CFG 1.0, 26500 points, official defaults.
Both are BF16, temperature 0.1, offload/tiling disabled, final ERP 4096x2048.
Actual dynamically sampled model-input shape histograms are recorded; nominal
height/width never force a fixed patch size.

The official static launcher sets the SANA config.solver_order attribute to 1,
although the serialized mapping remains 2. Both values and actual schedules
are recorded. FLUX keeps its preparatory generator-consuming draw before the
spherical draw. Global RNG seeding precedes loading; a fresh CUDA generator is
created after loading, matching the historical wrapper's order.

No TT/CEA code, manifests, outputs, historical validation hashes, original
checkout, or frozen snapshot is modified. No metrics, method tuning, training,
new DiffPano runs or additional 20-step FLUX case are included. Final output
images are unmodified official images[0]; the overview is visualization only.
