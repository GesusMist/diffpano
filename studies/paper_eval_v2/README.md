# Reproducible panorama evaluation

The active run is `outputs/paper-eval-v2/20261026-core-v1`. Its `RESUME.md`,
`scheduler.json`, `controller/state.json`, and per-case metadata are authoritative
for remaining work. The run directory resolves to home storage; historical
scratch outputs remain unchanged.

The user selected the core generation tier and explicitly authorized OpenRouter
provider-managed spending limits without additional local request/USD caps.
Only private API workers load the credential file. The shared SQLite API ledger
retains every attempt, including uncertain outcomes, and valid observations are
reused. API request bodies and authorization headers must never be logged.

Inspect the run:

```bash
cd /home/shig/diffpano
source studies/paper_eval_v2/clean_environment.sh
bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2 --run outputs/paper-eval-v2/20261026-core-v1 status
```

After a finite controller stops, the following command reconciles the prior
epoch and starts a new one. It refuses to duplicate an active controller and
preserves completed batches and scientific/API blockers:

```bash
bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.controller resume --run outputs/paper-eval-v2/20261026-core-v1
```

Generation, native references, metric calculations, and reporting run in finite
Slurm allocations through `resources.submit`. The shared limit is two generation
GPUs and one evaluation GPU. Controller workers load no API credential. Completed
files are verified before indexing; a submitted or finished Slurm job alone is
not a completed scientific result.

Frozen generation, renderer, evaluator, and API sources must not be edited while
this run uses them. Use an explicit new version for scientific changes. New
integration/report adapters can read those frozen implementations. In particular,
`prepare_production` validates canonical JSON equality with the frozen pilot;
the original VLM preparation function compares saved lists with runtime tuples.

The initial development pass completed DS, CS, and MUSIQ for all 417 selected
historical cases. Development, confirmation, real-reference, native-reference,
and legacy results have separate cohorts or namespaces. All 24 real-scene
captions were reviewed before confirmation generation. Missing capture regions
at the poles of ODI-SR031 and036 are documented in `dataset/capture_coverage.json`.

The VLM repeat rule uses a deterministic hash threshold with nominal probability
0.05. It selected 73 of the initial 1,218 view images (5.99%); both judges repeat
each selected image, yielding 146 additional requests. These observations assess
repeat variability and never replace primary scores.

The human viewer remains offline and requires actual collected ratings. Browser
fixtures, when present, are synthetic validation data in a separate directory,
not study responses.

Refresh a report through a finite CPU job (no API calls):

```bash
bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2.snapshot --launch --run outputs/paper-eval-v2/20261026-core-v1
```

The native-reference banks are now verified complete (240/240). If a future run
is interrupted, `python -m studies.paper_eval_v2.resume_native --run RUN` submits
only groups with unfinished verified images and refuses overlapping generation
allocations. Add `--dry-run` to inspect without submitting.

The confirmation controller now submits sequential arrays of at most six images,
retaining the frozen four-hour task limit and at most two generation GPUs. Each
batch reserves its full Slurm walltime plus the next evaluation stages. It skips
verified outputs and never automatically repeats a submitted scientific failure.

`distribution_resume.py` continues only incomplete linked-reference uncertainty:
it preserves prior bootstrap draws and the exact RNG sequence, checkpoints new
draws, and runs through `distribution_resume.slurm` in a finite CPU allocation.
