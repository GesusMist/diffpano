# SphereDiff-style GPT-4o evaluation

This isolated study evaluates existing panorama PNGs. It does not import diffusion pipelines, generate panoramas, or alter historical images. The first-stage preparation, tests, reports, and dry-run make **zero paid API requests**.

The primary comparison is fixed before scoring: official SphereDiff FLUX20, DiffPano RGB Old89, screened `poisson_select` Old89, and screened `poisson_mean_max` Old89 with alpha **0.50**. Here `lambda_blending` means `reference_mode="screened"`, and alpha is the **maximum-gradient coefficient**. Alpha 0.25 and 0.75 have separate predeclared paired cohorts. Official FLUX28 is a separately labeled additional baseline.

## Environment and commands

Use Python 3.10 or later. The validated environment uses Python 3.11.5; the three image-processing dependencies are pinned in `requirements.txt`. The API client uses the Python standard library and does not need the OpenAI SDK, PyTorch, a GPU, or model weights.

From the checkout root, in an environment with these packages:

```bash
python -m pip install -r studies/spherediff_vlm_eval/requirements.txt
python -m unittest discover -s studies/spherediff_vlm_eval -t . -p 'tests_*.py' -v

python -m studies.spherediff_vlm_eval inventory \
  --output outputs/10.16spherediff-vlm-eval/seed0-v1

python -m studies.spherediff_vlm_eval prepare \
  --output outputs/10.16spherediff-vlm-eval/seed0-v1

python -m studies.spherediff_vlm_eval dry-run \
  --output outputs/10.16spherediff-vlm-eval/seed0-v1 --scope primary
```

The existing local CPU environment can run the same commands with
`bash studies/panorama_metrics/environment.sh -m studies.spherediff_vlm_eval ...`.
That historical environment wrapper is specific to the current machine; the new evaluator itself derives the checkout root from its module path and accepts `--root`.

Run large inventory/preprocessing jobs inside a CPU allocation when using an HPC cluster. No generation jobs are submitted by this suite.

Explicit historical roots and relocation mappings are supported:

```bash
python -m studies.spherediff_vlm_eval prepare \
  --root /path/to/current/diffpano \
  --historical-root /path/to/FASTER/historical-outputs \
  --path-map /old/checkouts/diffpano/outputs=/path/to/FASTER/historical-outputs \
  --output outputs/spherediff-vlm-eval/run-v2
```

`--historical-root` and `--path-map` can be repeated. Full canonical paths below `prompts/`, including `prompts/ournew/`, and effective text SHA256 identify prompts. A matching basename alone is insufficient. Ambiguous relocated paths or different verified images for the same method/prompt are excluded and reported.

Preparation freezes the accepted inventory, cohorts, judge configuration, evaluator files, published prompt, and renderer identity. Re-running preparation resumes that inventory and validates cached views. Use a new output version when changing source selection, evaluator code, judge model, prompt, or sampling parameters. Existing generation source freezes are neither checked against current revisions nor rewritten; actual historical configuration and provenance are audited instead.

## Published protocol

The versioned prompt is `assets/spherediff_arxiv_v2_table5_prompt_v1.txt`. Its full wording was checked against **Table 5, PDF page 17**, of [SphereDiff arXiv v2](https://arxiv.org/pdf/2504.14396v2). PDF line wrapping and formatting are normalized; words, criteria, scale, instructions, response format, and the `{image}` position are retained. The asset provenance records the PDF and prompt hashes, source version, verification method, and CC BY 4.0 attribution. Images of PDF pages 17 and 19 are preserved alongside it.

The fixed criteria are Image Quality, Aesthetic Appeal, Distortion Level, and Connectivity. Each score must be an integer 1–5, with larger better. Reports call Connectivity **End Continuity** and Aesthetic Appeal **Aesthetic Appearance**, matching the requested table headings. Four brief explanations and the raw API response are retained.

Each request contains one image at the published `{image}` position and the fixed prompt text. No method names, filenames, generation parameters, generation prompt lines, previous scores, comparison images, or human ratings are sent. Human-facing comparison figures remain separate from API inputs.

Appendix F.2, PDF page 19, specifies fourteen directions and a 90° field of view:

- Yaw 0°, 90°, 180°, 270° at pitch −45°, 0°, and +45°.
- One north pole and one south pole, both at yaw 0°.

Output resolution **512×512** and pole yaw **0°** are fixed implementation choices requested for this evaluation. The unchanged `studies/panorama_metrics/render.py` performs pixel-centered perspective projection, periodic longitude sampling, and latitude clamping. Yaw 0° points to ERP longitude 0 at its horizontal center; yaw 180° places the ERP wrap boundary through the perspective center. The seam convention and polar orientation have analytic and synthetic-image tests. No ERP shifting, cropping, color correction, enhancement, sharpening, or repair is performed.

Rendered caches include original image hash, renderer/protocol hash, and image-library versions. Every cache hit rechecks all fourteen rendered image hashes and dimensions. Source bytes are verified before and after rendering. Evaluation cameras are separate from generation cameras.

## Inventory and cohorts

The exact 15 prompts are read from `studies/mean_max_old89/common.py`: 11 original paths and four `prompts/ournew/` paths. The scanner reads manifests, metadata/configuration, completion evidence, archived audits, image hashes, and actual PNG dimensions. It discovers historical output families instead of assuming layout24's 216 slots completed.

Eligible DiffPano inputs use FLUX, seed 0, 20 steps, ERP4096×2048, saved Old89 geometry, no refinement/time travel/camera rotation, and the requested fusion/reference/alpha. Official SphereDiff preserves its native representation, initialization, and cameras. Historical source revisions can differ when actual configurations and immutable artifacts remain compatible.

October 16 cleanup removed many per-case status files and tensor artifacts. Surviving PNGs can still be accepted when preserved completed audit records independently bind their image and metadata hashes, and actual execution/configuration checks pass. Evidence gaps and the separate archived FLUX28 provenance limitations are recorded explicitly. A filename or directory name alone never establishes completion.

The primary cohort is the intersection of the four main methods. `alpha025` and `alpha075` each intersect the primary methods with their additional alpha; this preserves direct pairing with alpha 0.50 and the two baselines. The `sphere28` cohort intersects the primary methods with the additional official FLUX28 baseline. No scores influence source selection, alpha choice, or cohort membership.

## Paid execution: next stage only

Set `OPENAI_API_KEY` in the process environment when paid evaluation is authorized. The key is never stored in study files or logs. The only supported requested model is exactly `gpt-4o`; there is no model fallback. Explicit snapshot requests are rejected before cost estimation or API calls because their individual prices have not been verified. Exact returned dated model IDs, request IDs, system fingerprints, token usage, image detail, and sampling parameters are recorded. The default request parameters are high image detail, temperature 0, top-p 1, seed 0, and at most 700 output tokens. Sampling controls reduce variation but do not guarantee identical service responses.

The [official GPT-4o documentation](https://developers.openai.com/api/docs/models/gpt-4o) marks the model deprecated. Account/model access is not established by an offline dry-run; an unavailable model produces an explicit failure.

For the primary cohort, after reviewing `execution_plan.json`:

```bash
python -m studies.spherediff_vlm_eval evaluate \
  --output outputs/10.16spherediff-vlm-eval/seed0-v1 \
  --scope primary --model gpt-4o \
  --execute --budget-usd 20 --max-requests 1000
```

For the predeclared alpha ablations, reusing successful primary requests:

```bash
python -m studies.spherediff_vlm_eval evaluate \
  --output outputs/10.16spherediff-vlm-eval/seed0-v1 \
  --scope ablations --model gpt-4o \
  --execute --budget-usd 30 --max-requests 1500
```

`--scope sphere28` enables the separate FLUX28 diagnostic. `--scope all` enables the union of every declared cohort. `evaluate` without `--execute` is an offline dry-run. Both spending limits default to zero.

Budget and request limits are **cumulative per output directory**, including retries and uncertain failed attempts. The client checks that the planned remaining requests fit the configured limits before beginning, and checks again before every actual attempt. A retry consumes another request allowance. A conservative cost reservation includes maximum output length, a text-token upper bound, and a larger image-token allowance than the current expected cost. This is an application spending guard based on documented rates, not a provider billing guarantee.

Current estimate assumptions are $2.50 per million input tokens and $10 per million output tokens, with 255 image tokens for high-detail 512² images according to the [current vision rules](https://developers.openai.com/api/docs/guides/images-vision). Prompt/output token counts are estimates; actual usage is retained. The reservation uses a conservative 765 image-token allowance. Rates and assumptions are dated in the execution plan; no prompt-caching discount is assumed.

Successful content-addressed requests are never repeated on resume. Terminal failures are retained. An interrupted in-flight request can have been billed despite no local response; it stays blocked for inspection instead of automatically repeating. Raw received responses in the durable journal are recovered without a second paid call. Transient network/rate-limit/server errors receive bounded exponential backoff; authentication, quota, invalid model, parsing, and range errors do not silently trigger a different model or altered prompt.

## Scoring and outputs

Every panorama/view receives a row, including pending and failed evaluations. A panorama average requires all fourteen valid scores. Methods average complete panorama means with equal prompt weights. Every method in a table uses the same complete paired prompt subset; partial subsets are marked preliminary. No missing score becomes zero. Mixed resolved judge snapshots are blocked from combined statistics.

Paired differences against SphereDiff and RGB use a deterministic percentile bootstrap over **prompts**, never individual views. Fewer than two prompts produces no confidence interval. Elevation, individual/combined pole, seam-centered connectivity, and per-prompt rankings reuse the same fourteen evaluations.

The versioned output directory contains:

- `inventory.csv`, `matched_cases.csv`, `missing_cases.csv`, `inventory.json`, `provenance_audit.json`.
- `protocol.json`, `views.csv`, `views.json`, `rendered/` with the verified input views.
- `view_scores.csv`, `panorama_scores.csv`, `matched_summary.csv`, `paired_comparisons.csv`, `alpha_ablation.csv`.
- `diagnostics.csv`, `panorama_diagnostics.csv`, `per_prompt_ranking.csv`, `aggregation.json`.
- `results.tex`, `report.md`, and `figures/` with matched fourteen-view and seam panels.
- `execution_plan.json` (latest), append-only `execution_plans/`, `api_dry_run.json`, and later `api_run_summary.json`.
- `responses/`, `api_attempts/`, `judge_configuration.json`, `implementation.json`, and `implementation.tar.gz`.
- `environment_preflight.json` and `preservation_check.json`.

Refresh reports and verify preserved historical evidence without paid calls:

```bash
python -m studies.spherediff_vlm_eval report --output outputs/10.16spherediff-vlm-eval/seed0-v1
python -m studies.spherediff_vlm_eval verify-preservation --output outputs/10.16spherediff-vlm-eval/seed0-v1
python -m studies.spherediff_vlm_eval check-environment --network --output outputs/10.16spherediff-vlm-eval/seed0-v1
```

The optional network check sends only an unauthenticated HEAD request to the API model-list endpoint. It does not submit images, use an API key, verify model entitlement, or incur inference charges. The API dry-run itself makes no network calls.
