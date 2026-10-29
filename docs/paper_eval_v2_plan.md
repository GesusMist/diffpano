# DiffPano paper evaluation v2

Run: `outputs/paper-eval-v2/20261026-core-v1`. The run is backed by `/home/shig/paper-eval-v2` to avoid the nearly full scratch inode quota. Historical experiments and scores are read-only. Actual starting HEAD is `f9b321a0fa8877c1eec3d94d120e1fa609da6fa2`; the user request document was initially untracked.

The selected tier is **core**: 24 fresh real scenes × final DiffPano, official SphereDiff and official SpheRoPE = 72 ERP slots. The two independent native FLUX reference banks add 240 ordinary perspective images on the development24 prompts. Optional seeds24, trained24 and generality32 slots are manifested but not selected. No additional all-variant factorial is authorized by this tier selection.

The user explicitly superseded the document's local API-cap requirement: OpenRouter's configured limit controls spending, with no additional local request or USD cap. The shared ledger records every attempt, reservation, uncertain outcome and reconciliation across GPT-4o, GPT-6 Sol, calibration, repeats and captioning. Provider restrictions and paired-prompt ordering remain mandatory. Credentials are loaded only by API workers from the private user-owned environment file; no secret is saved in this repository or run artifacts.

## Frozen scientific identities

Development24 is the exact 20 original + four new five-line prompts; development15 is the established 11 original + four new subset. Identities include complete relative paths and exact prompt SHA256 values. Confirmation24 is a separate real-scene cohort, selected before generation or scoring, targeting six indoor and eighteen outdoor scenes. ODI-SR from the official LAU-Net release is preferred; only failed acquisition permits the named, already verified SUN360 release fallback. A public link alone is not acquisition evidence.

Primary development methods are official SphereDiff FLUX20 (`sphere20`), Old89 RGB (`rgb`), Old89 screened poisson_select (`select`) and Old89 screened mean-max alpha0.50 (`meanmax050`). Alpha is the maximum-gradient coefficient. Other variants are explicit ablations. The fallback final method is screened select without refinement if matched development quality/geometry/semantics/runtime evidence does not identify a clear winner. Confirmation results cannot change this selection.

`legacy` retains exact existing metric definitions. `common_v2` uses one fixed periodic pixel-center ERP2048×1024 resampler, eight horizontal FOV90 512-square crops for ordinary crop metrics, and fourteen FOV90 views for VLM/MUSIQ. The seam-centered views have yaw180 and pitches−45/0/+45; analytical geometry tests verify this. `native_five_v2` samples native ERPs at yaw0, pitches+90/+10/0/−10/−90, FOV80, 1024 square. Native resolution and upsampling are recorded. FAED has its own verified RGB full-ERP protocol and is blank without an authenticated checkpoint.

## Outcomes and inference

DS is lower-is-better wrap discontinuity and can favor smoothness. CS reports 100×raw CLIP cosine, including horizontal, north/south/polar and equally weighted five-band directional means. MUSIQ preserves its pretrained KonIQ score scale. IS is one collection-level score, never averaged image scores. Seam-SSIM and edge-adjacent Sobel are supplementary diagnostics, not whole-panorama geometric proof.

Predeclared primary paired contrasts are select versus SphereDiff, select versus RGB, and mean-max0.50 versus select, on explicit matched prompt intersections. Views are averaged within panoramas and seeds within prompts. Means and differences use 2,000 fixed-seed paired prompt-cluster bootstrap replicates. Missing cases remain missing. FID/KID and OmniFID retain full feature dimensionality, float64 statistics and declared reference/cohort weighting. Distribution comparisons on compact cohorts are exploratory; subset SD is not a confidence interval. Native-model preservation and real-data realism never share a ranking column.

GPT-4o is the primary judge; GPT-6 Sol is secondary. Both receive identical cached anonymous PNG bytes and the same anchored four-criterion v2 rubric. Their tables remain separate. The pilot checks image transmission, valid parsing, corruption response and style confounding; preselected repeats measure variability and never replace low scores.

## Execution and durable state

The shared resource limit is two generation GPUs plus one evaluation GPU; native reference generation counts as generation. Initial account balance was 7,353.35 SUs and the user queue was empty. Finite job intents are journaled before submission; ambiguous submissions require scheduler/accounting reconciliation before retry. GPU reservations, actual completion and metric validity remain distinct states.

The run saves `run_config.json`, `dependency_graph.json`, `environment.json`, `resource_plan.json`, `scheduler.json`, `status.json`, the protocol registry, inventory/cohorts, exact generation manifests and API ledger. Atomic writes and locks protect shared state. View and response caches use packed stores to limit inode growth. No historical output, prompt, freeze or score is altered.

## Reproduction entry points

Use `bash studies/panorama_metrics/environment.sh -m studies.paper_eval_v2` followed by `inventory`, `prepare`, `evaluate`, `launch`, `status`, or `report`. Heavy inventory/rendering and GPU feature extraction run through finite Slurm wrappers. `RESUME.md` in the run tracks the exact next commands and outstanding IDs as implementations and outputs become available.

Reports distinguish implementation readiness, job submission, verified output completion, scored completion and scientific evidence. Human materials are offline only, with anonymous balanced A/B assignments and a private method key; 240 requested pair evaluations remain awaiting real ratings. No recruitment, external publication, messages or remote Git pushes are performed.
