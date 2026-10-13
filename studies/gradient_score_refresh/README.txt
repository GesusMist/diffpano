October 9 FLUX DS / IS / CS refresh

Requested collections:
  refinement64: 4 prompts x 4 Poisson references x 2 guidance choices x 2 tails.
  blending20: 20 prompts x RGB / Poisson mean / Poisson select = 60 images.
The twenty-prompt source is outputs/10.7gradient-blending-flux (the request called
it 10.8gradient-blending-flux). Source images remain in their existing folders.

Run through the pinned evaluation environment:
  sbatch studies/gradient_score_refresh/evaluate.slurm --allow-partial
  sbatch studies/gradient_score_refresh/evaluate.slurm
The first command exports available per-image rows and complete matched groups.
Incomplete group scores are blank, never calculated over a shorter prompt set.
The second command requires all 124 outputs. Jobs use an exclusive evaluator lock.

Outputs: outputs/10.9gradient-metrics/seed0-v1
  per-image-metrics.csv: DS and CS, with image/prompt/protocol/metadata hashes.
  group-metrics.csv: mean DS/CS and one collection IS per method/configuration.
  refinement64/: 64 rows and 16 groups of four prompts (32 views per group).
  blending20/: 60 rows and three all20 groups (160 views per group), plus the
    same four-prompt subset as the factorial study for later matched comparison.
  protocol.json: exact historical definitions, checkpoints, source hashes,
    rendering, directional prompts, and long-text policies.
  baseline-validation.json: all twenty RGB controls must match earlier DS/CS
    values within 1e-4. Valid existing feature caches are reused by hash.
  input-inventory.json, status.json, completion.json, report.txt, cache/, logs/.
Each source study exposes its results at evaluation/ds-is-cs -> the relevant
subdirectory here. Historical evaluation files and scores are retained.

DS is the same explicitly documented Scharr adaptation as earlier results, at
native 4096x2048 ERP resolution. CS uses the same eight 90-degree horizontal
views and CLIP ViT-B/32 prompt policy. IS uses the same Inception checkpoint and
one full-collection split; it is NOT averaged over per-image IS. Four-prompt IS
is descriptive; the rendered views are dependent. Never pool factorial methods.

Two missing outputs are recovered separately with the validated current ordinary
FLUX trajectory: firework/coarse_color/select/off at previously validated rtol
1e-4, and clouds_over_grass/poisson_mean at original screened rtol1e-5. Exact
conditioning/schedule/camera/initialization checks remain enabled. Failed attempt
records are archived in recovery/. Current production sources and completed
image metadata are not modified. Maximum two simultaneous generation GPUs plus
one evaluator. Final evaluation depends on recovery and the first scoring pass.

84 existing refinement Slurm logs were moved to its logs/{generation,validation,
evaluation,reports}; logs/index.json records old/new locations and hashes.
New standard Slurm templates use those log directories. The frozen older retry
suite is retained; new jobs always override its former output path if used.
