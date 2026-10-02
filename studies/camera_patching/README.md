# Camera patching study

This isolated study changes only fixed perspective camera centers and their stable GWTFlow IDs.
It reuses `studies/all_prompts` ordinary execution, `studies/tt_cea` ERP/CEA operators and interval implementation,
and the unchanged core GWTFlow initializer/local bridge. No historical modules are edited.

Generic constructors support arbitrary positive integer N. Scientific runs are restricted to N=89:
Fibonacci phase 0 and Random layout seed 0, four backends, three prompts, ERP/CEA, 48 new outputs.
Random draws are one dedicated CPU FP32 `torch.rand((N,2))` tensor, rows ordered by camera index;
columns map to uniform yaw and uniform sin(pitch). IDs encode strategy, N, index, and random layout seed.
Angular hashes use canonical ordered yaw/pitch/roll/FoV JSON, excluding raster dimensions.

Run through the established `studies/all_prompts/environment.sh diffpano` environment in Slurm:

- `python -m studies.camera_patching.validate`: tests, 24 old89 audits, full ERP/CEA and spherical probe geometry gates.
- `python -m studies.camera_patching.preflight [--backend NAME]`: real-backend initialization replay plus each enabled layout/projection's first synchronized interval.
- `python -m studies.camera_patching.launch --dry-run`, `--submit`, or `--status`.
- `python -m studies.camera_patching.run --strategy fibonacci --camera-count 89 --backend sana --projection erp --prompt ruins`.
- `python -m studies.camera_patching.report`: validate final image/config pairs, worker exits, paired initialization hashes and print counts.

Validation and preflight evidence live only in normal Slurm logs and are bound to the full source/prompt fingerprint.
A single production array enforces the global five-GPU ceiling. Fresh worker process per sample; no duplicate active submission.
A valid completed image/config pair is never overwritten. Both temporary files are verified then atomically renamed individually;
a crash between renames leaves an incomplete pair, which is never counted as complete.

Geometry diagnostics are analytic center weights and exact production frustum conventions, processed in ray chunks.
Generation retains the existing bilinearly sampled center-weight raster. Small positive weights are covered, never thresholded as holes.
If a layout has uncovered rays, that layout is blocked without changing N, seed, phase or FoV.

Result folders contain only final.png and compact runtime config.json; three required descriptive READMEs live at root/strategy levels.
The 24 old89 baselines remain in `outputs/all-prompts-erp-cea-seed0`.
Geometry changes also affect noise transport/covariance, prompt routing and weighted overlap; this is a camera-patching intervention.
No image-quality metrics, N/seed/phase tuning or new original SphereDiff runs are part of this study.
