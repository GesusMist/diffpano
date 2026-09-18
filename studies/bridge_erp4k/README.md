# Selected ruins cells at 2048×4096 ERP

This harness reuses the completed bridge-factorial runner without editing it.
Each selected configuration differs from its completed baseline at exactly
`erp.height` and `erp.width`. Local rasters, saved 89 cameras, 80°×80° FOV,
native noise grids and initial local states remain fixed.

Requested ABCD codes:

- FLUX: 0001, 0101, 1001, 1011, 1101, 1111
- PixelDiT: 1001, 1011, 1111, 0001
- SANA: 0001, 0011, 1001, 1011, 1111
- SD2: 1001, 1011, 1111
- SD3.5: 1000, 1001, 1011, 1010

The original saved-camera verifier is called at its verified baseline raster.
The same camera objects are then used at the target ERP raster. Full-size GPU
preflights independently verify target coverage and standard/LPW fusion.

`run.py` loads a private module instance of the original runner and replaces
only its study I/O, resolution configuration, camera verification and provenance
hooks. The model prediction, VAE bridge, RGB fusion, local-state transition,
initialization and diagnostic functions are reused unchanged. No shared module
globals or preexisting scientific files are modified.

Submission from the repository root:

```bash
python3 studies/bridge_erp4k/launch.py validate
python3 studies/bridge_erp4k/launch.py preflight  # after validation succeeds
python3 studies/bridge_erp4k/launch.py run        # after all five preflights pass
python3 studies/bridge_erp4k/launch.py report     # depends on incomplete run jobs
python3 studies/bridge_erp4k/launch.py status
```

The launcher reuses completed results and checks active/previous jobs to avoid
duplicates. The output root is `outputs/bridge-erp4k-ruins/20260918/`.
Each scientific cell retains only `final_result.png` and `metadata.json`.
Study-level output includes the frozen manifest, validation/preflight records,
paired CSV/JSON diagnostics and five paired contact sheets.

This is an incomplete subset of the factorial, so analysis reports paired ERP
resolution changes, not new main effects or interactions. Compare local views
at matching raster/FOV; do not compare raw ERP pixel-frequency metrics directly
across native ERP resolutions. No original SphereDiff reference is rerun.
