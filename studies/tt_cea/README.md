# Original-noise replay and cylindrical equal-area controlled study

This study implements the frozen request in REQUEST.txt. It adds no production
pipeline defaults and changes no historical sources. See experiment.yaml and the
immutable 60-row manifest in outputs/tt-cea/seed0-v1 for all allowed interventions.
R0 reuses the ten audited 4K GWTFlow panoramas. T1 performs one original-noise
backward bridge and fresh two-pass replay for each original interval satisfying
N <= 5*k < 4*N. TB prepares the ordinary N+K-step schedule. C0 changes clean RGB
projection only; P0 changes the fixed camera layout; C1 combines CEA and EA89,
without time travel. Model input frusta stay 80 by 80 degrees at inherited rasters.

The study-local interval executor is derived from
`diffpano/dense_consensus.py:138-263` and reuses the frozen bridge and transition
functions from bridge_factorial.py/current_state_transition.py. Independent mock
and real-backend comparisons execute the actual frozen loop as the oracle.
No old study globals are patched by experiment jobs. Replays leave the prepared
timestep list untouched, reset scheduler call state for every camera, and use an
immutable CPU FP32 original-noise bank. The generalized alpha/sigma bridge in
REQUEST.txt defines this experiment, not a substituted paper re-noising operator.

CEA stores clean RGB at uniform longitude/sine-latitude centers. Fusion uses the
existing center weights without a cos(latitude) multiplier. Its explicit polar
boundary convention fuses exact pole rays and interpolates in polar angle from
those values to the first/last row only. Longitude sampling in the caps uses the
selected nearest/bilinear interior convention so the cap boundary is continuous.
This is a study boundary convention, not an exact author-provided implementation.
The GWTFlow source remains ERP-indexed for every row. EA89 uses fixed Fibonacci
centers, not an exact equal-area partition of perspective footprints.

Report viewports use bilinear ray sampling at fixed 80-degree 1024-square views;
yaw 180 degrees is the longitude-seam viewport. CEA direct views are rendered
from the native floating canvas before PNG quantization. Historical R0 has only saved PNGs,
so its report viewport source is explicitly identified as PNG rather than float.

Context references (the attached equations control this study):
- https://arxiv.org/html/2504.08902v1 (intermediate-interval replay context)
- https://proj.org/en/stable/operations/projections/cea.html (CEA context)

Run the audit, validation, geometry and backend preflights before production.
`python -m studies.tt_cea.launch <phase>` prints a dry-run plan; `--submit` submits
only after checking outputs, the ledger, squeue and sacct. GPU work uses Grace's
working offline model environment; there is no heavy inference on the login node.
Phases: validate, preflight, time-travel, projection, report, status.
Each production process is fresh, with at most five concurrent GPU jobs. No model
quality tuning, additional seeds or prompts, combined CEA/replay, commits or push.

The baseline audit retains the historical SANA scheduler's `lambda_min_clipped`
negative-infinity sentinel verbatim. New preflight/production metadata records
that configuration sentinel as the string `-Infinity`; numeric diagnostics,
interval coefficients, tensors and all other serialized values must be finite.
Validation corrections use separate recorded CPU attempts. Production retries
are never automatic; numerical failures remain evidence, not retried samples.

The launcher removes an inherited virtual-environment Python from the submitted
PATH so Grace's `activate_venv` helper can run after `module purge`. The first six
GPU preflight attempts failed before Python startup due to a missing libpython
shared library; their Slurm/log evidence is retained. An explicit infrastructure
retry creates new ledger entries, without changing any experiment setting.
