# Underwater shared versus GWTFlow

Ten fresh runs, five models, seed 0, clean RGB ERP 4096 wide by 2048 high.
Use all five directional lines of prompts/underwater.txt. Reuse the exact 89
80-degree cameras, validated local/native resolutions and source-noise grids,
B0C0D1, bridge, model revisions, schedules, precision and guidance from the 4K
ruins study. The only within-model intervention is nearest shared ERP versus
GWTFlow initialization. Both initializers must reproduce their previously
validated seed-0 state digests. Each arm uses a fresh process/model/scheduler.

The frozen generation function is imported privately without modifying previous
source or artifacts. Runtime audit permits only changed conditioning versus
ruins; the final report requires exact configuration and provenance equality
between the underwater arms. Initializer gates forbid model/VAE calls and
verify full target-ERP coverage. Figures contain ERP images only.

The original factorial validator retains its ruins-only guard. This follow-up
validates the unchanged baseline and then permits exactly prompt.path to change;
all remaining configuration and runtime invariants stay explicitly checked.
