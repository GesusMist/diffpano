# GWTFlow 4096 x 2048 ERP follow-up

All five backends, seed 0, exact directional ruins prompt, B0C0D1.
Only `erp.height=2048` and `erp.width=4096` change from the completed
GWTFlow study. The 89 saved 80 x 80 degree perspective cameras, local RGB/native
rasters, native ERP noise grids, GWTFlow implementation and exact initial
states, checkpoints, schedules and inference step counts remain unchanged.
This is a new generation at the larger consensus grid, not image upscaling.

The frozen generation function is loaded in a private module with isolated
output/provenance hooks. Every backend checks the real model/schedule and exact
initialization digest and performs full-size constant-RGB coverage/round-trip
validation before any denoiser calls. The earlier statistical gate remains
applicable because initialization geometry and source noise are unchanged.

Run validation first, then five independent GPU jobs, then the report:
`sbatch studies/gwtf_erp4k/validate.slurm`
`sbatch studies/gwtf_erp4k/run.slurm sd2` (also sana, flux, sd35, pixeldit)
`sbatch studies/gwtf_erp4k/report.slurm`
Use afterok dependencies on the prior phase. All launchers check frozen hashes.
Outputs: `outputs/gwtf-erp4k/20260923/<backend>/gwtf-final.png`.
Historical output files and scientific source files are not edited.
