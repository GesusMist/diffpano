2Dtest: sbatch --export=ALL,NUM_INFERENCE_STEPS=30,RUN_LABEL=planar_residual_bridge tests/planar_patch_sana_smoke.slurm

sbatch --export=ALL,\PIPELINE=ruins,\PROMPT_NAME=underwater,\N_SPHERICAL_POINTS=26500,\PIXEL_FUSION_CONFIG_PATH="$HOME/diffpano/configs/pixel_fusion_default.yaml" a.slurm