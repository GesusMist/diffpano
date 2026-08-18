# DiffPano usage guide

## Environment

DiffPano targets Python 3.10+, CUDA-capable NVIDIA GPUs, and PyTorch 2.7.x. Model memory requirements vary substantially; the checked-in TAMU HPRC launchers request one A100. Install a CUDA-compatible PyTorch build before the repository dependencies.

```bash
conda create -n diffpano python=3.10
conda activate diffpano
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

The editable install is required for direct commands such as `python scripts/generate.py`.

## Layout

```text
diffpano/
├── config.yaml                 complete default experiment
├── diffpano/                   reusable package
│   ├── config.py               typed loading and validation
│   ├── initialization.py       seeds and directional prompts
│   ├── diffusion.py            high-level pixel-fusion orchestration
│   ├── geometry.py             spherical geometry shared with baseline
│   ├── projection.py           FP32 perspective/ERP transforms
│   ├── lpw.py                  Laplacian Pyramid Warping
│   ├── fusion.py               overlap aggregation and RGB fusion
│   ├── vae.py                  decode/encode and residual bridge
│   ├── reinjection.py          scheduler-consistent clean correction
│   ├── writeback.py            spherical ownership and write-back
│   ├── diagnostics.py          optional debug outputs
│   ├── metadata.py             compact reproducibility metadata
│   └── pipelines/              model-specific adapters and registry
├── experiments/planar/         isolated no-warp ablation
├── spherediff/                 baseline/reference namespace
├── scripts/                    executable entrypoints
├── slurm/                      cluster launchers
├── prompts/                    directional prompt files
└── tests/                      unit and regression tests
```

## Models

`model.path` has precedence over `model.id`. A non-null local path is passed directly to Diffusers; otherwise the Hugging Face model ID is used. To prefetch a remote model:

```bash
python scripts/download_models.py --config config.yaml
```

The current registered spherical adapters are `sana`, `flux`, `hunyuan_video`, and `ltx_video`. Pixel fusion is implemented for SANA and FLUX. Set `fusion.enabled: false` for video adapters. Gated repositories such as FLUX may require `huggingface-cli login` and accepted model terms.

## Configuration reference

All unspecified fields receive the typed defaults below. The checked-in root `config.yaml` spells out every field and preserves the active SANA DiffPano experiment defaults. Unknown fields are rejected.

### `experiment`

| Field | Type | Default | Meaning |
|---|---|---:|---|
| `experiment.name` | string | `diffpano-sana` | Output experiment directory. |
| `experiment.seed` | integer or null | `1` | RNG seed; null preserves caller/global RNG state. |

### `model`

| Field | Type | Default | Meaning |
|---|---|---:|---|
| `model.pipeline` | string | `sana` | `sana`, `flux`, `hunyuan_video`, `ltx_video`; `planar_sana` is accepted only by the planar runner. |
| `model.id` | string | SANA 1600M ID | Remote model repository. |
| `model.path` | string or null | null | Local model path; takes precedence over `id`. |
| `model.revision` | string or null | null | Optional repository revision. |
| `model.variant` | string or null | `bf16` | Diffusers weight variant. |
| `model.precision` | string | `bf16` | `fp16`, `bf16`, or `fp32`. |
| `model.cpu_offload` | bool | false | Enable Diffusers model CPU offload. |
| `model.vae_tiling` | bool | false | Enable backend VAE tiling where supported. |
| `model.vae_slicing` | bool | false | Enable backend VAE slicing where supported. |
| `model.additional_pipeline_kwargs` | mapping | `{}` | Explicit model-construction overrides. |

### `prompt` and `output`

| Field | Type | Default | Meaning |
|---|---|---:|---|
| `prompt.path` | string | `prompts/ruins.txt` | Five-line directional prompt file. |
| `prompt.negative_path` | string or null | null | Optional negative-prompt file. |
| `output.directory` | string | `outputs` | Root for standardized run directories. |
| `output.save_final` | bool | true | Save the final image/video. |
| `output.save_metadata` | bool | true | Save `metadata.json`. |
| `output.save_intermediate` | bool | false | Include configured intermediate tensors. |
| `output.run_id` | string or null | null | Explicit run folder; null uses timestamp plus SLURM job/local suffix. |
| `output.fps` | integer | `24` | Video export rate. |

### `generation` and `sphere`

| Field | Type | Default | Meaning |
|---|---|---:|---|
| `generation.num_inference_steps` | integer | `20` | Diffusion denoising steps. |
| `generation.guidance_scale` | float | `4.5` | Model guidance scale. |
| `generation.height` | integer | `1024` | Perspective generation height. |
| `generation.width` | integer | `1024` | Perspective generation width. |
| `generation.num_frames` | integer or null | null | Video frame count; null preserves backend default. |
| `generation.use_resolution_binning` | bool | true | SANA resolution binning; ignored for other adapters. |
| `generation.additional_call_kwargs` | mapping | `{}` | Explicit backend call overrides. |
| `sphere.num_points` | integer | `2600` | Number of spherical latent points. |
| `sphere.fov` | pair or null | null | Reserved; current dynamic view geometry derives FOV internally, so leave null. |
| `sphere.erp_height` | integer | `2048` | Temporary/debug ERP height. |
| `sphere.erp_width` | integer | `4096` | Temporary/debug ERP width. |
| `sphere.weighted_average_temperature` | float | `0.1` | Original SphereDiff overlap weighting temperature. |

### `fusion.warp`, `fusion.lpw`, and `fusion.aggregation`

| Field | Type | Default | Meaning / allowed values |
|---|---|---:|---|
| `fusion.enabled` | bool | true | Enable DiffPano pixel-space fusion. |
| `fusion.start_ratio` | float | `0.0` | First eligible normalized denoising-step ratio. |
| `fusion.end_ratio` | float | `1.0` | Last eligible ratio; must be at least start. |
| `fusion.every_n_steps` | integer | `1` | Apply fusion every N eligible steps. |
| `fusion.warp.mode` | string | `lpw` | `standard` or `lpw`. |
| `fusion.lpw.levels` | integer | `4` | Number of pyramid levels. |
| `fusion.lpw.lod_mode` | string | `jacobian` | `jacobian` or `none`. |
| `fusion.lpw.lod_interpolation` | string | `nearest` | `nearest` or `linear`. |
| `fusion.lpw.erp_vertical_padding_mode` | string | `reflect` | Perspective pyramid padding: `reflect` or `replicate`; ERP filtering stays spherical. |
| `fusion.lpw.erp_to_perspective_interpolation_mode` | string | `nearest` | `nearest` or `bilinear`. |
| `fusion.aggregation.mode` | string | `detail_preserving_average` | `average`, `weighted_average`, or `detail_preserving_average`. |
| `fusion.aggregation.weight_mode` | string | `distance_to_boundary` | `uniform`, `cosine`, `gaussian`, or `distance_to_boundary`. |
| `fusion.aggregation.alpha` | float | `1.0` | DPA correction strength; zero reduces to weighted average. |
| `fusion.aggregation.power` | float | `1.0` | Nonnegative DPA coefficient-magnitude power. |
| `fusion.aggregation.epsilon` | float | `1e-6` | Positive numerical floor. |

The reserved `fusion.time_travel` mapping preserves the existing fields: `enabled` (bool, default false), `every_n_steps` (integer, 1), `jump_length` (integer, 1), `num_repeats` (integer, 1), and `strength` (float, 1.0). Enabling it still raises `NotImplementedError`; the refactor does not invent scheduler behavior.

### Reinjection, write-back, and performance

| Field | Type | Default | Meaning / allowed values |
|---|---|---:|---|
| `reinjection.mode` | string | `noise_consistent` | `noise_consistent`, `replace`, `weighted_replace`, or `residual`. |
| `reinjection.strength` | float | `1.0` | Fused clean correction strength. |
| `writeback.mode` | string | `exclusive` | `exclusive` or SphereDiff-compatible `weighted_average`. |
| `writeback.owner_mode` | string | `max_center_weight` | Stable highest-center-weight ownership. |
| `writeback.owner_map_static` | bool | true | Cache ownership while geometry is unchanged. |
| `writeback.uncovered_mode` | string | `error` | `error` or `weighted_average_fallback`. |
| `performance.vae_chunk_size` | integer | `1` | Views per VAE encode/decode chunk. |
| `performance.projection_chunk_size` | integer | `1` | Same-sized views per ERP projection chunk. |
| `performance.vae_sample_posterior` | bool | false | Compatibility field; residual bridge uses deterministic mean/mode. |

### `debug`

| Field | Type | Default | Meaning |
|---|---|---:|---|
| `debug.enabled` | bool | false | Save compact diagnostics. |
| `debug.save_predicted_x0` | bool | false | Save original predicted-clean ERP observer frames. |
| `debug.save_original_clean_erp` | bool | false | Save baseline predicted-clean ERP frames when fusion is disabled. |
| `debug.save_fused_erp` | bool | false | Save fused predicted-clean ERP frames. |
| `debug.save_owner_map` | bool | false | Save owner IDs, scores, coverage, and write counts. |
| `debug.save_projection_diagnostics` | bool | false | Add center/round-trip projection diagnostics. |
| `debug.save_masks` | bool | false | Save validity, count, weight, and overlap masks. |
| `debug.measure_performance` | bool | false | Synchronize CUDA for accurate stage timings. |

Debug artifacts go under the run's `intermediates/` directory. Large intermediates are not emitted unless requested.

## Prompt format

Prompt files contain exactly five non-empty lines, ordered as the directional groups used by SphereDiff: north pole, upper/equatorial directions, equator, lower/equatorial directions, and south pole. Existing examples are under `prompts/`. The model adapters retain the original directional assignment logic.

## Local generation

```bash
python scripts/generate.py --config config.yaml
```

The resolved config is printed at startup and copied into the run directory. For a local model, set `model.path`; for a cached/remote model, leave it null and set `model.id`.

## SLURM

The SLURM files contain resources, activation/cache setup, and a command only. Experiment choices remain in YAML.

```bash
mkdir -p logs
sbatch slurm/generate_a100.slurm config.yaml
sbatch slurm/debug.slurm my_debug_config.yaml
sbatch slurm/test.slurm
```

Set `DIFFPANO_ROOT` if the checkout is not `$HOME/diffpano`. The provided HPRC launchers use `activate_venv diffpano` for GPU jobs and the existing explicit test virtual environment for CPU tests.

## Outputs

```text
outputs/<experiment-name>/<run-id>/
├── result.png        # or result.mp4
├── config.yaml       # fully resolved configuration
├── metadata.json     # compact parameters, runtime, environment, SLURM_JOB_ID
├── run.log
└── intermediates/    # only when output/debug options request it
```

## Planar ablation

The no-warp planar experiment isolates overlapping patch diffusion, RGB blending, the VAE residual bridge, reinjection, and latent write-back from spherical projection. It is intentionally not in the normal panorama registry.

```bash
python scripts/planar_test.py --config experiments/planar/config.yaml
sbatch slurm/smoke_planar.slurm experiments/planar/config.yaml
```

Its `planar` section preserves the dedicated patch fields. The checked-in config uses 20×20 latent patches and stride 5 in both directions.

| Planar field | Type | Default/config value | Meaning |
|---|---|---:|---|
| `patch_latent_height`, `patch_latent_width` | integer | `20` | Patch size on the latent canvas. |
| `patch_stride_height`, `patch_stride_width` | integer | `5` in the checked-in experiment | Dense patch stride; each must not exceed its patch size. |
| `fusion_space` | string | `pixel` | `pixel` for VAE/RGB fusion or `latent` for direct latent blending. |
| `aggregation_mode` | string | `average` in the checked-in experiment | `average`, `weighted_average`, or `detail_preserving_average`. |
| `weight_mode` | string | `distance_to_boundary` | Patch weight construction mode. |
| `dpa_alpha`, `dpa_power`, `dpa_eps` | float | `1.0`, `1.0`, `1e-6` | DPA correction, coefficient power, and numerical floor. |
| `reinjection_mode`, `reinjection_strength` | string, float | `noise_consistent`, `1.0` | Scheduler-state correction mode and strength. |
| `latent_writeback_mode` | string | `exclusive` | `exclusive` or `weighted_average`. |
| `vae_chunk_size` | integer | `4` | Planar patches per VAE call. |
| `vae_sample_posterior` | bool | false | Compatibility field; residual bridge remains deterministic. |
| `save_diagnostics`, `diagnostics_dir` | bool, path/null | false, null | Optional compact planar diagnostics and destination. |

The runner injects `experiment.seed` as the planar random seed unless the `planar` mapping explicitly overrides it.

## SphereDiff baseline

`python scripts/generate.py --config spherediff/config.yaml` uses the shared spherical adapter with fusion disabled, preserving the original weighted spherical latent write-back path. See `spherediff/README.md` and the archived legacy commands in `spherediff/LEGACY_USAGE.md`.

## Testing

```bash
python -m unittest discover -s tests -v
# TAMU HPRC
sbatch slurm/test.slurm
```

GPU/model smokes accept a complete config as their only experiment argument. For example: `sbatch slurm/smoke_sana.slurm smoke.yaml`.

## Troubleshooting

- `CUDA is required`: run on a GPU allocation, or enable model CPU offload while still providing a CUDA device for inference.
- Model access/download failures: authenticate with Hugging Face, accept gated terms, or set `model.path` to a local snapshot.
- Out of memory: reduce `performance.vae_chunk_size` and `performance.projection_chunk_size`, then consider CPU offload. Do not change projection math to work around memory.
- Uncovered owner points: inspect `debug.save_owner_map`; use `weighted_average_fallback` only as an explicit experiment choice.
- Invalid config: unknown fields and unsupported modes are rejected before model loading.

## Adding a backend

1. Add a thin Diffusers adapter in `diffpano/pipelines/<backend>.py`. Keep model loading, prompt encoding, transformer invocation, packing/unpacking, scheduler differences, and VAE shape adapters there.
2. Reuse the model-independent stages from `diffpano.diffusion`, `vae`, `projection`, `lpw`, `fusion`, `reinjection`, and `writeback`.
3. Register the class once in `diffpano/pipelines/__init__.py`.
4. Extend typed pipeline-name validation and call-argument construction.
5. Add registry/import tests plus a model smoke config. Do not put backend parameters in SLURM.
