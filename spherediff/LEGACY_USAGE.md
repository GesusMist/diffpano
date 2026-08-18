# Legacy SphereDiff command migration

The original release used `generate_static_wallpaper.py`, `generate_live_wallpaper.py`, and long `--config_add` argument lists. Those filenames remain thin compatibility adapters, but new work should use one experiment YAML.

| Legacy setting | Canonical field |
|---|---|
| `pipeline_cls=SphericalSanaPipeline` | `model.pipeline: sana` |
| `pipeline_cls=SphericalFluxPipeline` | `model.pipeline: flux` |
| `pipeline_cls=SphericalHunyuanVideoPipeline` | `model.pipeline: hunyuan_video` |
| `pipeline_cls=SphericalLTXPipeline` | `model.pipeline: ltx_video` |
| `pretrained_model_name_or_path=...` | `model.id` or `model.path` |
| `mixed_precision=bf16` | `model.precision: bf16` |
| `call_kwargs.prompt_txt_path=...` | `prompt.path` |
| `call_kwargs.n_spherical_points=...` | `sphere.num_points` |
| `save_path=...` | `output.directory` plus `experiment.name` |

Use the baseline config directly:

```bash
python scripts/generate.py --config spherediff/config.yaml
```

Copy that config to create additional baseline experiments. Keep `fusion.enabled: false` to select the original SphereDiff path. Prompt files now live under `prompts/`. Legacy wrappers use the standardized run-directory layout rather than the old timestamped filename prefix.
