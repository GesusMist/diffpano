# DiffPano ERP-RGB and PixelDiT usage

## Persistent-state invariant

`diffpano.erp_pipeline.ERPRGBPipeline` carries one FP32 tensor shaped `[B,3,H_erp,W_erp]` between global timesteps. Every camera reads the same frozen source tensor. Each proposal is accumulated into streaming full-ERP sums, and fusion updates the ERP only after the complete camera cover. No `N_views × ERP` tensor stack is retained.

For SANA, FLUX, and SD2, the tensor is an RGB synchronization canvas and each adapter performs its existing deterministic encode/model-step/decode sequence. For PixelDiT, the tensor is the actual pixel-space diffusion state `X_t`; projection, the model, the solver step, and fusion all operate directly on three-channel floating-point pixels. No autoencoder boundary or alternate persistent representation exists in the PixelDiT backend.

## Official PixelDiT integration

The setup script installs `NVlabs/PixelDiT` at commit `41f73006ae532b0b41fee72b181dc22891a5a01a`, the official `master` revision inspected for this integration. The adapter lazily imports the upstream definitions so ordinary DiffPano imports and CPU tests do not require PixelDiT dependencies.

The adapter reads:

```text
third_party/PixelDiT/t2i/configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml
```

and uses its model builder, `PixDiTTrainer`, checkpoint format, checkpoint resolver, Gemma-2 tokenizer/text encoder, CHI prompt, maximum sequence length, masks, classifier-free guidance interval, and stage-3 `flow_shift` default. `model_path: null` resolves the official `pixeldit_t2i_v1.pth` checkpoint (`nvidia/PixelDiT-1300M-1024px`). The checkout commit is verified before model loading.

Install with:

```bash
bash scripts/setup_pixeldit.sh
pip install -r requirements-pixeldit.txt
```

The extra requirements are layered on the existing DiffPano environment so SANA/FLUX/SD2 remain available. The official checkout is ignored by Git and can be placed elsewhere by changing `pixeldit.repo_path` and `pixeldit.config_path` together.

## PixelDiT schedule and first-order update

For `N` steps, start/end times `T=1`, `epsilon=0.001`, and shift `q`, the schedule exactly follows upstream `time_uniform_flow`:

```python
betas = torch.linspace(1.0, 0.001, N + 1)
sigmas = 1.0 - betas
times = (q * sigmas / (1.0 + (q - 1.0) * sigmas)).flip(0)
```

The raw model receives `1000 * s` at continuous time `s`. Let `v_theta(x_s,s)` be its flow prediction. The official flow wrapper and first-order DPM-Solver++ primitive used by DiffPano are:

```text
alpha_s = 1 - s                     sigma_s = s
epsilon_theta = alpha_s v_theta + x_s
x0_theta = (x_s - sigma_s epsilon_theta) / alpha_s
lambda_s = log(alpha_s) - log(sigma_s)
h = lambda_t - lambda_s
x_t = (sigma_t / sigma_s) x_s - alpha_t expm1(-h) x0_theta
```

Here `t` is the next, smaller schedule time. The implementation preserves this official arithmetic rather than replacing it with an algebraically simplified expression. A CPU regression test compares it directly with upstream `DPM_Solver.dpm_solver_first_update`.

The panorama backend evaluates one upstream `model_fn` call and one upstream `dpm_solver_first_update` per view per global interval. A local adapter reproduces the official shifted schedule, while the actual model wrapping and first-order primitive are called directly from the pinned checkout. No solver history survives ERP fusion. Upstream order-2 multistep inference is exposed only by `PixelDiTViewDenoiser.official_sample(..., order=2)` and the standalone single-image test.

## Initialization and value range

PixelDiT requires:

```yaml
initialization:
  mode: pixel_gaussian
  distribution: gaussian
  mean: 0.0
  std: 1.0
  clamp: false
```

This samples `[B,3,H_erp,W_erp]` directly with `torch.randn`. The mode is rejected for other backends, and PixelDiT rejects all other initialization modes and any initial clamp. Intermediate pixel states may be outside `[-1,1]`; projection, LPW, fusion, and diagnostics do not clamp them. Display conversion alone maps the final/debug tensor to an image.

## Prompt conditioning and metadata

`prompt.path` contains one global non-empty line or five directional lines ordered north, upper-equatorial, equator, lower-equatorial, and south. The five bands expand to 20 anchors. Positive camera pitch is north. All positive directional embeddings and the negative embedding are computed once before the timestep loop, then indexed by camera.

PixelDiT prepends the official CHI prompt and uses the official token selection. For every model call, `img_hw` and `aspect_ratio` derive from the actual perspective tensor, including rectangular views; time and text-mask batches follow the selected camera batch.

## Baseline configuration

[configs/pixeldit_standard_average.yaml](../configs/pixeldit_standard_average.yaml) is the requested first experiment:

- 50 steps, 1024×1024 views, 2048×4096 ERP;
- fixed 89-view SphereDiff cover, no rotation;
- standard warp, nearest ERP-to-view and bilinear view-to-ERP;
- plain uniform average, no LPW and no DPA;
- direct unclamped Gaussian ERP state;
- selected-step/selected-view previews and scalar state diagnostics;
- a two-entry GPU geometry cache plus reusable CPU overflow, avoiding both unbounded VRAM and repeated full-ERP grid construction.

[configs/pixeldit_smoke.yaml](../configs/pixeldit_smoke.yaml) uses the same 89-camera algorithm at 256×256 with one global step for a quick real-checkpoint integration test.

Run either through the generic entrypoint:

```bash
python scripts/generate.py --config configs/pixeldit_smoke.yaml
python scripts/generate.py --config configs/pixeldit_standard_average.yaml
```

## Required staged validation

Real-model checks are separate from the CPU suite and should run in this order:

```bash
# A/B: official sampler and DiffPano order-1 loop on identical full-image noise.
python scripts/pixeldit_single_view_test.py --save-official-order-two

# C: full-image ViewDenoiser path, no projection/fusion.
python scripts/pixeldit_full_image_test.py

# D: one camera projected from and fused back to ERP.
python scripts/pixeldit_one_view_erp_test.py

# E: all 89 cameras, standard warp and average.
python scripts/generate.py --config configs/pixeldit_smoke.yaml
```

The first script saves DiffPano order-1, official order-1, optional official order-2 images, their order-1 numerical error, provenance, and peak GPU memory. The full-image command intentionally calls the same `ViewDenoiser` trajectory with identical seed and dimensions, so its image must equal the DiffPano image from the first command. The one-view script records final state statistics and peak memory.

## Geometry-only distribution diagnostics

PixelDiT can record, per global step, source/fused ERP mean, standard deviation, min/max, L2 norm, horizontal/vertical correlation; streaming source-view, flow-prediction, and proposal moments; and the fused/source standard-deviation ratio. Scalar statistics are retained without large image dumps.

To isolate geometry from the model:

```bash
python scripts/test_pixel_state_warp_statistics.py \
  --steps 50 --erp-to-view nearest --view-to-erp bilinear \
  --output outputs/pixel-state-warp-statistics.json
```

This uses an identity local model and repeats projection/inverse projection/plain averaging on Gaussian ERP noise.

## Other configuration

All existing geometry remains backend-neutral:

- `sampling.strategy`: `spherediff_fixed` or `spherediff_rotated`;
- `warp.mode`: `standard` or `lpw`;
- either `nearest` or `bilinear` independently in both directions;
- `fusion.mode`: `average`, `weighted_average`, or `detail_preserving_average`;
- weights: `uniform`, `cosine`, `gaussian`, `distance_to_boundary`, or `spherediff_center`.

`performance.view_batch_size` batches same-sized camera states without changing synchronous semantics. `performance.projection_cache_max_entries` bounds each device geometry cache; `null` retains the original unbounded behavior for existing experiments. `performance.projection_cache_cpu_fallback` preserves evicted grids in host RAM for fixed-cover reuse. `debug.save_step_indices` and `debug.save_view_indices` restrict optional image dumps.

## CPU tests and legacy code

```bash
python -m unittest discover -s tests -v
```

These tests do not download a real model. When the pinned source checkout is present, the order-1 primitive test imports only its solver code. Existing SANA, FLUX, SD2, LPW, DPA, camera covers, and archived spherical baseline remain intact.
