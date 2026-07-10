You are working inside an existing SphereDiff repository. The environment, pretrained models, and inference dependencies are already available.

The untouched SphereDiff baseline is stored in another repository. You do not need to preserve latent-space fusion as a runtime option here.

Your task is to replace SphereDiff’s latent-space aggregation with ERP-based pixel-space fusion inspired by LookingGlass, while making the smallest possible changes to the existing architecture.

# 1. Non-negotiable requirements

1. Preserve SphereDiff’s existing model calls, prompt conditioning, scheduler, spherical latent representation, camera definitions, and dynamic sampling.
2. Modify only the view aggregation/write-back path unless a small adapter is required.
3. Reuse existing geometry, projection, VAE, scheduler, and logging utilities.
4. Do not introduce a new pipeline framework or rewrite unrelated code.
5. Implement every new feature as a separate function or small module with an explicit configuration flag for ablation studies.
6. Add clear comments around new geometry, tensor shapes, scheduler conversions, and write-back logic.
7. ERP is the temporary canonical pixel space. Do not implement a spherical-pixel representation in this version.

# 2. Inspect before editing

Before changing code, identify and report:

* inference entry point;
* denoising loop;
* original dynamic sampler and its state;
* view extraction function;
* current latent aggregation/write-back function;
* perspective/ERP/spherical projection utilities;
* VAE scaling and image-range conventions;
* scheduler type, timestep representation, and prediction type;
* current configuration and SLURM arguments;
* exact files that need modification.

Then make the smallest targeted patch.

# 3. Required pipeline

Keep the original SphereDiff dynamic-sampling loop.

For each denoising timestep or dynamic-sampling update group:

```text
original dynamic sampler selects perspective patches
→ original view extraction
→ original model prediction
→ convert prediction to predicted-clean view latents x0
→ optional pixel-space fusion
→ convert fused clean latents to the representation expected by write-back
→ original spherical-latent write-back
```

Pixel-space fusion must perform:

```text
predicted-clean view latents x0
→ VAE decode to RGB
→ perspective-to-ERP warp
→ joint fusion of all overlapping patches
→ ERP-to-perspective warp for the same sampled patches
→ VAE encode to fused clean latents
→ scheduler-consistent reinjection
```

Do not decode arbitrary noisy latents unless the existing scheduler explicitly defines that as correct. Pixel fusion should normally operate on the predicted-clean sample.

# 4. Preserve dynamic sampling

Do not replace SphereDiff’s dynamic sampler with fixed views, all-view processing, raster traversal, or a new view scheduler.

The new fusion module must consume exactly the patches selected by the original dynamic sampler.

The sampled patch count, camera directions, order, and sampler state may vary across timesteps.

If the current code processes sampled patches one at a time, minimally buffer the patches belonging to the same update group so their overlaps can be fused jointly before write-back. Do not change which patches are sampled.

For time travel, rerun the original dynamic sampler at each repeated denoising step.

# 5. ERP canonical space

ERP is a temporary canvas created only for the current fusion call.

Do not maintain a persistent ERP image across timesteps in the first implementation. SphereDiff’s spherical latent remains the persistent global state.

ERP projection must:

* wrap horizontally at longitude ±π;
* correctly handle seam-crossing patches;
* produce a validity mask for each patch;
* use the existing SphereDiff camera and spherical-coordinate conventions;
* cache projection grids when view metadata and resolution are unchanged.

When a sampled output patch reads an invalid ERP location, fall back to that patch’s original decoded predicted-clean RGB value. Do not fill invalid ERP regions with unrelated data.

# 6. Joint handling of overlapping patches

Several sampled patches may cover the same ERP pixel. This is expected.

Never use sequential overwrite:

```python
erp[..., y, x] = patch[..., y, x]
```

Do not use order-dependent pairwise blending.

All valid contributions for the same ERP pixel, or the same ERP pyramid coefficient, must be fused jointly across the patch dimension.

Use shapes equivalent to:

```python
projected_rgb:     [num_patches, 3, H_erp, W_erp]
projected_mask:    [num_patches, 1, H_erp, W_erp]
projected_weight:  [num_patches, 1, H_erp, W_erp]
```

The result must be independent of patch order.

For ordinary weighted fusion:

```python
effective_weight = projected_mask * projected_weight
fused = sum(projected_rgb * effective_weight, dim=patch)
fused /= clamp_min(sum(effective_weight, dim=patch), eps)
```

Implement a central function such as:

```python
aggregate_overlap_contributions(
    values,
    masks,
    weights,
    mode,
    ...
)
```

It must support arbitrary patch counts and return at least:

* fused values;
* accumulated weight;
* contributor count;
* valid output mask.

Memory-efficient sufficient-statistic accumulation is acceptable, provided the result remains joint and order-independent.

# 7. Independent warp and aggregation ablations

Warping and aggregation are separate components and must have separate configuration options.

```python
warp_mode:
    "standard"
    "lpw"

aggregation_mode:
    "average"
    "weighted_average"
    "detail_preserving_average"
```

Required ablations:

```text
standard warp + average
standard warp + weighted average
standard warp + DPA
LPW + average
LPW + weighted average
LPW + DPA
```

Do not hide LPW and DPA behind one combined flag.

# 8. Standard perspective–ERP warp

Implement a basic perspective-to-ERP and ERP-to-perspective path using the existing SphereDiff geometry.

This path is required for debugging and ablation.

Support boundary weights computed in perspective-patch coordinates before projection:

```python
weight_mode:
    "uniform"
    "cosine"
    "gaussian"
    "distance_to_boundary"
```

Boundary weighting must remain independent from DPA and LPW.

# 9. Laplacian Pyramid Warping

Implement LookingGlass-inspired LPW for perspective patches and the ERP canonical space.

When `warp_mode == "lpw"`:

1. Build Gaussian/Laplacian pyramids for each predicted-clean RGB patch.
2. Compute the local perspective-to-ERP scale or LOD from the real projection mapping, preferably from its Jacobian or image-space derivatives.
3. Inverse-warp each patch pyramid into an ERP canonical pyramid.
4. Preserve a validity mask and geometric confidence for every patch and pyramid level.
5. Fuse all overlapping patch coefficients jointly at each level.
6. Reconstruct the fused ERP RGB image.
7. Use the corresponding forward LPW operation to sample the fused ERP result back into the sampled perspective patches.

Do not approximate LPW as ordinary full-resolution warping followed by a Laplacian blend.

Reuse the repository’s existing projection equations. Add resolution parameters or small wrappers rather than duplicating camera math.

ERP pyramid filtering must use:

* circular padding horizontally;
* reflection or replication vertically;
* no zero-padding seam at the left/right ERP boundary.

Cache static projection grids and LOD maps.

Expose at least:

```python
lpw_num_levels
lpw_lod_mode
lpw_lod_interpolation
erp_vertical_padding_mode
```

`warp_mode` already controls whether LPW is enabled, so do not add a redundant `lpw_enabled` flag unless required by the existing configuration style.

# 10. Detail-Preserving Average

Implement DPA independently from LPW.

For values `x_i`, masks `m_i`, geometric weights `w_i`, detail power `q`, and epsilon `eps`:

```python
ordinary = (
    sum(m_i * w_i * x_i)
    / clamp_min(sum(m_i * w_i), eps)
)

detail = (
    sum(m_i * w_i * (abs(x_i) + eps) ** q * x_i)
    / clamp_min(
        sum(m_i * w_i * (abs(x_i) + eps) ** q),
        eps,
    )
)

output = ordinary + alpha * (detail - ordinary)
```

Use this multi-input form directly. Do not repeatedly apply a two-image DPA.

Apply DPA jointly:

* across all patches covering one ERP pixel for standard warping;
* across all patches covering one ERP coefficient at each pyramid level for LPW.

Expose:

```python
dpa_alpha       # 0 = ordinary weighted average, 1 = full DPA
dpa_power       # default 1
dpa_eps
```

A single valid contributor must be returned unchanged.

# 11. VAE encode/decode

Use the current model’s actual VAE conventions.

Do not hard-code the latent scaling factor. Use the existing value, such as `vae.config.scaling_factor`, if that is what SphereDiff uses.

Requirements:

* operate under inference/no-grad mode;
* preserve device and dtype;
* support configurable chunking;
* flatten and restore batch/view dimensions explicitly;
* use posterior mean by default for deterministic encoding;
* optionally allow posterior sampling;
* convert RGB ranges correctly;
* never clamp latent tensors;
* clamp RGB only when required by the VAE contract or when saving images.

Document the actual tensor shapes in comments.

# 12. Reinjection

After fusion:

1. Sample or forward-warp the fused ERP result into the same dynamically sampled views.
2. VAE-encode those RGB views into fused predicted-clean latents.
3. Convert them into the representation expected by SphereDiff’s existing write-back path.

Implement separate modes:

```python
reinjection_mode:
    "noise_consistent"
    "replace"
    "weighted_replace"
    "residual"
```

`noise_consistent` should preserve the current timestep’s noise or flow state while replacing or correcting the predicted-clean content.

Derive this conversion from the scheduler and model prediction type actually used by SphereDiff. Do not assume epsilon prediction or DDPM equations.

If a scheduler does not support a correct noise-consistent conversion, raise a clear error rather than silently using an incorrect approximation. Other reinjection modes should remain available for ablation.

Expose:

```python
reinjection_strength
```

# 13. Time travel

Implement time travel only after ordinary pixel fusion works.

It must be optional and minimally invasive:

```text
reach a configured timestep
→ perform pixel fusion
→ jump to an earlier/noisier scheduler state
→ rerun the short denoising interval
→ resume normal inference
```

Requirements:

* use scheduler-consistent noising;
* reuse the original denoising code;
* rerun SphereDiff’s original dynamic sampling on every repeated step;
* do not recursively call the full pipeline;
* do not redesign the scheduler loop.

Expose:

```python
time_travel_enabled
time_travel_every_n_steps
time_travel_jump_length
time_travel_num_repeats
time_travel_strength
```

# 14. Configuration

Create one clear configuration file using the repository’s existing style. It may be YAML, a dataclass, or the project’s current configuration system.

Move relevant parameters currently hard-coded in the inference script or `a.slurm` into this configuration where practical. Keep SLURM responsible only for cluster resources, environment setup, and launching the command.

At minimum expose:

```python
pixel_fusion_enabled
pixel_fusion_every_n_steps
pixel_fusion_start_ratio
pixel_fusion_end_ratio

warp_mode
aggregation_mode
weight_mode

lpw_num_levels
lpw_lod_mode
lpw_lod_interpolation
erp_vertical_padding_mode

dpa_alpha
dpa_power
dpa_eps

reinjection_mode
reinjection_strength

time_travel_enabled
time_travel_every_n_steps
time_travel_jump_length
time_travel_num_repeats
time_travel_strength

vae_chunk_size
save_intermediates
save_masks
save_diagnostics
```

Every option must be passed as a function argument or through the configuration object. Avoid hidden global state.

# 15. Suggested modular functions

Use small functions or wrappers. Names may follow the repository style.

At minimum separate:

```python
should_apply_pixel_fusion(...)
predict_clean_latents(...)
decode_view_latents(...)
encode_view_images(...)

project_views_to_erp_standard(...)
extract_views_from_erp_standard(...)

build_gaussian_pyramid(...)
build_laplacian_pyramid(...)
reconstruct_laplacian_pyramid(...)
inverse_lpw_to_erp(...)
forward_lpw_to_views(...)

create_patch_weight_map(...)
aggregate_overlap_contributions(...)
detail_preserving_average(...)

apply_pixel_space_fusion(...)
reinject_fused_latents(...)

should_apply_time_travel(...)
run_time_travel(...)
```

The main denoising loop should receive only a small conditional call to `apply_pixel_space_fusion(...)`, followed by the existing write-back call.

# 16. File organization

Prefer:

* one new pixel-fusion module;
* one configuration file;
* one optional time-travel module;
* minimal edits to the existing pipeline;
* minimal edits to projection utilities only when RGB resolution or LPW support requires them.

Follow the repository’s directory and naming conventions. Do not create a large new package hierarchy.

# 17. Diagnostics

Diagnostics must be optional and must not affect generation when disabled.

Useful outputs include:

```python
fused_erp
valid_mask
contributor_count
accumulated_weight
overlap_mask
dominant_patch_index
overlap_variance
latent_delta_norm
sampled_patch_indices
sampled_camera_directions
```

Use one centralized result writer. Do not scatter image-saving code throughout the denoising loop.

# 18. Tests

Add focused tests for:

1. One patch returns itself.
2. Identical overlapping patches remain unchanged.
3. Three or more overlapping patches are fused correctly.
4. Changing patch order does not change the result.
5. Invalid patches contribute zero.
6. Invalid ERP pixels produce no NaNs.
7. DPA with `alpha=0` equals weighted averaging.
8. A single valid DPA contributor remains unchanged.
9. Laplacian-pyramid reconstruction reproduces the input within tolerance.
10. ERP horizontal seam padding is circular.
11. A seam-crossing perspective-to-ERP-to-perspective round trip works.
12. Reinjection strengths 0 and 1 behave correctly.
13. Disabling pixel fusion leaves the existing path unchanged.
14. Fixed seeds produce deterministic results.
15. VAE encode/decode shapes and ranges are correct.

Run one small end-to-end smoke test using the installed model.

Do not claim a test passed unless it was actually executed.

# 19. Performance

Avoid unnecessary overhead:

* decode/encode only on scheduled fusion steps;
* use mixed precision consistent with SphereDiff;
* support VAE chunking;
* cache projection grids, patch weights, and LOD maps;
* avoid CPU transfers;
* disable diagnostics and saving by default;
* use inference mode;
* avoid storing every projected patch when order-independent accumulators are sufficient.

Measure:

```text
vae_decode
projection_or_inverse_lpw
overlap_fusion
erp_reconstruction
erp_to_view_or_forward_lpw
vae_encode
reinjection
time_travel
```

# 20. Implementation order

## Phase 1: inspect and report

Report the existing architecture and exact modification points before editing.

## Phase 2: standard pixel-fusion baseline

Implement:

```text
predicted-clean latent
→ decode
→ standard perspective-to-ERP warp
→ joint average/weighted fusion
→ ERP-to-perspective sampling
→ encode
→ reinject
```

Run tests and a smoke test.

## Phase 3: DPA

Add multi-input, order-independent DPA and its ablations.

## Phase 4: LPW

Add inverse/forward LPW, LOD computation, seam-aware pyramids, and per-level joint fusion.

## Phase 5: time travel

Add optional time travel without changing the original dynamic sampler.

# 21. Final report

After implementation, report:

* modified and added files;
* why each existing file changed;
* approximate changed-line count in existing files;
* exact location of the pixel-fusion call;
* exact location of the preserved dynamic sampler;
* tensor shapes through the fusion pipeline;
* commands or configurations for all required ablations;
* tests actually executed and their results;
* smoke-test result;
* performance timings;
* unsupported or unvalidated behavior.

The priority is minimal invasive change to SphereDiff’s existing architecture, not minimizing the amount of new research code.
