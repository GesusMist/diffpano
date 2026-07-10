You are working inside an existing SphereDiff repository. The repository, pretrained models, environment, and inference dependencies are already available locally.

We maintain the untouched original SphereDiff implementation in another repository. Therefore, you do not need to preserve the original latent-blending behavior as a runtime baseline in this repository.

However, you must preserve the current SphereDiff software architecture as much as possible.

The implementation principle is:

```text
minimum modification to existing code
+ reuse existing SphereDiff modules
+ add new research functionality through small modular functions
+ avoid large-scale refactoring
+ avoid duplicating existing projection, scheduler, model, or view-management code
```

The research goal is to replace SphereDiff’s latent-space blending with pixel-space, detail-preserving blending inspired by LookingGlass, while retaining SphereDiff’s spherical multi-view denoising framework.

Every added research feature must be implemented as a separate function or small module with explicit parameters controlling whether it is enabled. This is required for ablation studies.

# 1. Main design

Keep SphereDiff’s existing overall pipeline:

```text
spherical/global latent state
→ extract perspective-view latent
→ denoise each perspective view
→ aggregate the updated views
→ continue to next diffusion timestep
```

Modify only the aggregation portion.

The new main pipeline should be:

```text
spherical/global latent state
→ extract perspective-view latent
→ denoise each perspective view
→ decode denoised view latents into RGB
→ project RGB views into a canonical ERP canvas
→ perform pixel-space fusion
→ extract fused RGB perspective views from ERP
→ encode fused RGB views back into latent space
→ write/reinject these latents into SphereDiff’s existing spherical latent state
→ continue to the next diffusion timestep
```

Do not redesign SphereDiff’s model invocation, camera definitions, view extraction system, prompt conditioning, scheduler setup, or spherical latent representation unless a minimal adapter is necessary.

# 2. Minimal-modification requirement

Before editing any file, inspect the repository and identify:

1. the inference entry point;
2. the main denoising loop;
3. where perspective views are extracted from the spherical latent representation;
4. where denoised views are currently written back or blended;
5. the existing projection and inverse-projection utilities;
6. the existing VAE encode/decode utilities;
7. scheduler type and prediction type;
8. configuration and command-line structure.

Then make the smallest possible patch.

Prefer this pattern:

```python
if pixel_fusion_config.enabled:
    updated_views = apply_pixel_space_fusion(...)
else:
    updated_views = existing_view_update
```

Do not move the denoising loop into a new framework unless absolutely necessary.

Do not introduce a new pipeline class if the existing pipeline class can be extended with one or two helper calls.

Do not rewrite projection code already available in SphereDiff.

Do not rewrite the spherical latent representation.

Do not replace existing camera metadata structures.

Do not introduce unnecessary abstractions that change how the rest of SphereDiff operates.

# 3. Scope of the first implementation

Implement the following components:

1. RGB decoding of denoised perspective-view latents.
2. Projection of RGB perspective views into ERP space.
3. Simple weighted RGB fusion in ERP space.
4. LookingGlass-inspired Laplacian-pyramid detail-preserving fusion.
5. Extraction of fused RGB perspective views from ERP.
6. VAE re-encoding of fused RGB views.
7. Reinjection of fused view latents into SphereDiff’s existing spherical latent state.
8. Optional pixel fusion at selected denoising steps.
9. Optional time travel.
10. Configurable diagnostics and intermediate output.

The canonical pixel representation should initially be ERP.

Do not implement a separate complex spherical-pixel storage format at this stage. Instead, encapsulate ERP projection and sampling operations cleanly enough that another spherical representation could be introduced later.

# 4. Feature-control requirements

Every added component must be independently configurable.

At minimum, expose the following options:

```python
pixel_fusion_enabled: bool
pixel_fusion_mode: str
pixel_fusion_every_n_steps: int
pixel_fusion_start_ratio: float
pixel_fusion_end_ratio: float

blend_weight_mode: str
laplacian_levels: int
high_frequency_mode: str
high_frequency_temperature: float

reinjection_mode: str
reinjection_strength: float

spherical_area_weighting_enabled: bool

time_travel_enabled: bool
time_travel_every_n_steps: int
time_travel_jump_length: int
time_travel_num_repeats: int

save_intermediates: bool
save_masks: bool
save_diagnostics: bool
```

Suggested fusion modes:

```python
"rgb_average"
"rgb_weighted"
"rgb_laplacian"
```

Suggested high-frequency modes:

```python
"weighted"
"winner_take_most"
```

Suggested reinjection modes:

```python
"replace"
"weighted_replace"
"residual"
"noise_consistent"
```

The new code does not need to keep latent blending as an available runtime mode unless retaining it requires no additional effort.

# 5. Configuration design

Follow the repository’s current configuration convention.

If SphereDiff uses argparse, add argparse options.

If it uses YAML or OmegaConf, add fields to the existing configuration.

If it uses a pipeline constructor, add a small configuration dataclass or dictionary passed into the pipeline.

Avoid creating a second independent configuration system.

A possible lightweight configuration object is:

```python
from dataclasses import dataclass
from typing import Literal


@dataclass
class PixelFusionConfig:
    enabled: bool = True

    mode: Literal[
        "rgb_average",
        "rgb_weighted",
        "rgb_laplacian",
    ] = "rgb_laplacian"

    every_n_steps: int = 1
    start_ratio: float = 0.0
    end_ratio: float = 1.0

    weight_mode: Literal[
        "uniform",
        "cosine",
        "gaussian",
        "distance_to_boundary",
    ] = "cosine"

    spherical_area_weighting_enabled: bool = True

    laplacian_levels: int = 5
    high_frequency_mode: Literal[
        "weighted",
        "winner_take_most",
    ] = "winner_take_most"
    high_frequency_temperature: float = 0.1

    reinjection_mode: Literal[
        "replace",
        "weighted_replace",
        "residual",
        "noise_consistent",
    ] = "weighted_replace"

    reinjection_strength: float = 1.0

    save_intermediates: bool = False
    save_masks: bool = False
    save_diagnostics: bool = False


@dataclass
class TimeTravelConfig:
    enabled: bool = False
    every_n_steps: int = 0
    jump_length: int = 1
    num_repeats: int = 1
    strength: float = 1.0
```

Adapt names and placement to the repository style.

# 6. Required functions

Add small reusable functions rather than placing all logic inside the denoising loop.

At minimum, implement or wrap the following functions:

```python
def should_apply_pixel_fusion(
    step_index,
    total_steps,
    config,
) -> bool:
    ...
```

```python
def decode_view_latents(
    view_latents,
    vae,
    scaling_factor,
    chunk_size=None,
):
    ...
```

```python
def encode_view_images(
    view_images,
    vae,
    scaling_factor,
    sample_mode="mean",
    generator=None,
    chunk_size=None,
):
    ...
```

```python
def project_views_to_erp(
    view_images,
    view_metadata,
    erp_height,
    erp_width,
    projection_cache=None,
):
    ...
```

```python
def extract_views_from_erp(
    erp_image,
    view_metadata,
    output_height,
    output_width,
    projection_cache=None,
):
    ...
```

```python
def compute_blending_weights(
    projected_masks,
    view_metadata,
    mode="cosine",
    spherical_area_weighting_enabled=True,
):
    ...
```

```python
def blend_rgb_average(
    projected_images,
    projected_masks,
):
    ...
```

```python
def blend_rgb_weighted(
    projected_images,
    projected_weights,
    projected_masks,
    eps=1e-8,
):
    ...
```

```python
def laplacian_pyramid_blend(
    projected_images,
    projected_weights,
    projected_masks,
    num_levels=5,
    high_frequency_mode="winner_take_most",
    temperature=0.1,
):
    ...
```

```python
def reinject_fused_view_latents(
    current_view_latents,
    fused_view_latents,
    mode="weighted_replace",
    strength=1.0,
    timestep=None,
    scheduler=None,
    noise=None,
):
    ...
```

```python
def apply_pixel_space_fusion(
    denoised_view_latents,
    current_view_latents,
    view_metadata,
    timestep,
    vae,
    scheduler,
    projection_utils,
    config,
    generator=None,
):
    """
    Returns:
        fused_view_latents
        fused_erp_rgb
        diagnostics
    """
    ...
```

The main denoising loop should only need a small insertion similar to:

```python
if should_apply_pixel_fusion(
    step_index,
    len(timesteps),
    pixel_fusion_config,
):
    denoised_view_latents, fused_erp, diagnostics = (
        apply_pixel_space_fusion(
            denoised_view_latents=denoised_view_latents,
            current_view_latents=current_view_latents,
            view_metadata=view_metadata,
            timestep=t,
            vae=self.vae,
            scheduler=self.scheduler,
            projection_utils=existing_projection_utils,
            config=pixel_fusion_config,
            generator=generator,
        )
    )
```

Then use SphereDiff’s existing function for writing view latents back into the spherical latent state.

# 7. Reuse the existing SphereDiff geometry

SphereDiff already contains mappings between:

* spherical latent representation;
* perspective latent grids;
* camera directions;
* field of view;
* possibly ERP or spherical coordinates.

Reuse those mappings wherever possible.

Do not maintain two separate definitions of camera orientation or spherical coordinates.

If existing projection utilities only support latent tensors, inspect whether they are resolution-agnostic. If they are, reuse them for RGB tensors by changing only the input spatial dimensions.

If they assume latent resolution, add a small resolution parameter rather than duplicating the implementation.

If RGB and latent projections require different grids, generate them through the same underlying camera-coordinate function.

Cache projection grids by:

```text
view direction
field of view
input resolution
output resolution
device
dtype
```

# 8. ERP canvas

Use ERP as the canonical pixel-space fusion canvas.

The ERP canvas must:

* wrap horizontally at longitude ±π;
* correctly handle views crossing the left/right ERP boundary;
* create a valid mask for every projected perspective image;
* avoid NaNs in uncovered regions;
* optionally apply spherical-area weighting;
* use the existing panorama resolution or a configurable ERP resolution.

Implement a lightweight helper only if necessary:

```python
class ERPPixelCanvas:
    def __init__(
        self,
        height,
        width,
        device,
        dtype,
    ):
        ...

    def project_view(
        self,
        rgb_view,
        view_metadata,
    ):
        ...

    def extract_view(
        self,
        erp_rgb,
        view_metadata,
        output_size,
    ):
        ...
```

This class should wrap the repository’s existing geometry functions. It should not become a replacement for SphereDiff’s spherical latent representation.

# 9. VAE decoding and encoding

Inspect the exact VAE convention used by the current SphereDiff model.

Do not hard-code the latent scaling factor.

Use:

```python
vae.config.scaling_factor
```

or the equivalent value used by the existing code.

Requirements:

* preserve batch dimension;
* preserve view dimension or flatten and restore it explicitly;
* preserve dtype and device;
* support chunking;
* use `torch.inference_mode()`;
* convert RGB ranges correctly;
* use the posterior mean by default for deterministic re-encoding;
* optionally allow posterior sampling;
* do not clamp latent tensors;
* clamp RGB only when saving or when required by the VAE input contract.

Document tensor shapes clearly, for example:

```python
view_latents: [num_views, latent_channels, latent_height, latent_width]
view_images:  [num_views, 3, image_height, image_width]
projected:    [num_views, 3, erp_height, erp_width]
masks:        [num_views, 1, erp_height, erp_width]
weights:      [num_views, 1, erp_height, erp_width]
```

# 10. Simple RGB fusion baselines

Implement two simple pixel-space baselines.

## RGB average

```python
fused = sum(image_i * mask_i) / sum(mask_i)
```

## RGB weighted

```python
fused = sum(image_i * weight_i * mask_i) \
        / sum(weight_i * mask_i)
```

Where no view contributes, use one of the following, selected by a parameter:

```python
"keep_previous"
"zero"
"nearest_valid"
```

Default to keeping the previous ERP canvas where available.

These simple modes are required for ablation studies and debugging.

# 11. Detail-preserving fusion

Implement a LookingGlass-inspired multiscale pixel-space blending method using Laplacian pyramids.

Required helpers:

```python
def build_gaussian_pyramid(
    tensor,
    num_levels,
):
    ...
```

```python
def build_laplacian_pyramid(
    tensor,
    num_levels,
):
    ...
```

```python
def reconstruct_laplacian_pyramid(
    levels,
):
    ...
```

```python
def build_weight_pyramid(
    weights,
    num_levels,
):
    ...
```

At every pyramid level:

* use smooth weighted blending for low-frequency components;
* preserve locally reliable high-frequency detail;
* prevent ordinary averaging from blurring textures;
* account for masks and invalid regions.

Support two high-frequency strategies.

## Weighted

```python
fused_level = sum(
    laplacian_i * normalized_weight_i
)
```

## Winner-take-most

Use a temperature-controlled softmax across views:

```python
level_weights = torch.softmax(
    confidence / temperature,
    dim=0,
)
```

A low temperature should approximate selecting the locally dominant view while remaining differentiable and numerically stable.

Add a separate function:

```python
def fuse_laplacian_level(
    level_images,
    level_weights,
    level_masks,
    mode,
    temperature,
    eps=1e-8,
):
    ...
```

ERP pyramid filtering must respect horizontal periodicity. Avoid zero padding at the left and right panorama boundaries. Use circular horizontal padding where applicable.

Vertical padding may use reflection or replication.

# 12. Weight generation

Implement weight generation separately from fusion.

Support:

```python
"uniform"
"cosine"
"gaussian"
"distance_to_boundary"
```

The default should reduce confidence near perspective-image boundaries.

Possible interface:

```python
def create_view_weight_map(
    height,
    width,
    mode,
    device,
    dtype,
    sigma=0.5,
):
    ...
```

After projection to ERP, combine:

```python
final_weight = (
    projected_view_weight
    * projected_valid_mask
    * optional_spherical_area_weight
)
```

Keep spherical-area weighting behind a separate Boolean flag.

Do not combine different ablation features implicitly.

# 13. Reinjection

After ERP fusion:

1. sample the fused ERP back into every perspective camera;
2. encode those RGB views through the VAE;
3. combine the re-encoded latents with the current denoised view latents;
4. pass the result into SphereDiff’s existing write-back mechanism.

Implement these independently selectable modes.

## Replace

```python
output = fused_latent
```

## Weighted replacement

```python
output = (
    1.0 - strength
) * current_latent \
    + strength * fused_latent
```

## Residual correction

```python
correction = fused_latent - current_latent
output = current_latent + strength * correction
```

## Noise-consistent reinjection

When possible, map the re-encoded fused latent to the noise level associated with the current diffusion timestep before writing it back.

Inspect the actual scheduler and model prediction type.

Do not assume DDPM epsilon prediction.

Check whether the implementation uses:

```text
epsilon prediction
v prediction
sample/x0 prediction
flow prediction
```

Use the scheduler’s existing formulas and APIs.

If noise-consistent reinjection is unsupported for the current scheduler, implement a clear guarded error and keep the other modes operational.

# 14. Time travel

Implement time travel as an optional feature with minimum disturbance to the denoising loop.

Do not refactor the full pipeline solely to support time travel.

First inspect whether the existing denoising step can be called repeatedly over selected scheduler indices.

Add only the smallest helper needed.

Required behavior:

```text
denoise to current step
→ perform pixel fusion
→ add scheduler-consistent noise to jump to an earlier/noisier step
→ repeat denoising over the selected short interval
→ optionally perform pixel fusion again
→ resume normal denoising
```

Parameters:

```python
enabled
every_n_steps
jump_length
num_repeats
strength
```

Add:

```python
def should_apply_time_travel(
    step_index,
    total_steps,
    config,
) -> bool:
    ...
```

```python
def add_time_travel_noise(
    latents,
    current_timestep,
    target_timestep,
    scheduler,
    noise,
    strength,
):
    ...
```

```python
def run_time_travel(
    latents,
    current_step_index,
    timesteps,
    denoise_step_fn,
    fusion_fn,
    config,
    generator=None,
):
    ...
```

Reuse the existing denoising-step code. If it is currently embedded in the loop, extract only that small block into a function.

Do not redesign the full pipeline.

# 15. Main-loop modification target

The preferred main-loop patch should be conceptually small:

```python
for step_index, t in enumerate(timesteps):
    # Existing SphereDiff view extraction
    view_latents = extract_existing_views(...)

    # Existing model denoising
    denoised_view_latents = denoise_existing_views(...)

    # New optional RGB-space fusion
    if should_apply_pixel_fusion(
        step_index,
        len(timesteps),
        pixel_fusion_config,
    ):
        denoised_view_latents, fused_erp, diagnostics = (
            apply_pixel_space_fusion(
                denoised_view_latents,
                view_latents,
                view_metadata,
                t,
                self.vae,
                self.scheduler,
                existing_projection_utils,
                pixel_fusion_config,
                generator,
            )
        )

    # Existing SphereDiff write-back operation
    spherical_latent = existing_write_back(
        spherical_latent,
        denoised_view_latents,
        view_metadata,
    )

    # New optional time travel
    if should_apply_time_travel(...):
        spherical_latent = run_time_travel(...)
```

Adapt this to the real code structure.

The actual modification to the main loop should ideally be limited to a few function calls and configuration checks.

# 16. File organization

Prefer adding one or two focused files rather than many new packages.

A possible layout is:

```text
spherediff/
    existing_pipeline_file.py          # minimal edits
    existing_projection_file.py        # only if RGB resolution support is needed
    pixel_fusion.py                    # new
    time_travel.py                     # new, only if sufficiently independent
```

Possible contents of `pixel_fusion.py`:

```text
VAE decode/encode helpers
ERP projection wrappers
weight generation
simple blending
Laplacian blending
reinjection
diagnostics
```

Avoid distributing one feature across many unrelated files.

Follow the repository’s current naming and directory conventions instead of imposing this exact structure.

# 17. Diagnostics

Diagnostics must not affect generation when disabled.

Add optional outputs:

```python
{
    "fused_erp": ...,
    "valid_mask": ...,
    "accumulated_weight": ...,
    "view_count": ...,
    "dominant_view": ...,
    "overlap_variance": ...,
    "latent_delta_norm": ...,
}
```

Implement:

```python
def compute_fusion_diagnostics(
    projected_images,
    projected_weights,
    projected_masks,
    fused_erp,
    current_latents=None,
    fused_latents=None,
):
    ...
```

Only compute expensive diagnostics when enabled.

# 18. Intermediate saving

Use one centralized helper instead of scattered saving calls.

```python
class PixelFusionResultWriter:
    def __init__(
        self,
        output_dir,
        enabled=False,
    ):
        ...

    def save_erp(self, image, step_index):
        ...

    def save_mask(self, mask, name, step_index):
        ...

    def save_diagnostics(self, diagnostics, step_index):
        ...
```

Only instantiate or call it when saving is enabled.

Do not introduce CPU synchronization during every timestep unless required.

# 19. Tests

Add focused tests for the new functions without building a second test framework.

Required tests:

1. one-view fusion returns that view;
2. identical overlapping views remain unchanged;
3. weighted fusion normalizes correctly;
4. zero-contribution pixels do not produce NaNs;
5. Laplacian reconstruction approximately reproduces the original image;
6. circular ERP seam padding works;
7. perspective-to-ERP-to-perspective round trip;
8. reinjection strength zero returns the current latent;
9. reinjection strength one returns the fused latent;
10. disabled pixel fusion does not alter its input;
11. fixed seeds produce deterministic output;
12. VAE encode/decode maintains expected shape and valid range.

Also run one small end-to-end smoke test using the locally available model.

Do not claim that tests passed unless they were actually executed.

# 20. Performance

The VAE encode/decode round trip is expensive.

Implement:

* optional VAE chunking;
* mixed precision consistent with the current pipeline;
* cached projection grids;
* cached view weight maps;
* no RGB decoding on steps where fusion is disabled;
* no diagnostic computation when disabled;
* no saving when disabled;
* no unnecessary CPU transfers;
* no gradients.

Add lightweight timing measurements for:

```text
vae_decode
view_to_erp_projection
pixel_blending
erp_to_view_sampling
vae_encode
latent_reinjection
time_travel
```

# 21. Coding style

Use:

* type hints;
* concise docstrings;
* explicit tensor-shape comments;
* assertions for unexpected dimensions;
* device- and dtype-safe tensor creation;
* passed `torch.Generator` objects for random noise;
* the repository’s existing logging utility.

Avoid:

* global mutable state;
* hard-coded CUDA devices;
* hard-coded latent scaling constants;
* hard-coded view counts;
* hard-coded ERP dimensions;
* duplicated camera math;
* large architecture changes;
* changing unrelated code formatting;
* renaming existing public functions unnecessarily.

# 22. Implementation sequence

Proceed in this order.

## Step 1: repository analysis

Before editing, report:

1. inference entry point;
2. denoising-loop file and function;
3. current view extraction function;
4. current view write-back/blending function;
5. projection utilities;
6. VAE conventions;
7. scheduler and prediction type;
8. exact files that require modification.

## Step 2: minimal RGB fusion baseline

Implement:

* configuration fields;
* VAE decode and encode;
* projection wrappers;
* RGB average;
* RGB weighted fusion;
* reinjection;
* minimal denoising-loop insertion.

Run a small smoke test.

## Step 3: detail-preserving fusion

Implement:

* Gaussian pyramid;
* Laplacian pyramid;
* seam-aware filtering;
* weighted high-frequency fusion;
* winner-take-most high-frequency fusion.

Run unit tests and a smoke test.

## Step 4: time travel

Implement time travel through the smallest possible reuse of the current denoising-step code.

Do not begin with time travel before basic pixel fusion works.

# 23. Required final report

After implementation, report:

1. modified files;
2. newly added files;
3. why each existing file had to be changed;
4. approximate number of changed lines in existing files;
5. where the new fusion call was inserted;
6. tensor shapes at every stage;
7. commands for:

   * RGB average;
   * RGB weighted;
   * Laplacian fusion;
   * Laplacian fusion with winner-take-most detail;
   * Laplacian fusion plus time travel;
8. tests actually executed;
9. smoke-test result;
10. unvalidated or unsupported behavior.

The most important criterion is not minimizing the total number of new lines. The most important criterion is minimizing invasive changes to SphereDiff’s existing architecture.

New research logic may live in separate modules, while existing SphereDiff files should receive only small, targeted modifications.
