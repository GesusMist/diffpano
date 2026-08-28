# Dynamic planar patches

The planar SANA, FLUX, and SD2 pipelines support two patch-layout strategies:

```yaml
planar:
  patch_strategy: dynamic
  dynamic_patch_step_size: 1
```

`fixed` is the default and preserves the previous behavior exactly. In
`dynamic` mode, denoising step zero uses the ordinary patch grid. At step `k`,
the interior lattice offset on each axis is:

```text
(k * dynamic_patch_step_size) mod patch_stride
```

Patch starts at zero and at the final valid canvas start remain anchored.
Interior starts are regenerated between those edge patches, so all latent
cells stay covered even when the shifted lattice changes the number of patches
for a step. If the image is one patch high, the vertical layout cannot move and
only the horizontal starts change.

Every step recomputes the RGB-scaled layout, directional prompt assignment,
and exclusive owner map. Run metadata records `num_planar_patches_by_step`,
`planar_patch_positions_by_step`, and
`planar_patch_prompt_indices_by_step`. Final VAE decoding uses the last
denoising step's layout.
