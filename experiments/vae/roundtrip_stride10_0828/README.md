# VAE patch roundtrip study (2026-08-28)

This matrix contains 15 independent jobs: SANA, FLUX, and SD2 crossed with
1, 2, 5, 10, and 20 cumulative VAE encode/decode roundtrips.

- Input: `experiments/vae/image.png` (`4096x640`, converted from RGBA to RGB)
- Patch size: `640x640` pixels (one patch high)
- Patch stride: 10 raw VAE latent points
- Patch write-back: direct replacement, with no fusion or averaging
- Encoder posterior: deterministic mean/mode (no sampling)
- Repeated input: the previous decoded FP32 tensor, clamped to `[-1, 1]`
- Precision: SANA BF16, FLUX BF16, SD2 FP16, matching the current pipelines
- Output: `test_outputs/vae_roundtrip_stride10_0828/`

Because the VAE ratios differ, stride 10 corresponds to 320 pixels for SANA
(32x compression) and 80 pixels for FLUX/SD2 (8x compression). FLUX stride is
measured in raw VAE latents; transformer token packing is not involved.

Decoded patches are written in the layout's deterministic row-major order. If
patches overlap, the later patch directly overwrites the earlier patch in that
overlap. For this one-patch-high image, that means left-to-right overwrite.

Submit all 15 independent jobs with:

```bash
bash experiments/vae/roundtrip_stride10_0828/submit_all.sh
```

Every run saves `input.png`, `result.png`, `metadata.json`, and `run.log`.
`metadata.json` includes per-round MSE, MAE, and PSNR against both the original
input and the preceding reconstruction.
