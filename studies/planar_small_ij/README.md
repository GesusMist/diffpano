# I/J smaller planar patches

User-approved follow-up: 256×256 **RGB** patches, stride 64 RGB pixels, four original latent backends × I/J. Reuses the existing pipeline without changes. I fuses VAE residuals on the aligned native canvas; J disables residual correction and own-RGB roundtrip diagnostics. Both preserve current noisy states and use synchronous uniform RGB averaging.

Only native patch size and stride change in the historical resolved configs. SD2/FLUX/SD3.5 use native patch 32, stride 8; SANA uses native patch 8, stride 2. Canvas, seed, prompt, checkpoint, precision, guidance and step counts stay as in I/J. SD2 has 65 patches on its 512×1024 RGB canvas; other backends have 377 on 1024×2048. All use exact integer crops of the saved global native initial tensors. This is a planar I/J experiment, not a perspective/ERP experiment.

The existing backend prepares its schedule at the actual 256×256 local resolution. FLUX's dynamic shift therefore changes relative to its historical 1024 patch, but matches within the new I/J pair. Other prepared timesteps and sigmas must match their historic values. Real VAE shape/finite checks precede generation. A40 GPU and original software versions are enforced.

Outputs: `outputs/planar-small-ij/20260918-p256-s64/{backend}/{I,J}/generation/result.png` with metadata, diagnostics and completion markers. Previous outputs are preserved. CPU validation checks historic configs, geometry, coverage, source hashes, initial files and the existing I/J algorithm tests. GPU jobs require its successful validation record.
