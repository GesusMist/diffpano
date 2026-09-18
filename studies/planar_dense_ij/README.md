# Planar I/J: 512×512 RGB patches, stride 32

Eight runs: I/J for SD2, SANA, FLUX and SD3.5. This additive follow-up changes only patch size and stride from the validated 256/64 study. It uses the unchanged I/J pipeline, historical prompt and global initial native tensors, seeds, canvases, model settings, precision, guidance and step counts. Existing jobs and outputs remain intact.

|Backend|RGB canvas H×W|Native patch/stride|Patches per step|Calls per run|
|---|---|---|---|---|
|SD2|512×1024|64/4|17|510|
|SANA|1024×2048|16/1|833|16660|
|FLUX|1024×2048|64/4|833|16660|
|SD3.5|1024×2048|64/4|833|33320|

Adjacent patches overlap by 93.75% along each sampled axis. All RGB origins map exactly to integer native coordinates. SD2 has one vertical row because its original canvas is 512 pixels high; the larger patch reduces its patch count despite the smaller stride.

I uses synchronized native VAE residual correction; J disables residual correction and own-RGB roundtrip diagnostics. Both retain current local noisy states, Jacobi updates, and uniform RGB averaging. This is a planar experiment.

Schedules use the existing backend preparation at the actual 512×512 local resolution. FLUX's resolution-derived dynamic shift changes accordingly and must match within its I/J pair. Other timesteps and sigmas must match historical values. GPU jobs enforce A40 and the original software versions, check real VAE shapes, then run the complete schedules.

Validation verifies the unchanged previously tested I/J implementation by source hashes, checks that only patch/stride differ from the 256/64 study, and checks all eight layouts, coverage, initial tensors and conditioning. Outputs and automatic paired report: `outputs/planar-dense-ij/20260918-p512-s32/`.
