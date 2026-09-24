# GWTFlow-style spherical shared noise: fixed method before image generation

Source: Burgert et al., *Go-with-the-Flow*, arXiv:2501.08331v4 / CVPR 2025, Algorithm 1 and Proposition 1. The official repository redirects to [Eyeline-Labs/Go-with-the-Flow](https://github.com/Eyeline-Labs/Go-with-the-Flow); its `make_warped_noise.py` imports Ryan Burgert's `CommonSource/noise_warp.py`. The inspected snapshots and commit records are retained in `references/`.

We implement the **published graph algorithm**, not an exact reproduction of the released video pipeline. The released `warp_xyωc` uses a forward particle scatter, backward gap filling, `regaussianize` with centered innovations, and density-weighted contraction. Its normalized per-edge Gaussian convention and density stabilization differ from the explicit Algorithm 1 equations. It also uses noise-value grouping, approximate inverse flow, optional offset propagation/degradation, and high-resolution-to-latent downsampling. We instead follow the published equations below exactly; use explicit geometry edges, exact inverse camera projection, and the actual raw native raster. We do not copy upstream executable code, estimate optical flow, resize noise from RGB, apply empirical whitening, add post-warp noise, or implement integrated white noise.

For each view separately, source vertices are the historical primary ERP cells and target vertices are native perspective pixels. Forward edges project ERP **cell centers** into visible camera pixels using existing FP32 world rays and round-to-even nearest rasterization. For targets with no incoming forward edge, add the historical nearest backward ERP source edge. Longitude and reflected-pole semantics are inherited from `NearestERPIndexProjector`. No geometry/FOV/cover change occurs.

Let q_s be the shared iid source Gaussian; p_s=1 is the initial source density. For source out-degree d_s and independent edge innovations Z_se, use

```
X_se = q_s/d_s + (Z_se - mean_{e out of s} Z_se)/sqrt(d_s)
alpha_se = p_s/d_s
v_t = sum_{e into t} alpha_se^2/d_s
q'_t = sum_{e into t} alpha_se X_se / sqrt(v_t)
p'_t = sum_{e into t} alpha_se
```

Unconnected targets receive fresh standard Gaussian samples (not expected for full-sphere ERP backward coverage). Conditional innovations have zero sum per source. Unconditionally Var(X_se)=1/d_s and Cov(X_se,X_sf)=0 for distinct edges. Different sources are independent. Each edge enters only one target, so target Gaussians are independent and division by sqrt(v_t) gives unit variance. All normalization is determined by graph density/degrees, not realized noise values. The unit ERP density matches the existing iid ERP source measure; it is not a claim of equal-solid-angle white noise. Output density is reported then discarded because initialization is one hop.

Every view transports the **same** q_s, but has independent innovations determined by stable camera slot, not execution order. Thus within a view the conditional cancellation preserves iid noise, while across two views covariance comes from shared source terms. If w_vs,t = alpha_vs,t/(d_vs sqrt(v_vt)), cross-view covariance is sum_s w_vs,t w_ws,u, nonnegative, and generally smaller than copied nearest noise. Distinct views are not jointly iid; that correlation is intentional. No temporal chaining or noise reinjection occurs. The backend's native initial scaling is applied exactly once after transport. Only local states persist.

CPU FP32 and canonical source/target edge ordering are fixed. Channel chunks bound temporary random buffers. Source seed 0 produces the same full ERP Gaussian as V-shared-erp. Innovations use dedicated SHA256-derived CPU seeds by camera identity; diagnostics use separate seeds. Scientific runs pass explicit canonical slot IDs 0–88. An execution-order parameter changes processing order only.

The statistical preflight uses full native geometry. Sparse Monte Carlo keeps **all sibling edges** for every sampled source, so it samples an exact marginal of the full graph, not an approximate reduced-degree graph. Gate thresholds are calibrated to independent realizations and matched iid controls, before viewing generated panoramas. Tests compare against an independent scalar transcription of Algorithm 1.
