"""VAE conversion helpers and the identity-preserving residual bridge."""

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, List, Optional

import torch

from experiments.legacy_spherical.diffpano_legacy.config import FUSION_DTYPE, PixelFusionConfig

TensorAdapter = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class VaeResidualBridgeResult:
    target_clean_latents: torch.Tensor
    original_roundtrip_vae_latents: torch.Tensor
    fused_roundtrip_vae_latents: torch.Tensor
    fusion_delta_vae_latents: torch.Tensor
    vae_roundtrip_error_norm: torch.Tensor
    fusion_delta_norm: torch.Tensor

def _chunk_tensor(tensor: torch.Tensor, chunk_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, tensor.shape[0], chunk_size):
        yield tensor[start:start + chunk_size]


def _vae_scaling_factor(vae: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    value = getattr(getattr(vae, "config", None), "scaling_factor", 1.0)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _vae_shift_factor(vae: Any, *, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    value = getattr(getattr(vae, "config", None), "shift_factor", None)
    if value is None:
        return None
    return torch.as_tensor(value, device=device, dtype=dtype)


def _decode_latent_input(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    scaling_factor = _vae_scaling_factor(vae, device=latents.device, dtype=latents.dtype)
    shift_factor = _vae_shift_factor(vae, device=latents.device, dtype=latents.dtype)
    latents = latents / scaling_factor
    if shift_factor is not None:
        latents = latents + shift_factor
    return latents


def _encode_latent_output(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    scaling_factor = _vae_scaling_factor(vae, device=latents.device, dtype=latents.dtype)
    shift_factor = _vae_shift_factor(vae, device=latents.device, dtype=latents.dtype)
    if shift_factor is not None:
        latents = latents - shift_factor
    return latents * scaling_factor


def _extract_encoded_latents(encoded: Any, *, generator: Optional[torch.Generator], sample_posterior: bool) -> torch.Tensor:
    if hasattr(encoded, "latent_dist"):
        if sample_posterior:
            return encoded.latent_dist.sample(generator=generator)
        if hasattr(encoded.latent_dist, "mean"):
            return encoded.latent_dist.mean
        if hasattr(encoded.latent_dist, "mode"):
            return encoded.latent_dist.mode()
    if hasattr(encoded, "latents"):
        return encoded.latents
    if hasattr(encoded, "latent"):
        return encoded.latent
    if isinstance(encoded, (tuple, list)):
        return encoded[0]
    return encoded


def decode_view_latents(
    vae: Any,
    view_latents: torch.Tensor,
    config: PixelFusionConfig,
) -> torch.Tensor:
    """Decode VAE latents [views, latent_channels, h, w] into RGB [-1, 1] [views, 3, H, W]."""

    original_dtype = view_latents.dtype
    vae_dtype = getattr(vae, "dtype", original_dtype)
    decoded: List[torch.Tensor] = []
    with torch.inference_mode():
        for chunk in _chunk_tensor(view_latents, config.vae_chunk_size):
            model_chunk = _decode_latent_input(vae, chunk.to(dtype=vae_dtype))
            decoded_chunk = vae.decode(model_chunk, return_dict=False)[0]
            decoded.append(decoded_chunk.to(dtype=original_dtype))
    return torch.cat(decoded, dim=0)


def encode_view_images(
    vae: Any,
    view_images: torch.Tensor,
    config: PixelFusionConfig,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Encode RGB [-1, 1] [views, 3, H, W] into scaled VAE latents [views, latent_channels, h, w]."""

    original_dtype = view_images.dtype
    vae_dtype = getattr(vae, "dtype", original_dtype)
    encoded: List[torch.Tensor] = []
    with torch.inference_mode():
        for chunk in _chunk_tensor(view_images, config.vae_chunk_size):
            encoded_chunk = vae.encode(chunk.to(dtype=vae_dtype))
            latents = _extract_encoded_latents(
                encoded_chunk,
                generator=generator,
                sample_posterior=config.vae_sample_posterior,
            )
            encoded.append(_encode_latent_output(vae, latents).to(dtype=original_dtype))
    return torch.cat(encoded, dim=0)

def _adapt_latents(latents: torch.Tensor, adapter: Optional[TensorAdapter]) -> torch.Tensor:
    return adapter(latents) if adapter is not None else latents


def build_identity_preserving_vae_target(
    vae: Any,
    clean_latents: torch.Tensor,
    original_view_images: torch.Tensor,
    fused_view_images: torch.Tensor,
    config: PixelFusionConfig,
    *,
    latent_to_vae_latents: Optional[TensorAdapter] = None,
    vae_latents_to_latent: Optional[TensorAdapter] = None,
) -> VaeResidualBridgeResult:
    """Carry only the RGB fusion residual across the non-invertible VAE round trip."""

    if original_view_images.shape != fused_view_images.shape:
        raise ValueError("Original and fused RGB views must have the same shape")
    deterministic_config = replace(config, vae_sample_posterior=False)
    original_roundtrip = encode_view_images(vae, original_view_images, deterministic_config).to(
        dtype=FUSION_DTYPE
    )
    fused_roundtrip = encode_view_images(vae, fused_view_images, deterministic_config).to(
        dtype=FUSION_DTYPE
    )
    vae_clean_latents = _adapt_latents(clean_latents, latent_to_vae_latents).to(dtype=FUSION_DTYPE)
    if not (
        original_roundtrip.shape
        == fused_roundtrip.shape
        == vae_clean_latents.shape
    ):
        raise ValueError(
            "VAE residual bridge shapes disagree: "
            f"clean={tuple(vae_clean_latents.shape)}, "
            f"original_roundtrip={tuple(original_roundtrip.shape)}, "
            f"fused_roundtrip={tuple(fused_roundtrip.shape)}"
        )

    fusion_delta_vae = fused_roundtrip - original_roundtrip
    target_vae_clean = vae_clean_latents + fusion_delta_vae
    target_clean = _adapt_latents(target_vae_clean, vae_latents_to_latent).to(dtype=FUSION_DTYPE)
    if target_clean.shape != clean_latents.shape:
        raise ValueError(
            f"Adapted VAE target shape {tuple(target_clean.shape)} does not match "
            f"scheduler clean shape {tuple(clean_latents.shape)}"
        )
    return VaeResidualBridgeResult(
        target_clean_latents=target_clean,
        original_roundtrip_vae_latents=original_roundtrip,
        fused_roundtrip_vae_latents=fused_roundtrip,
        fusion_delta_vae_latents=fusion_delta_vae,
        vae_roundtrip_error_norm=(original_roundtrip - vae_clean_latents).norm(),
        fusion_delta_norm=fusion_delta_vae.norm(),
    )
