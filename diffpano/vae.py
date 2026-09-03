"""Deterministic backend-neutral VAE encode/decode helpers."""

from typing import Any, Iterable, List, Optional

import torch


def _chunks(tensor: torch.Tensor, size: int) -> Iterable[torch.Tensor]:
    for start in range(0, tensor.shape[0], size):
        yield tensor[start:start + size]


def _scaling(vae: Any, tensor: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(
        getattr(getattr(vae, "config", None), "scaling_factor", 1.0),
        device=tensor.device,
        dtype=tensor.dtype,
    )


def _shift(vae: Any, tensor: torch.Tensor) -> Optional[torch.Tensor]:
    value = getattr(getattr(vae, "config", None), "shift_factor", None)
    if value is None:
        return None
    return torch.as_tensor(value, device=tensor.device, dtype=tensor.dtype)


def _posterior_mode(encoded: Any) -> torch.Tensor:
    if hasattr(encoded, "latent_dist"):
        distribution = encoded.latent_dist
        if hasattr(distribution, "mode"):
            mode = distribution.mode
            return mode() if callable(mode) else mode
        if hasattr(distribution, "mean"):
            return distribution.mean
    if hasattr(encoded, "latents"):
        return encoded.latents
    if hasattr(encoded, "latent"):
        return encoded.latent
    if isinstance(encoded, (tuple, list)):
        return encoded[0]
    return encoded


def encode_view_images(vae: Any, rgb: torch.Tensor, *, chunk_size: int = 1) -> torch.Tensor:
    """Encode ``[-1,1]`` RGB using posterior mode/mean, never posterior sampling."""

    source_dtype = rgb.dtype
    model_dtype = getattr(vae, "dtype", source_dtype)
    outputs: List[torch.Tensor] = []
    with torch.inference_mode():
        for chunk in _chunks(rgb, chunk_size):
            latent = _posterior_mode(vae.encode(chunk.to(dtype=model_dtype)))
            shift = _shift(vae, latent)
            if shift is not None:
                latent = latent - shift
            outputs.append((latent * _scaling(vae, latent)).to(dtype=source_dtype))
    return torch.cat(outputs, dim=0)


def decode_view_latents(vae: Any, latents: torch.Tensor, *, chunk_size: int = 1) -> torch.Tensor:
    source_dtype = latents.dtype
    model_dtype = getattr(vae, "dtype", source_dtype)
    outputs: List[torch.Tensor] = []
    with torch.inference_mode():
        for chunk in _chunks(latents, chunk_size):
            value = chunk / _scaling(vae, chunk)
            shift = _shift(vae, value)
            if shift is not None:
                value = value + shift
            decoded = vae.decode(value.to(dtype=model_dtype), return_dict=False)[0]
            outputs.append(decoded.to(dtype=source_dtype))
    return torch.cat(outputs, dim=0)
