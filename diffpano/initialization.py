"""Modular ERP-RGB initialization and prompt/seed helpers."""

import random
from pathlib import Path
from typing import List, Optional

import torch

from diffpano.camera import CameraSampler
from diffpano.config import InitializationConfig
from diffpano.fusion import RGBFusionAccumulator
from diffpano.warp import WarpOperator


def set_random_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_directional_prompts(path: str) -> List[str]:
    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) == 1:
        prompts *= 5
    if len(prompts) != 5:
        raise ValueError("Directional prompt files must contain one or five non-empty lines")
    return prompts


def _erp_rgb_noise(
    config: InitializationConfig,
    *,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if config.distribution != "gaussian":
        raise ValueError(f"Unsupported initialization distribution {config.distribution!r}")
    erp = torch.randn(
        batch_size, 3, height, width, device=device, dtype=torch.float32, generator=generator
    )
    erp = erp * config.std + config.mean
    if config.clamp:
        erp = erp.clamp(config.clamp_min, config.clamp_max)
    return erp


def initialize_erp_canvas(
    config: InitializationConfig,
    *,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    generator: torch.Generator,
    camera_sampler: CameraSampler,
    warp_operator: WarpOperator,
    fusion_config,
    view_denoiser=None,
) -> torch.Tensor:
    """Initialize the sole persistent state and return ``[B,3,H_erp,W_erp]``."""

    if config.mode == "pixel_gaussian":
        if config.clamp:
            raise ValueError("Native pixel diffusion noise must not be clamped")
        if config.distribution != "gaussian":
            raise ValueError(f"Unsupported initialization distribution {config.distribution!r}")
        return (
            torch.randn(
                batch_size, 3, height, width,
                device=device, dtype=torch.float32, generator=generator,
            )
            * config.std
            + config.mean
        )

    if config.mode == "erp_rgb_noise":
        return _erp_rgb_noise(
            config,
            batch_size=batch_size,
            height=height,
            width=width,
            device=device,
            generator=generator,
        )
    if config.mode != "latent_native_bootstrap":
        raise ValueError(f"Unsupported initialization mode {config.mode!r}")
    if view_denoiser is None or not hasattr(view_denoiser, "sample_native_rgb"):
        raise ValueError("latent_native_bootstrap requires a denoiser with native latent sampling")
    previous = torch.zeros(batch_size, 3, height, width, device=device, dtype=torch.float32)
    accumulator = RGBFusionAccumulator(previous, fusion_config)
    for camera in camera_sampler.sample(step_index=0, num_steps=1):
        rgb = view_denoiser.sample_native_rgb(
            batch_size=batch_size,
            height=camera.height,
            width=camera.width,
            generator=generator,
        )
        accumulator.accumulate(warp_operator.perspective_to_erp(rgb, camera, (height, width)))
    return accumulator.finalize().erp_rgb
