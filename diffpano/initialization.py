"""Common deterministic initialization and prompt-loading helpers."""

from pathlib import Path
from typing import List, Optional, Union

import torch

from diffpano.config import PixelFusionConfig


def apply_configured_random_seed(
    generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    config: PixelFusionConfig,
    *,
    device: torch.device,
) -> Optional[Union[torch.Generator, List[torch.Generator]]]:
    """Create the generation generator when the experiment config specifies a seed."""

    if config.random_seed is None:
        return generator
    return torch.Generator(device=device).manual_seed(config.random_seed)


def set_random_seed(seed: Optional[int]) -> None:
    """Seed common host and Torch RNGs when a seed is configured."""

    if seed is None:
        return
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_directional_prompts(path: str) -> List[str]:
    """Load five directional prompts, repeating a single global prompt when supplied."""

    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) == 1:
        prompts = prompts * 5
    if len(prompts) != 5:
        raise ValueError(f"Directional prompt file must contain 1 or 5 non-empty lines: {prompt_path}")
    return prompts
