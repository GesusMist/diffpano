"""Backend-neutral directional prompt assignment."""

import math
from dataclasses import dataclass
from typing import List, Sequence

import torch

from diffpano.camera import PerspectiveCamera


@dataclass(frozen=True)
class DirectionalPromptSet:
    prompts: List[str]
    directions: torch.Tensor


def expand_directional_prompts(five_prompts: Sequence[str]) -> DirectionalPromptSet:
    if len(five_prompts) != 5:
        raise ValueError("directional prompting requires five prompt bands")
    expanded: List[str] = []
    directions = []
    # Prompt files are ordered north, upper-equatorial, equator,
    # lower-equatorial, south. Positive pitch points north in camera geometry.
    for prompt, pitch_degrees in zip(five_prompts, (90, 10, 0, -10, -90)):
        pitch = math.radians(pitch_degrees)
        for yaw_degrees in (0, 90, 180, 270):
            yaw = math.radians(yaw_degrees)
            expanded.append(prompt)
            directions.append(
                [math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)]
            )
    return DirectionalPromptSet(expanded, torch.tensor(directions, dtype=torch.float32))


def camera_prompt_indices(cameras: Sequence[PerspectiveCamera], prompt_directions: torch.Tensor) -> torch.Tensor:
    if not cameras:
        return torch.empty(0, dtype=torch.long)
    camera_directions = torch.stack([camera.forward() for camera in cameras])
    directions = prompt_directions.to(dtype=torch.float32, device=camera_directions.device)
    return (camera_directions @ directions.t()).argmax(dim=1)


def expanded_prompt_indices(
    prompt_indices: Sequence[int],
    *,
    batch_size: int,
    num_prompts: int,
    device: torch.device,
) -> torch.Tensor:
    """Repeat spatial prompt slots for the image batch on a backend device."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    indices = torch.as_tensor(prompt_indices, dtype=torch.long)
    if indices.ndim != 1:
        raise ValueError("prompt_indices must be one-dimensional")
    if indices.numel() and (
        int(indices.min()) < 0 or int(indices.max()) >= num_prompts
    ):
        raise IndexError("prompt index is outside the prepared conditioning bank")
    return indices.repeat_interleave(batch_size).to(device=device)
