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
    for prompt, pitch_degrees in zip(five_prompts, (-90, -10, 0, 10, 90)):
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
