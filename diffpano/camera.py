"""Perspective camera representation and SphereDiff-derived camera covers."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

from diffpano.config import RotationConfig, SamplingConfig, ViewConfig
from diffpano.geometry import camera_rotation, euler_rotation, rotation_to_yaw_pitch_roll


@dataclass(frozen=True)
class PerspectiveCamera:
    """A pinhole camera. Angles are radians and FOV values are degrees."""

    yaw: float
    pitch: float
    roll: float
    fov_x: float
    fov_y: float
    height: int
    width: int

    def rotation(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return camera_rotation(self.yaw, self.pitch, self.roll, device=device)

    def forward(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return self.rotation(device)[:, 2]

    def cache_key(self) -> Tuple[float, ...]:
        return (
            round(self.yaw, 7), round(self.pitch, 7), round(self.roll, 7),
            round(self.fov_x, 5), round(self.fov_y, 5), self.height, self.width,
        )


class CameraSampler(ABC):
    @abstractmethod
    def sample(self, step_index: int, num_steps: int) -> List[PerspectiveCamera]:
        """Return a complete overlapping camera cover for one global step."""


def spherediff_camera_cover(view: ViewConfig) -> List[PerspectiveCamera]:
    """Reproduce SphereDiff's dense-equator 89-view, 80-degree cover."""

    overlap_x = view.fov_x * 0.6
    overlap_y = view.fov_y * 0.6
    num_latitudes = math.ceil((90 + view.fov_y / 2) / (view.fov_y - overlap_y))
    positive = torch.linspace(0, 90, num_latitudes, dtype=torch.float32).tolist()
    latitudes = positive + [-value for value in positive[1:]]
    latitudes.sort(key=lambda value: abs(value + 1.0e-2))
    cameras: List[PerspectiveCamera] = []
    for latitude_deg in latitudes:
        circumference = math.cos(math.radians(latitude_deg)) * 360
        around = math.ceil(circumference / (view.fov_x - overlap_x)) + 3
        yaws = torch.linspace(-math.pi, math.pi, around + 1, dtype=torch.float32)[:-1]
        cameras.extend(
            PerspectiveCamera(
                yaw=float(yaw),
                pitch=math.radians(latitude_deg),
                roll=0.0,
                fov_x=view.fov_x,
                fov_y=view.fov_y,
                height=view.height,
                width=view.width,
            )
            for yaw in yaws
        )
    return cameras


class SphereDiffFixedCameraSampler(CameraSampler):
    def __init__(self, view: ViewConfig):
        self._cameras = spherediff_camera_cover(view)

    def sample(self, step_index: int, num_steps: int) -> List[PerspectiveCamera]:
        del step_index, num_steps
        return list(self._cameras)


class SphereDiffRotatedCameraSampler(CameraSampler):
    """Apply one deterministic global SO(3) rotation to the full cover per step."""

    def __init__(self, view: ViewConfig, rotation: RotationConfig, seed: int):
        self._base = spherediff_camera_cover(view)
        self._rotation = rotation
        self._seed = seed
        self._cache = {}

    def _angles(self, num_steps: int) -> Sequence[Tuple[float, float, float]]:
        if num_steps not in self._cache:
            generator = torch.Generator(device="cpu").manual_seed(self._seed)
            unit = torch.rand((num_steps, 3), generator=generator, dtype=torch.float32) * 2 - 1
            maxima = torch.tensor(
                [self._rotation.max_yaw_deg, self._rotation.max_pitch_deg, self._rotation.max_roll_deg],
                dtype=torch.float32,
            )
            radians = torch.deg2rad(unit * maxima)
            self._cache[num_steps] = [tuple(float(value) for value in row) for row in radians]
        return self._cache[num_steps]

    def sample(self, step_index: int, num_steps: int) -> List[PerspectiveCamera]:
        if not 0 <= step_index < num_steps:
            raise IndexError("step_index is outside the denoising schedule")
        global_rotation = euler_rotation(*self._angles(num_steps)[step_index], device=torch.device("cpu"))
        cameras = []
        for camera in self._base:
            rotated = global_rotation @ camera.rotation()
            yaw, pitch, roll = rotation_to_yaw_pitch_roll(rotated)
            cameras.append(
                PerspectiveCamera(
                    yaw, pitch, roll, camera.fov_x, camera.fov_y, camera.height, camera.width
                )
            )
        return cameras


def build_camera_sampler(config: SamplingConfig, view: ViewConfig, seed: int) -> CameraSampler:
    if config.strategy == "spherediff_fixed":
        return SphereDiffFixedCameraSampler(view)
    if config.strategy == "spherediff_rotated":
        return SphereDiffRotatedCameraSampler(view, config.rotation, seed)
    raise ValueError(f"Unsupported camera sampler {config.strategy!r}")


def camera_for_direction(
    yaw_degrees: float,
    pitch_degrees: float,
    *,
    height: int,
    width: int,
    fov_x: float = 80.0,
    fov_y: float = 80.0,
    roll_degrees: float = 0.0,
) -> PerspectiveCamera:
    """Convenience constructor primarily useful for diagnostics and tests."""

    return PerspectiveCamera(
        math.radians(yaw_degrees), math.radians(pitch_degrees), math.radians(roll_degrees),
        fov_x, fov_y, height, width,
    )
