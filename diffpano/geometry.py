"""FP32 spherical coordinate and camera-rotation geometry.

The sphere in this module is geometry only.  No feature, latent, noise, or
diffusion tensor is attached to its directions.
"""

import math
from typing import Tuple

import torch


def camera_rotation(yaw: float, pitch: float, roll: float, *, device: torch.device) -> torch.Tensor:
    """Return world-from-camera rotation with columns (right, up, forward)."""

    dtype = torch.float32
    yaw_t = torch.tensor(yaw, device=device, dtype=dtype)
    pitch_t = torch.tensor(pitch, device=device, dtype=dtype)
    roll_t = torch.tensor(roll, device=device, dtype=dtype)
    forward = torch.stack(
        [
            torch.cos(pitch_t) * torch.sin(yaw_t),
            torch.sin(pitch_t),
            torch.cos(pitch_t) * torch.cos(yaw_t),
        ]
    )
    right_zero = torch.stack([torch.cos(yaw_t), torch.zeros_like(yaw_t), -torch.sin(yaw_t)])
    up_zero = torch.linalg.cross(forward, right_zero)
    right = torch.cos(roll_t) * right_zero + torch.sin(roll_t) * up_zero
    up = -torch.sin(roll_t) * right_zero + torch.cos(roll_t) * up_zero
    return torch.stack([right, up, forward], dim=1)


def rotation_to_yaw_pitch_roll(rotation: torch.Tensor) -> Tuple[float, float, float]:
    """Invert :func:`camera_rotation`, including the roll around the view ray."""

    rotation = rotation.to(dtype=torch.float32)
    forward = rotation[:, 2]
    pitch = torch.asin(forward[1].clamp(-1, 1))
    horizontal = torch.linalg.vector_norm(forward[[0, 2]])
    if float(horizontal) < 1.0e-7:
        # At a pole, preserve orientation using the projected camera right axis.
        right = rotation[:, 0]
        yaw = torch.atan2(-right[2], right[0])
    else:
        yaw = torch.atan2(forward[0], forward[2])
    right_zero = torch.stack([torch.cos(yaw), torch.zeros_like(yaw), -torch.sin(yaw)])
    up_zero = torch.linalg.cross(forward, right_zero)
    up_zero = up_zero / torch.linalg.vector_norm(up_zero).clamp_min(1.0e-12)
    roll = torch.atan2(torch.dot(rotation[:, 0], up_zero), torch.dot(rotation[:, 0], right_zero))
    return float(yaw), float(pitch), float(roll)


def euler_rotation(yaw: float, pitch: float, roll: float, *, device: torch.device) -> torch.Tensor:
    """Return a global XYZ rotation used to rotate an entire camera cover."""

    dtype = torch.float32
    y, p, r = [torch.tensor(value, device=device, dtype=dtype) for value in (yaw, pitch, roll)]
    one, zero = torch.ones_like(y), torch.zeros_like(y)
    ry = torch.stack(
        [torch.stack([torch.cos(y), zero, torch.sin(y)]),
         torch.stack([zero, one, zero]),
         torch.stack([-torch.sin(y), zero, torch.cos(y)])]
    )
    rx = torch.stack(
        [torch.stack([one, zero, zero]),
         torch.stack([zero, torch.cos(p), -torch.sin(p)]),
         torch.stack([zero, torch.sin(p), torch.cos(p)])]
    )
    rz = torch.stack(
        [torch.stack([torch.cos(r), -torch.sin(r), zero]),
         torch.stack([torch.sin(r), torch.cos(r), zero]),
         torch.stack([zero, zero, one])]
    )
    return rz @ rx @ ry


def erp_world_directions(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    """Return pixel-centred ERP unit rays as ``[H,W,3]`` in FP32."""

    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    latitude = (0.5 - (y + 0.5) / height) * math.pi
    longitude = ((x + 0.5) / width - 0.5) * (2 * math.pi)
    lat, lon = torch.meshgrid(latitude, longitude, indexing="ij")
    cos_lat = torch.cos(lat)
    return torch.stack(
        [torch.sin(lon) * cos_lat, torch.sin(lat), torch.cos(lon) * cos_lat], dim=-1
    )


def world_to_longitude_latitude(directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    directions = directions.to(dtype=torch.float32)
    longitude = torch.atan2(directions[..., 0], directions[..., 2])
    latitude = torch.asin(directions[..., 1].clamp(-1, 1))
    return longitude, latitude


def pairwise_angular_distances(directions: torch.Tensor) -> torch.Tensor:
    normalized = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    return torch.acos((normalized @ normalized.t()).clamp(-1, 1))
