"""Perspective rendering for standalone-reference FID, without changing old metrics.

World coordinates and pixel centres match ``panorama_metrics.render``. The
angular frustum is independent of raster resolution; latitude is clamped and
longitude interpolates periodically across the ERP seam.
"""
import functools
import math

import cv2
import numpy as np


PROTOCOL = {
    "version": "standalone-fid-pixel-center-periodic-v1",
    "world_axes": "x right, y north, z yaw0; yaw atan2(x,z)",
    "roll": 0,
    "fov_x": 80.0,
    "fov_y": 80.0,
    "view_width": 1024,
    "view_height": 1024,
    "resampler": "OpenCV remap INTER_LINEAR, longitude wrap padding, latitude clamp",
    "pixel_centers": "ray x=(2*(col+0.5)/W-1)*tan(fov_x/2); "
    "y=(1-2*(row+0.5)/H)*tan(fov_y/2); "
    "ERP u=(lon/(2pi)+0.5)*W-0.5",
}


def _rotation(yaw, pitch):
    a, b = math.radians(yaw), math.radians(pitch)
    forward = np.array([math.cos(b) * math.sin(a), math.sin(b),
                        math.cos(b) * math.cos(a)])
    right = np.array([math.cos(a), 0.0, -math.sin(a)])
    up = np.cross(forward, right)
    return np.stack([right, up, forward], axis=1)


def _rays(yaw, pitch, size, fov):
    # Exactly 1.0 at 90 degrees also preserves the old renderer bit for bit.
    extent = 1.0 if fov == 90.0 else math.tan(math.radians(fov) / 2)
    x = ((np.arange(size) + 0.5) * 2 / size - 1) * extent
    y = (1 - (np.arange(size) + 0.5) * 2 / size) * extent
    xx, yy = np.meshgrid(x, y)
    vectors = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    vectors = vectors @ _rotation(yaw, pitch).T
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


# Reuse all 20 fixed FID views across each panorama (about 160 MiB of maps).
@functools.lru_cache(maxsize=20)
def _maps(height, width, yaw, pitch, size, fov):
    vectors = _rays(yaw, pitch, size, fov)
    longitude = np.arctan2(vectors[..., 0], vectors[..., 2])
    latitude = np.arcsin(np.clip(vectors[..., 1], -1.0, 1.0))
    x = ((longitude / (2 * np.pi) + 0.5) * width - 0.5 + 1).astype(np.float32)
    y = np.clip((0.5 - latitude / np.pi) * height - 0.5,
                0, height - 1).astype(np.float32)
    x.flags.writeable = y.flags.writeable = False
    return x, y


def render(rgb, yaw, pitch, size=1024, fov=80.0):
    """Return one square RGB uint8 view, with yaw/pitch/FOV in degrees."""
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 NumPy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("Expected a nonempty H x W x 3 RGB array")
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("View size must be a positive integer")
    yaw, pitch, fov = float(yaw), float(pitch), float(fov)
    if not all(math.isfinite(value) for value in (yaw, pitch, fov)):
        raise ValueError("Camera angles must be finite")
    if not -90 <= pitch <= 90 or not 0 < fov < 180:
        raise ValueError("Pitch must be in [-90, 90] and FOV in (0, 180)")
    padded = np.concatenate([rgb[:, -1:], rgb, rgb[:, :1]], axis=1)
    coordinates = _maps(*rgb.shape[:2], yaw, pitch, int(size), fov)
    return cv2.remap(padded, *coordinates, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
