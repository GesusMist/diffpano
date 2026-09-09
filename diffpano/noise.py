"""Fixed per-spatial-slot Gaussian noise for predicted-clean consensus."""

import hashlib
from typing import Any, Dict, List, Sequence

import torch

from diffpano.config import CleanConsensusConfig
from diffpano.pipelines.clean_prediction import CleanPredictionBackend


def _noise_identity(value: torch.Tensor) -> str:
    sample = value.detach().float().cpu().contiguous().flatten()[:1024]
    return hashlib.sha256(sample.numpy().tobytes()).hexdigest()[:16]


class FixedPatchNoiseBank:
    """Own one backend-native Gaussian per camera or planar-patch slot.

    ``num_cameras`` is retained as a backward-compatible constructor name; the
    planar caller passes its stable patch count and retrieves by patch index.
    """

    def __init__(
        self,
        config: CleanConsensusConfig,
        *,
        backend: CleanPredictionBackend,
        num_cameras: int,
        batch_size: int,
        height: int,
        width: int,
        seed: int,
    ):
        if num_cameras < 1:
            raise ValueError("FixedPatchNoiseBank requires at least one camera")
        self.config = config
        self.backend = backend
        self.num_cameras = num_cameras
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.seed = int(seed)
        self._values: List[torch.Tensor] = []
        self._seeds: List[int] = []
        self.identities: Dict[int, str] = {}

        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        for camera_index in range(num_cameras):
            if config.noise_storage == "seed":
                camera_seed = self.seed + 1_000_003 * camera_index
                self._seeds.append(camera_seed)
                value = self._sample_seed(camera_seed)
            else:
                value = backend.sample_fixed_noise(
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    generator=generator,
                ).detach().float()
                target = (
                    backend.device
                    if config.noise_storage == "gpu"
                    else torch.device("cpu")
                )
                value = value.to(device=target)
                self._values.append(value)
            self.identities[camera_index] = _noise_identity(value)

    def _sample_seed(self, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return self.backend.sample_fixed_noise(
            batch_size=self.batch_size,
            height=self.height,
            width=self.width,
            generator=generator,
        ).detach().float().cpu()

    def get(
        self, camera_indices: Sequence[int], *, device: torch.device
    ) -> torch.Tensor:
        values = []
        for camera_index in camera_indices:
            if not 0 <= camera_index < self.num_cameras:
                raise IndexError(f"Unknown camera noise slot {camera_index}")
            if self.config.noise_storage == "seed":
                value = self._sample_seed(self._seeds[camera_index])
            else:
                value = self._values[camera_index]
            values.append(value.to(device=device, dtype=torch.float32))
        return torch.cat(values, dim=0)

    def identity(self, camera_index: int) -> str:
        return self.identities[camera_index]


class GlobalNativeNoiseBank:
    """One raw native Gaussian field, retrieved by exact aligned patch crops.

    CPU and GPU modes retain the field. Seed mode regenerates the same whole
    CPU field on retrieval (trades memory for work, never resamples per patch).
    Overlapping clean VAE encodings can still differ due to boundary context.
    """

    def __init__(self, config, *, backend, layout, stride, batch_size, seed):
        from diffpano.planar import PlanarPatch
        self.config = config
        self.backend = backend
        self.seed = int(seed)
        factor = backend.native_spatial_factor
        self.native_height, self.native_width = backend.native_spatial_shape_for_rgb(
            layout.canvas_height, layout.canvas_width)
        ph, pw = backend.native_spatial_shape_for_rgb(layout.patch_size, layout.patch_size)
        if ph != pw or stride % factor:
            raise ValueError(f"RGB stride {stride} must be exactly aligned to native spatial factor {factor}; no rounding is allowed")
        self.patches = {}
        for patch in layout.patches:
            if patch.x % factor or patch.y % factor:
                raise ValueError(f"RGB patch origin {(patch.y, patch.x)} is not aligned to native spatial factor {factor}")
            self.patches[patch.index] = PlanarPatch(patch.index, patch.y // factor, patch.x // factor, ph)
        self.shape = (batch_size, backend.native_channels, self.native_height, self.native_width)
        value = self._sample()
        if config.noise_storage not in {'cpu', 'gpu', 'seed'}:
            raise ValueError('Unknown global noise storage mode')
        target = backend.device if config.noise_storage == 'gpu' else torch.device('cpu')
        self._value = None if config.noise_storage == 'seed' else value.to(target)
        from diffpano.planar import extract_planar_patch
        self.identities = {i: _noise_identity(extract_planar_patch(value, patch)) for i, patch in self.patches.items()}
        self.global_identity = _noise_identity(value)

    def _sample(self):
        generator = torch.Generator(device='cpu').manual_seed(self.seed)
        return torch.randn(*self.shape, generator=generator, dtype=torch.float32)

    def get(self, patch_indices, *, device):
        from diffpano.planar import extract_planar_patch
        field = self._sample() if self._value is None else self._value
        return torch.cat([extract_planar_patch(field, self.patches[i]).to(device) for i in patch_indices], dim=0)

    def identity(self, index):
        return self.identities[index]
