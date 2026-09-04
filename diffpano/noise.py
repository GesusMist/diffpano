"""Fixed per-camera Gaussian noise for clean-ERP consensus generation."""

import hashlib
from typing import Any, Dict, List, Sequence

import torch

from diffpano.config import CleanConsensusConfig
from diffpano.pipelines.clean_prediction import CleanPredictionBackend


def _noise_identity(value: torch.Tensor) -> str:
    sample = value.detach().float().cpu().contiguous().flatten()[:1024]
    return hashlib.sha256(sample.numpy().tobytes()).hexdigest()[:16]


class FixedPatchNoiseBank:
    """Own exactly one backend-native Gaussian realization per camera slot."""

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
