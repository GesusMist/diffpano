"""Native diffusion operations, separate from the RGB view interface."""

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class NativeStateBackend(Protocol):
    @property
    def native_channels(self) -> int: ...

    @property
    def native_spatial_factor(self) -> int: ...

    def native_spatial_shape_for_rgb(self, height: int, width: int) -> tuple[int, int]: ...

    def rgb_spatial_shape_for_native(self, height: int, width: int) -> tuple[int, int]: ...

    def sample_initial_native_state(self, *, batch_size: int, native_height: int,
                                    native_width: int, generator: torch.Generator) -> torch.Tensor: ...

    def denoise_native_step(self, native_state: torch.Tensor, timestep: Any,
                            conditioning: Any) -> torch.Tensor: ...

    def decode_native_canvas(self, native_state: torch.Tensor) -> torch.Tensor: ...


class NativeStateMixin:
    """Shared exact geometry and Gaussian initialization; no RGB round trips."""

    def native_spatial_shape_for_rgb(self, height, width):
        factor = self.native_spatial_factor
        if min(height, width) < 1 or height % factor or width % factor:
            raise ValueError(f"RGB dimensions {(height, width)} must be positive multiples of native factor {factor}")
        return height // factor, width // factor

    def rgb_spatial_shape_for_native(self, height, width):
        if min(height, width) < 1:
            raise ValueError("Native dimensions must be positive")
        return height * self.native_spatial_factor, width * self.native_spatial_factor

    def sample_initial_native_state(self, *, batch_size, native_height, native_width, generator):
        epsilon = torch.randn(batch_size, self.native_channels, native_height, native_width,
                              generator=generator, device=generator.device, dtype=torch.float32)
        return self.initialize_native_state(epsilon)

    def initialize_native_state(self, epsilon):
        """Ordinary sampling initialization from an explicitly supplied Gaussian."""
        return epsilon.to(self.device) * self.native_initial_noise_sigma

    def decode_native_canvas(self, native_state):
        return self.decode_clean(native_state)

    def record_guided_prediction(self):
        self.guided_prediction_count = getattr(self, "guided_prediction_count", 0) + 1

    def native_step_with_endpoints(self, state, timestep, conditioning):
        next_state = self.denoise_native_step(state, timestep, conditioning)
        return next_state, self._endpoints_from_last_prediction(state, timestep)

    def predict_clean_and_endpoint(self, state, timestep, conditioning):
        clean = self.predict_clean_native(state, timestep, conditioning)
        return self._endpoints_from_last_prediction(state, timestep, clean=clean)
