"""FLUX adapter with backend-local token packing and one-step scheduling."""

import inspect
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch

from diffpano.pipelines.native_state import NativeStateMixin
from diffpano.camera import PerspectiveCamera
from diffpano.conditioning import (
    camera_prompt_indices,
    expand_directional_prompts,
    expanded_prompt_indices,
)
from diffpano.pipelines.base import (
    ViewDenoiser,
    ensure_first_order_scheduler,
    release_prompt_encoders,
    reset_scheduler_step_state,
)
from diffpano.pipelines.clean_prediction import (
    flow_add_noise,
    flow_predicted_clean,
)
from diffpano.vae import decode_view_latents, encode_view_images


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * slope + base_shift - slope * base_seq_len


@dataclass
class FluxPromptBank:
    prompt_directions: torch.Tensor
    positive: torch.Tensor
    pooled: torch.Tensor
    text_ids: torch.Tensor
    negative: Optional[torch.Tensor] = None
    negative_pooled: Optional[torch.Tensor] = None
    negative_text_ids: Optional[torch.Tensor] = None


class FluxViewDenoiser(NativeStateMixin, ViewDenoiser):
    """Keep FLUX packing inside the adapter; the ERP loop only sees RGB."""

    def __init__(
        self,
        pipeline: Any,
        *,
        guidance_scale: float,
        true_cfg_scale: float = 1.0,
        vae_chunk_size: int = 1,
        measure_performance: bool = False,
    ):
        self.pipeline = pipeline
        self.guidance_scale = guidance_scale
        self.true_cfg_scale = true_cfg_scale
        self.vae_chunk_size = vae_chunk_size
        self.measure_performance = measure_performance
        self.last_timings = {}
        self._timesteps = torch.empty(0)
        self._view_size = (0, 0)

    @classmethod
    def from_pretrained(
        cls,
        source: str,
        *,
        guidance_scale: float,
        true_cfg_scale: float = 1.0,
        vae_chunk_size: int = 1,
        measure_performance: bool = False,
        **kwargs,
    ):
        from diffusers import FluxPipeline

        return cls(
            FluxPipeline.from_pretrained(source, **kwargs),
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            vae_chunk_size=vae_chunk_size,
            measure_performance=measure_performance,
        )

    def _timed(self, name: str, operation):
        if self.measure_performance and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        output = operation()
        if self.measure_performance and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_timings[name] = self.last_timings.get(name, 0.0) + time.perf_counter() - started
        return output

    @property
    def device(self) -> torch.device:
        return torch.device(self.pipeline._execution_device)

    @property
    def dtype(self) -> torch.dtype:
        return self.pipeline.transformer.dtype

    @property
    def timesteps(self) -> torch.Tensor:
        return self._timesteps

    def to(self, *args, **kwargs):
        self.pipeline.to(*args, **kwargs)
        return self

    def enable_model_cpu_offload(self) -> None:
        self.pipeline.enable_model_cpu_offload()

    def prepare(self, *, num_steps: int, view_height: int, view_width: int) -> None:
        ensure_first_order_scheduler(self.pipeline.scheduler)
        self._view_size = (view_height, view_width)
        scale = int(self.pipeline.vae_scale_factor) * 2
        if view_height % scale or view_width % scale:
            raise ValueError(f"FLUX model resolution must be divisible by {scale}")
        image_seq_len = (view_height // scale) * (view_width // scale)
        sigma = float(getattr(self.pipeline.scheduler, "init_noise_sigma", 1.0))
        if sigma != 1.0:
            raise ValueError("FLUX expects unscaled Gaussian initialization")
        cfg = self.pipeline.scheduler.config
        mu = calculate_shift(
            image_seq_len,
            cfg.get("base_image_seq_len", 256),
            cfg.get("max_image_seq_len", 4096),
            cfg.get("base_shift", 0.5),
            cfg.get("max_shift", 1.15),
        )
        self.scheduler_image_seq_len = image_seq_len
        self.scheduler_shift_mu = mu
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        kwargs = {"sigmas": sigmas, "device": self.device}
        if "mu" in inspect.signature(self.pipeline.scheduler.set_timesteps).parameters:
            kwargs["mu"] = mu
        self.pipeline.scheduler.set_timesteps(**kwargs)
        self._timesteps = self.pipeline.scheduler.timesteps
        self.pipeline._guidance_scale = self.guidance_scale
        self.pipeline._joint_attention_kwargs = {}

    @torch.no_grad()
    def prepare_prompt_conditioning(
        self, prompts: Sequence[str], negative_prompt: str = ""
    ) -> FluxPromptBank:
        directional = expand_directional_prompts(prompts)
        positive, pooled, text_ids = self.pipeline.encode_prompt(
            prompt=directional.prompts,
            prompt_2=None,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        if self.true_cfg_scale > 1.0 and negative_prompt:
            negative, negative_pooled, negative_text_ids = self.pipeline.encode_prompt(
                prompt=[negative_prompt] * len(directional.prompts),
                prompt_2=None,
                device=self.device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )
        else:
            negative = negative_pooled = negative_text_ids = None
        bank = FluxPromptBank(
            directional.directions,
            positive,
            pooled,
            text_ids,
            negative,
            negative_pooled,
            negative_text_ids,
        )
        release_prompt_encoders(self.pipeline)
        return bank

    def conditioning_for_cameras(
        self,
        prepared_conditioning: FluxPromptBank,
        cameras: Sequence[PerspectiveCamera],
        *,
        batch_size: int,
    ):
        indices = camera_prompt_indices(cameras, prepared_conditioning.prompt_directions)
        indices = indices.repeat_interleave(batch_size).to(device=self.device)
        result = {
            "embeds": prepared_conditioning.positive[indices],
            "pooled": prepared_conditioning.pooled[indices],
            "text_ids": prepared_conditioning.text_ids,
        }
        if prepared_conditioning.negative is not None:
            result.update(
                negative=prepared_conditioning.negative[indices],
                negative_pooled=prepared_conditioning.negative_pooled[indices],
                negative_text_ids=prepared_conditioning.negative_text_ids,
            )
        return result

    def conditioning_for_prompt_indices(
        self,
        prepared_conditioning: FluxPromptBank,
        prompt_indices: Sequence[int],
        *,
        batch_size: int,
    ):
        indices = expanded_prompt_indices(
            prompt_indices,
            batch_size=batch_size,
            num_prompts=prepared_conditioning.positive.shape[0],
            device=self.device,
        )
        result = {
            "embeds": prepared_conditioning.positive[indices],
            "pooled": prepared_conditioning.pooled[indices],
            "text_ids": prepared_conditioning.text_ids,
        }
        if prepared_conditioning.negative is not None:
            result.update(
                negative=prepared_conditioning.negative[indices],
                negative_pooled=prepared_conditioning.negative_pooled[indices],
                negative_text_ids=prepared_conditioning.negative_text_ids,
            )
        return result

    def _pack(self, raw: torch.Tensor) -> torch.Tensor:
        return self.pipeline._pack_latents(
            raw, raw.shape[0], raw.shape[1], raw.shape[-2], raw.shape[-1]
        )

    def _unpack(self, packed: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return self.pipeline._unpack_latents(
            packed, height, width, self.pipeline.vae_scale_factor
        )

    def _predict(self, packed: torch.Tensor, timestep: torch.Tensor, conditioning: Any, *, native_shape=None) -> torch.Tensor:
        batch = packed.shape[0]
        if native_shape is None:
            native_shape = self.native_spatial_shape_for_rgb(*self._view_size)
        packed_h, packed_w = self._validate_native_geometry(native_shape)
        if packed.shape[1] != packed_h * packed_w:
            raise ValueError("FLUX packed token count differs from native patch grid")
        image_ids = self.pipeline._prepare_latent_image_ids(
            batch, packed_h, packed_w, self.device, packed.dtype
        )
        if self.pipeline.transformer.config.guidance_embeds:
            guidance = torch.full(
                (batch,), self.guidance_scale, device=self.device, dtype=torch.float32
            )
        else:
            guidance = None
        return self.pipeline.transformer(
            hidden_states=packed.to(dtype=self.dtype),
            timestep=timestep.expand(batch) / 1000,
            guidance=guidance,
            pooled_projections=conditioning["pooled"],
            encoder_hidden_states=conditioning["embeds"],
            txt_ids=conditioning["text_ids"],
            img_ids=image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]

    def _guided_prediction(
        self, packed: torch.Tensor, timestep: torch.Tensor, conditioning: Any, *, native_shape=None
    ) -> torch.Tensor:
        self.record_guided_prediction()
        prediction = self._timed(
            "model_forward", lambda: self._predict(packed, timestep, conditioning, native_shape=native_shape)
        )
        if "negative" in conditioning:
            negative_conditioning = {
                "embeds": conditioning["negative"],
                "pooled": conditioning["negative_pooled"],
                "text_ids": conditioning["negative_text_ids"],
            }
            negative_prediction = self._timed(
                "model_forward",
                lambda: self._predict(
                    packed, timestep, negative_conditioning, native_shape=native_shape
                ),
            )
            prediction = negative_prediction + self.true_cfg_scale * (
                prediction - negative_prediction
            )
        return prediction

    @torch.no_grad()
    def denoise_step(self, rgb_view: torch.Tensor, timestep: Any, conditioning: Any) -> torch.Tensor:
        self.last_timings = {}
        raw = self._timed(
            "vae_encode",
            lambda: encode_view_images(self.pipeline.vae, rgb_view.float(), chunk_size=self.vae_chunk_size),
        )
        packed = self._pack(raw)
        timestep_tensor = torch.as_tensor(timestep, device=self.device, dtype=packed.dtype)
        prediction = self._guided_prediction(packed, timestep_tensor, conditioning, native_shape=raw.shape[-2:])
        reset_scheduler_step_state(self.pipeline.scheduler)
        next_packed = self._timed(
            "scheduler_step",
            lambda: self.pipeline.scheduler.step(
                prediction, timestep_tensor, packed, return_dict=False
            )[0],
        )
        next_raw = self._unpack(next_packed, rgb_view.shape[-2], rgb_view.shape[-1])
        return self._timed(
            "vae_decode",
            lambda: decode_view_latents(
                self.pipeline.vae, next_raw.float(), chunk_size=self.vae_chunk_size
            ).float(),
        )

    @property
    def native_channels(self):
        return int(self.pipeline.transformer.config.in_channels // 4)

    @property
    def native_spatial_factor(self):
        return int(self.pipeline.vae_scale_factor)

    @property
    def native_initial_noise_sigma(self):
        return 1.0

    @torch.no_grad()
    def denoise_native_step(self, native_state, timestep, conditioning):
        self.last_timings = {}
        height, width = self.rgb_spatial_shape_for_native(*native_state.shape[-2:])
        self._validate_native_geometry(native_state.shape[-2:])
        packed = self._pack(native_state)
        timestep_tensor = torch.as_tensor(timestep, device=self.device, dtype=packed.dtype)
        prediction = self._guided_prediction(
            packed, timestep_tensor, conditioning, native_shape=native_state.shape[-2:]
        )
        self.last_model_prediction = prediction.detach()
        reset_scheduler_step_state(self.pipeline.scheduler)
        next_packed = self.pipeline.scheduler.step(
            prediction, timestep_tensor, packed, return_dict=False
        )[0]
        return self._unpack(next_packed, height, width).float()

    def _validate_native_geometry(self, native_shape):
        height, width = native_shape
        if height % 2 or width % 2:
            raise ValueError("FLUX raw native patch dimensions must be even for 2x2 packing")
        rgb_shape = self.rgb_spatial_shape_for_native(height, width)
        if rgb_shape != self._view_size:
            raise ValueError(f"FLUX patch resolution {rgb_shape} differs from prepared scheduler resolution {self._view_size}")
        return height // 2, width // 2

    def sample_fixed_noise(
        self, *, batch_size: int, height: int, width: int, generator
    ) -> torch.Tensor:
        scale = int(self.pipeline.vae_scale_factor)
        channels = int(self.pipeline.transformer.config.in_channels // 4)
        return torch.randn(
            batch_size,
            channels,
            height // scale,
            width // scale,
            generator=generator,
            device=generator.device,
            dtype=torch.float32,
        )

    def make_initial_noisy_state(
        self, fixed_noise: torch.Tensor, timestep: Any
    ) -> torch.Tensor:
        del timestep
        # FluxPipeline.prepare_latents starts from raw Gaussian latents; its
        # FlowMatch scheduler intentionally has no init_noise_sigma multiplier.
        return fixed_noise.to(self.device)

    def encode_clean(self, rgb_clean: torch.Tensor) -> torch.Tensor:
        return encode_view_images(
            self.pipeline.vae, rgb_clean.float(), chunk_size=self.vae_chunk_size
        )

    def add_fixed_noise(
        self, clean_state: torch.Tensor, fixed_noise: torch.Tensor, timestep: Any
    ) -> torch.Tensor:
        return flow_add_noise(
            self.pipeline.scheduler, clean_state, fixed_noise, timestep
        )

    def predict_clean_native(
        self, noisy_state: torch.Tensor, timestep: Any, conditioning: Any
    ) -> torch.Tensor:
        self.last_timings = {}
        packed = self._pack(noisy_state)
        timestep_tensor = torch.as_tensor(
            timestep, device=self.device, dtype=packed.dtype
        )
        prediction = self._guided_prediction(
            packed, timestep_tensor, conditioning, native_shape=noisy_state.shape[-2:]
        )
        self.last_model_prediction = prediction.detach()
        predicted_clean_packed = flow_predicted_clean(
            self.pipeline.scheduler, packed, prediction, timestep
        )
        return self._unpack(
            predicted_clean_packed, *self.rgb_spatial_shape_for_native(*noisy_state.shape[-2:])
        )

    def _endpoints_from_last_prediction(self, state, timestep, clean=None):
        from diffpano.pipelines.endpoints import flow_bounds, flow_endpoints
        velocity = self._unpack(self.last_model_prediction,
                               *self.rgb_spatial_shape_for_native(*state.shape[-2:]))
        return flow_endpoints(state, velocity,
                              *flow_bounds(self.pipeline.scheduler, timestep, state), clean=clean)

    def decode_clean(self, clean_state: torch.Tensor) -> torch.Tensor:
        return decode_view_latents(
            self.pipeline.vae, clean_state.float(), chunk_size=self.vae_chunk_size
        ).float()

    @torch.no_grad()
    def sample_native_rgb(self, *, batch_size: int, height: int, width: int, generator):
        scale = int(self.pipeline.vae_scale_factor)
        channels = int(self.pipeline.transformer.config.in_channels // 4)
        raw = torch.randn(
            batch_size,
            channels,
            height // scale,
            width // scale,
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        return decode_view_latents(self.pipeline.vae, raw, chunk_size=self.vae_chunk_size).float()
