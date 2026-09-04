"""Stable Diffusion 2 adapter implementing one RGB-to-latent scheduler step."""

from dataclasses import dataclass
import time
from typing import Any, Optional, Sequence

import torch

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
from diffpano.pipelines.clean_prediction import ddim_predicted_clean
from diffpano.vae import decode_view_latents, encode_view_images


@dataclass
class SD2PromptBank:
    prompt_directions: torch.Tensor
    positive: torch.Tensor
    negative: Optional[torch.Tensor]


class SD2ViewDenoiser(ViewDenoiser):
    """Keep SD2's CLIP, U-Net, DDIM scheduler, and VAE inside the RGB adapter."""

    def __init__(
        self,
        pipeline: Any,
        *,
        guidance_scale: float,
        vae_chunk_size: int = 1,
        measure_performance: bool = False,
    ):
        self.pipeline = pipeline
        self.guidance_scale = guidance_scale
        self.vae_chunk_size = vae_chunk_size
        self.measure_performance = measure_performance
        self.last_timings = {}
        self._timesteps = torch.empty(0)

    @classmethod
    def from_pretrained(
        cls,
        source: str,
        *,
        guidance_scale: float,
        vae_chunk_size: int = 1,
        measure_performance: bool = False,
        **kwargs,
    ):
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        kwargs.setdefault("safety_checker", None)
        kwargs.setdefault("requires_safety_checker", False)
        pipeline = StableDiffusionPipeline.from_pretrained(source, **kwargs)
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        return cls(
            pipeline,
            guidance_scale=guidance_scale,
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
        return self.pipeline.unet.dtype

    @property
    def timesteps(self) -> torch.Tensor:
        return self._timesteps

    def to(self, *args, **kwargs):
        self.pipeline.to(*args, **kwargs)
        return self

    def enable_model_cpu_offload(self) -> None:
        self.pipeline.enable_model_cpu_offload()

    def prepare(self, *, num_steps: int, view_height: int, view_width: int) -> None:
        del view_height, view_width
        ensure_first_order_scheduler(self.pipeline.scheduler)
        self.pipeline.scheduler.set_timesteps(num_steps, device=self.device)
        self._timesteps = self.pipeline.scheduler.timesteps
        self.pipeline._guidance_scale = self.guidance_scale
        self.pipeline._cross_attention_kwargs = None

    @torch.no_grad()
    def prepare_prompt_conditioning(
        self, prompts: Sequence[str], negative_prompt: str = ""
    ) -> SD2PromptBank:
        directional = expand_directional_prompts(prompts)
        do_cfg = self.guidance_scale > 1.0
        positive, negative = self.pipeline.encode_prompt(
            prompt=directional.prompts,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=[negative_prompt] * len(directional.prompts),
        )
        bank = SD2PromptBank(directional.directions, positive, negative if do_cfg else None)
        release_prompt_encoders(self.pipeline)
        return bank

    def conditioning_for_cameras(
        self,
        prepared_conditioning: SD2PromptBank,
        cameras: Sequence[PerspectiveCamera],
        *,
        batch_size: int,
    ):
        indices = camera_prompt_indices(cameras, prepared_conditioning.prompt_directions)
        indices = indices.repeat_interleave(batch_size).to(device=self.device)
        positive = prepared_conditioning.positive[indices]
        if prepared_conditioning.negative is None:
            return positive
        return torch.cat([prepared_conditioning.negative[indices], positive], dim=0)

    def conditioning_for_prompt_indices(
        self,
        prepared_conditioning: SD2PromptBank,
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
        positive = prepared_conditioning.positive[indices]
        if prepared_conditioning.negative is None:
            return positive
        return torch.cat([prepared_conditioning.negative[indices], positive], dim=0)

    def _predict_noise(
        self, native_state: torch.Tensor, timestep: Any, conditioning: Any
    ) -> torch.Tensor:
        do_cfg = self.guidance_scale > 1.0
        model_input = (
            torch.cat([native_state, native_state], dim=0)
            if do_cfg
            else native_state
        )
        timestep_tensor = torch.as_tensor(timestep, device=self.device)
        model_input = self.pipeline.scheduler.scale_model_input(
            model_input, timestep_tensor
        )
        prediction = self._timed(
            "model_forward",
            lambda: self.pipeline.unet(
                model_input.to(dtype=self.dtype),
                timestep_tensor,
                encoder_hidden_states=conditioning,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0].float(),
        )
        if do_cfg:
            unconditional, conditional = prediction.chunk(2)
            prediction = unconditional + self.guidance_scale * (
                conditional - unconditional
            )
        return prediction

    @torch.no_grad()
    def denoise_step(self, rgb_view: torch.Tensor, timestep: Any, conditioning: Any) -> torch.Tensor:
        self.last_timings = {}
        latents = self._timed(
            "vae_encode",
            lambda: encode_view_images(self.pipeline.vae, rgb_view.float(), chunk_size=self.vae_chunk_size),
        )
        timestep_tensor = torch.as_tensor(timestep, device=self.device)
        prediction = self._predict_noise(latents, timestep_tensor, conditioning)
        reset_scheduler_step_state(self.pipeline.scheduler)
        next_latents = self._timed(
            "scheduler_step",
            lambda: self.pipeline.scheduler.step(
                prediction, timestep_tensor, latents, return_dict=False
            )[0],
        )
        return self._timed(
            "vae_decode",
            lambda: decode_view_latents(
                self.pipeline.vae, next_latents.float(), chunk_size=self.vae_chunk_size
            ).float(),
        )

    def sample_fixed_noise(
        self, *, batch_size: int, height: int, width: int, generator
    ) -> torch.Tensor:
        scale = int(self.pipeline.vae_scale_factor)
        channels = int(self.pipeline.unet.config.in_channels)
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
        return fixed_noise.to(self.device) * float(
            self.pipeline.scheduler.init_noise_sigma
        )

    def encode_clean(self, rgb_clean: torch.Tensor) -> torch.Tensor:
        return encode_view_images(
            self.pipeline.vae, rgb_clean.float(), chunk_size=self.vae_chunk_size
        )

    def add_fixed_noise(
        self, clean_state: torch.Tensor, fixed_noise: torch.Tensor, timestep: Any
    ) -> torch.Tensor:
        timestep_batch = torch.as_tensor(
            timestep, device=clean_state.device, dtype=torch.long
        ).reshape(1)
        timestep_batch = timestep_batch.expand(clean_state.shape[0])
        return self.pipeline.scheduler.add_noise(
            clean_state, fixed_noise.to(clean_state), timestep_batch
        )

    def predict_clean_native(
        self, noisy_state: torch.Tensor, timestep: Any, conditioning: Any
    ) -> torch.Tensor:
        self.last_timings = {}
        prediction = self._predict_noise(noisy_state, timestep, conditioning)
        self.last_model_prediction = prediction.detach()
        return ddim_predicted_clean(
            self.pipeline.scheduler, noisy_state, prediction, timestep
        )

    def decode_clean(self, clean_state: torch.Tensor) -> torch.Tensor:
        return decode_view_latents(
            self.pipeline.vae, clean_state.float(), chunk_size=self.vae_chunk_size
        ).float()

    @torch.no_grad()
    def sample_native_rgb(self, *, batch_size: int, height: int, width: int, generator):
        scale = int(self.pipeline.vae_scale_factor)
        channels = int(self.pipeline.unet.config.in_channels)
        latents = torch.randn(
            batch_size,
            channels,
            height // scale,
            width // scale,
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        return decode_view_latents(
            self.pipeline.vae, latents, chunk_size=self.vae_chunk_size
        ).float()
