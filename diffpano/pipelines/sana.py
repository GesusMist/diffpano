"""SANA adapter implementing one explicit RGB -> latent -> RGB scheduler step."""

from dataclasses import dataclass
import time
from typing import Any, Optional, Sequence

import torch

from diffpano.camera import PerspectiveCamera
from diffpano.conditioning import camera_prompt_indices, expand_directional_prompts
from diffpano.pipelines.base import (
    ViewDenoiser,
    make_first_order_scheduler,
    release_prompt_encoders,
    reset_scheduler_step_state,
)
from diffpano.pipelines.clean_prediction import (
    flow_add_noise,
    flow_predicted_clean,
)
from diffpano.vae import decode_view_latents, encode_view_images


@dataclass
class SanaPromptBank:
    prompt_directions: torch.Tensor
    positive: torch.Tensor
    positive_mask: torch.Tensor
    negative: Optional[torch.Tensor]
    negative_mask: Optional[torch.Tensor]


class SanaViewDenoiser(ViewDenoiser):
    """Local SANA latent diffusion; no global latent is retained between calls."""

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
        from diffusers.pipelines.sana import SanaPipeline

        return cls(
            SanaPipeline.from_pretrained(source, **kwargs),
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
        del view_height, view_width
        self.pipeline.scheduler = make_first_order_scheduler(self.pipeline.scheduler)
        self.pipeline.scheduler.set_timesteps(num_steps, device=self.device)
        self._timesteps = self.pipeline.scheduler.timesteps
        self.pipeline._guidance_scale = self.guidance_scale
        self.pipeline._attention_kwargs = None

    @torch.no_grad()
    def prepare_prompt_conditioning(
        self, prompts: Sequence[str], negative_prompt: str = ""
    ) -> SanaPromptBank:
        directional = expand_directional_prompts(prompts)
        do_cfg = self.guidance_scale > 1.0
        negative = [negative_prompt] * len(directional.prompts)
        positive, positive_mask, negative_embeds, negative_mask = self.pipeline.encode_prompt(
            directional.prompts,
            do_cfg,
            negative_prompt=negative,
            num_images_per_prompt=1,
            device=self.device,
            clean_caption=False,
            max_sequence_length=300,
        )
        bank = SanaPromptBank(
            directional.directions,
            positive,
            positive_mask,
            negative_embeds if do_cfg else None,
            negative_mask if do_cfg else None,
        )
        release_prompt_encoders(self.pipeline)
        return bank

    def conditioning_for_cameras(
        self,
        prepared_conditioning: SanaPromptBank,
        cameras: Sequence[PerspectiveCamera],
        *,
        batch_size: int,
    ):
        indices = camera_prompt_indices(cameras, prepared_conditioning.prompt_directions)
        indices = indices.repeat_interleave(batch_size).to(device=self.device)
        positive = prepared_conditioning.positive[indices]
        positive_mask = prepared_conditioning.positive_mask[indices].bool()
        if prepared_conditioning.negative is None:
            return {"embeds": positive, "mask": positive_mask}
        negative = prepared_conditioning.negative[indices]
        negative_mask = prepared_conditioning.negative_mask[indices].bool()
        return {
            "embeds": torch.cat([negative, positive], dim=0),
            "mask": torch.cat([negative_mask, positive_mask], dim=0),
        }

    def _predict_flow(
        self, native_state: torch.Tensor, timestep: Any, conditioning: Any
    ) -> torch.Tensor:
        do_cfg = self.guidance_scale > 1.0
        model_input = (
            torch.cat([native_state, native_state], dim=0)
            if do_cfg
            else native_state
        )
        model_input = model_input.to(dtype=self.dtype)
        timestep_tensor = torch.as_tensor(
            timestep, device=self.device, dtype=native_state.dtype
        )
        expanded_timestep = timestep_tensor.expand(model_input.shape[0])
        prediction = self._timed(
            "model_forward",
            lambda: self.pipeline.transformer(
                model_input,
                encoder_hidden_states=conditioning["embeds"],
                encoder_attention_mask=conditioning["mask"],
                timestep=expanded_timestep,
                return_dict=False,
                attention_kwargs=None,
            )[0].float(),
        )
        if do_cfg:
            unconditional, conditional = prediction.chunk(2)
            prediction = unconditional + self.guidance_scale * (
                conditional - unconditional
            )
        if self.pipeline.transformer.config.out_channels // 2 == native_state.shape[1]:
            prediction = prediction.chunk(2, dim=1)[0]
        return prediction

    @torch.no_grad()
    def denoise_step(self, rgb_view: torch.Tensor, timestep: Any, conditioning: Any) -> torch.Tensor:
        self.last_timings = {}
        latents = self._timed(
            "vae_encode",
            lambda: encode_view_images(self.pipeline.vae, rgb_view.float(), chunk_size=self.vae_chunk_size),
        )
        timestep_tensor = torch.as_tensor(timestep, device=self.device, dtype=latents.dtype)
        prediction = self._predict_flow(latents, timestep_tensor, conditioning)
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
        scale = int(getattr(self.pipeline, "vae_scale_factor", 8))
        channels = int(self.pipeline.transformer.config.in_channels)
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
        return flow_add_noise(
            self.pipeline.scheduler, clean_state, fixed_noise, timestep
        )

    def predict_clean_native(
        self, noisy_state: torch.Tensor, timestep: Any, conditioning: Any
    ) -> torch.Tensor:
        self.last_timings = {}
        prediction = self._predict_flow(noisy_state, timestep, conditioning)
        self.last_model_prediction = prediction.detach()
        return flow_predicted_clean(
            self.pipeline.scheduler, noisy_state, prediction, timestep
        )

    def decode_clean(self, clean_state: torch.Tensor) -> torch.Tensor:
        return decode_view_latents(
            self.pipeline.vae, clean_state.float(), chunk_size=self.vae_chunk_size
        ).float()

    @torch.no_grad()
    def sample_native_rgb(self, *, batch_size: int, height: int, width: int, generator):
        scale = int(getattr(self.pipeline, "vae_scale_factor", 8))
        channels = int(self.pipeline.transformer.config.in_channels)
        latents = torch.randn(
            batch_size,
            channels,
            height // scale,
            width // scale,
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        return decode_view_latents(self.pipeline.vae, latents, chunk_size=self.vae_chunk_size).float()
