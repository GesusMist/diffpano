"""Planar patch adapters for Stable Diffusion 2 and FLUX experiments."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import FluxPipeline, StableDiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.utils.torch_utils import randn_tensor

from diffpano.geometry import SphericalFunctions
from diffpano.initialization import apply_configured_random_seed, load_directional_prompts
from diffpano.pipelines.flux import calculate_shift, retrieve_timesteps as retrieve_flux_timesteps
from diffpano.pixel_fusion import (
    build_identity_preserving_vae_target,
    decode_view_latents,
    encode_view_images,
    predict_clean_latents,
    reinject_fused_latents,
)
from experiments.planar.fusion import (
    PlanarPatchFusionConfig,
    blend_planar_patches,
    build_planar_owner_map,
    build_planar_patch_layout,
    extract_planar_patches,
    planar_patch_prompt_indices,
    scale_planar_patch_layout,
    write_back_planar_latents,
)


def _expanded_prompts(prompt_txt_path: str, device: torch.device):
    prompt_raw = load_directional_prompts(prompt_txt_path)
    prompt, thetas, phis = [], [], []
    for prompt_text, phi_degrees in zip(prompt_raw, (-90, -10, 0, 10, 90)):
        for theta_degrees in (0, 90, 180, 270):
            prompt.append(prompt_text)
            thetas.append(math.radians(theta_degrees))
            phis.append(math.radians(phi_degrees))
    directions = SphericalFunctions.spherical_to_cartesian(
        torch.tensor(thetas, device=device, dtype=torch.float32),
        torch.tensor(phis, device=device, dtype=torch.float32),
    )
    return prompt, directions


def _layouts(height, width, vae_scale_factor, planar_config):
    if height % vae_scale_factor or width % vae_scale_factor:
        raise ValueError(f"Output {(height, width)} must be divisible by VAE scale factor {vae_scale_factor}")
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    layout = build_planar_patch_layout(
        latent_height,
        latent_width,
        planar_config.patch_latent_height,
        planar_config.patch_latent_width,
        planar_config.patch_stride_height,
        planar_config.patch_stride_width,
    )
    rgb_layout = scale_planar_patch_layout(layout, vae_scale_factor, vae_scale_factor)
    return latent_height, latent_width, layout, rgb_layout


def _pixel_clean_targets(
    *,
    vae,
    clean_patches,
    layout,
    rgb_layout,
    planar_config,
    fusion_config,
    latent_to_vae_latents=None,
    vae_latents_to_latent=None,
):
    vae_clean_patches = (
        latent_to_vae_latents(clean_patches) if latent_to_vae_latents is not None else clean_patches
    )
    decoded = decode_view_latents(vae, vae_clean_patches, fusion_config).float()
    expected = (rgb_layout.patch_height, rgb_layout.patch_width)
    if decoded.shape[-2:] != expected:
        raise ValueError(f"Decoded patch size {tuple(decoded.shape[-2:])} does not match expected {expected}")
    fused_rgb = blend_planar_patches(decoded, rgb_layout, fusion_config).fused_values.unsqueeze(0)
    fused_rgb_patches = extract_planar_patches(fused_rgb, rgb_layout)
    if planar_config.use_vae_residual_bridge:
        bridge = build_identity_preserving_vae_target(
            vae,
            clean_patches,
            decoded,
            fused_rgb_patches,
            fusion_config,
            latent_to_vae_latents=latent_to_vae_latents,
            vae_latents_to_latent=vae_latents_to_latent,
        )
        return bridge.target_clean_latents
    fused_vae_latents = encode_view_images(vae, fused_rgb_patches, fusion_config).float()
    return vae_latents_to_latent(fused_vae_latents) if vae_latents_to_latent is not None else fused_vae_latents


def _final_patch_decode(vae, latents, layout, rgb_layout, fusion_config, image_processor):
    patches = extract_planar_patches(latents, layout)
    rgb_patches = decode_view_latents(vae, patches, fusion_config).float()
    image = blend_planar_patches(rgb_patches, rgb_layout, fusion_config).fused_values.unsqueeze(0)
    return image_processor.postprocess(image, output_type="pil")


def _sd2_alpha_beta(scheduler, timestep, sample):
    index = int(timestep.flatten()[0].item())
    alpha = scheduler.alphas_cumprod[index].to(device=sample.device, dtype=torch.float32)
    beta = 1 - alpha
    while alpha.ndim < sample.ndim:
        alpha = alpha.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
    return alpha, beta


def _sd2_predict_clean(scheduler, model_output, timestep, sample):
    sample = sample.float()
    model_output = model_output.float()
    alpha, beta = _sd2_alpha_beta(scheduler, timestep, sample)
    return (sample - beta.sqrt() * model_output) / alpha.sqrt().clamp_min(torch.finfo(torch.float32).eps)


def _sd2_reinject(scheduler, timestep, current, model_output, original_clean, target_clean, strength):
    alpha, beta = _sd2_alpha_beta(scheduler, timestep, current)
    correction = float(strength) * (target_clean.float() - original_clean.float())
    corrected_output = model_output.float() - alpha.sqrt() * correction / beta.sqrt().clamp_min(
        torch.finfo(torch.float32).eps
    )
    return scheduler.step(corrected_output, timestep, current, return_dict=False)[0]


class PlanarPatchSD2Pipeline(StableDiffusionPipeline):
    """SD2 DDIM planar patches with selectable latent/RGB fusion behavior."""

    @torch.no_grad()
    def __call__(
        self,
        prompt_txt_path: str,
        negative_prompt_txt_path: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        height: int = 640,
        width: int = 4096,
        use_resolution_binning: bool = False,
        planar_fusion_config: Optional[Union[PlanarPatchFusionConfig, Dict[str, Any], str]] = None,
        **kwargs,
    ):
        del use_resolution_binning, kwargs
        device = self._execution_device
        planar_config = PlanarPatchFusionConfig.from_any(planar_fusion_config)
        fusion_config = planar_config.to_pixel_fusion_config()
        generator = apply_configured_random_seed(None, fusion_config, device=device)
        prompts, prompt_directions = _expanded_prompts(prompt_txt_path, device)
        negative_prompt = ""
        if negative_prompt_txt_path:
            negative_prompt = open(negative_prompt_txt_path, encoding="utf-8").read().strip()
        self._guidance_scale = guidance_scale
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompts,
            device,
            1,
            True,
            negative_prompt=[negative_prompt] * len(prompts),
        )
        latent_height, latent_width, layout, rgb_layout = _layouts(
            height, width, self.vae_scale_factor, planar_config
        )
        prompt_indices = planar_patch_prompt_indices(layout, prompt_directions)
        latents = self.prepare_latents(
            1,
            self.unet.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )
        owner_map = None
        if planar_config.latent_writeback_mode == "exclusive":
            owner_map = build_planar_owner_map(layout, fusion_config, device=device)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.sphere_diff_run_metadata = {
            "planar_model": "sd2",
            "planar_fusion_config": planar_config.to_dict(),
            "num_denoising_steps": len(self.scheduler.timesteps),
            "num_dynamic_view_patches_per_step": layout.num_patches,
            "planar_latent_shape": list(latents.shape),
            "planar_patch_positions": [list(item) for item in layout.positions],
            "planar_patch_prompt_indices": prompt_indices.cpu().tolist(),
            "pixel_fusion_applied_by_step": [planar_config.fusion_space == "pixel"] * len(self.scheduler.timesteps),
            "denoise_patch_point_counts_by_step": [],
        }
        print(
            f"planar_sd2 latent={latent_height}x{latent_width} patch="
            f"{layout.patch_height}x{layout.patch_width} patches={layout.num_patches}"
        )
        for step_index, timestep in enumerate(self.scheduler.timesteps):
            current_patches = extract_planar_patches(latents, layout)
            clean_patches, model_outputs, previous_patches = [], [], []
            for patch_index, patch in enumerate(current_patches):
                patch = patch.unsqueeze(0)
                model_input = torch.cat([patch, patch], dim=0)
                model_input = self.scheduler.scale_model_input(model_input, timestep)
                prompt_index = int(prompt_indices[patch_index].item())
                embeds = torch.cat(
                    [negative_prompt_embeds[prompt_index : prompt_index + 1], prompt_embeds[prompt_index : prompt_index + 1]],
                    dim=0,
                )
                prediction = self.unet(model_input, timestep, encoder_hidden_states=embeds, return_dict=False)[0]
                uncond, cond = prediction.chunk(2)
                prediction = uncond + guidance_scale * (cond - uncond)
                previous = self.scheduler.step(prediction, timestep, patch, return_dict=False)[0]
                previous_patches.append(previous)
                if planar_config.fusion_space == "pixel":
                    clean_patches.append(_sd2_predict_clean(self.scheduler, prediction, timestep, patch))
                    model_outputs.append(prediction)
            previous_patches = torch.cat(previous_patches)
            if planar_config.fusion_space == "pixel":
                clean_patches = torch.cat(clean_patches)
                model_outputs = torch.cat(model_outputs)
                target_clean = _pixel_clean_targets(
                    vae=self.vae,
                    clean_patches=clean_patches,
                    layout=layout,
                    rgb_layout=rgb_layout,
                    planar_config=planar_config,
                    fusion_config=fusion_config,
                )
                corrected = _sd2_reinject(
                    self.scheduler,
                    timestep,
                    current_patches,
                    model_outputs,
                    clean_patches,
                    target_clean,
                    planar_config.reinjection_strength,
                )
            else:
                corrected = previous_patches
            latents = write_back_planar_latents(
                latents,
                corrected,
                layout,
                fusion_config,
                mode=planar_config.latent_writeback_mode,
                owner_map=owner_map,
            )
            self.sphere_diff_run_metadata["denoise_patch_point_counts_by_step"].append(
                [layout.patch_height * layout.patch_width] * layout.num_patches
            )
            print(f"completed_step={step_index + 1}/{num_inference_steps}", flush=True)
        images = _final_patch_decode(self.vae, latents, layout, rgb_layout, fusion_config, self.image_processor)
        self.maybe_free_model_hooks()
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=None)


class PlanarPatchFluxPipeline(FluxPipeline):
    """FLUX planar patches with packed scheduler latents and selectable fusion."""

    @torch.no_grad()
    def __call__(
        self,
        prompt_txt_path: str,
        negative_prompt_txt_path: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        height: int = 640,
        width: int = 4096,
        use_resolution_binning: bool = False,
        planar_fusion_config: Optional[Union[PlanarPatchFusionConfig, Dict[str, Any], str]] = None,
        max_sequence_length: int = 512,
        **kwargs,
    ):
        del negative_prompt_txt_path, use_resolution_binning, kwargs
        device = self._execution_device
        planar_config = PlanarPatchFusionConfig.from_any(planar_fusion_config)
        fusion_config = planar_config.to_pixel_fusion_config()
        generator = apply_configured_random_seed(None, fusion_config, device=device)
        prompts, prompt_directions = _expanded_prompts(prompt_txt_path, device)
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        vae_scale = self.vae_scale_factor
        latent_height, latent_width, layout, rgb_layout = _layouts(
            height, width, vae_scale, planar_config
        )
        if layout.patch_height % 2 or layout.patch_width % 2:
            raise ValueError("FLUX VAE-latent patch dimensions must be even for 2x2 packing")
        prompt_indices = planar_patch_prompt_indices(layout, prompt_directions)
        channels = self.transformer.config.in_channels // 4
        latents = randn_tensor(
            (1, channels, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=prompt_embeds.dtype,
        )
        owner_map = None
        if planar_config.latent_writeback_mode == "exclusive":
            owner_map = build_planar_owner_map(layout, fusion_config, device=device)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        patch_seq_len = (layout.patch_height // 2) * (layout.patch_width // 2)
        mu = calculate_shift(
            patch_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_flux_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        guidance = None
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        image_ids = self._prepare_latent_image_ids(
            1, layout.patch_height // 2, layout.patch_width // 2, device, prompt_embeds.dtype
        )
        self._joint_attention_kwargs = {}
        self.sphere_diff_run_metadata = {
            "planar_model": "flux",
            "planar_fusion_config": planar_config.to_dict(),
            "num_denoising_steps": len(timesteps),
            "num_dynamic_view_patches_per_step": layout.num_patches,
            "planar_latent_shape": list(latents.shape),
            "planar_patch_positions": [list(item) for item in layout.positions],
            "planar_patch_prompt_indices": prompt_indices.cpu().tolist(),
            "pixel_fusion_applied_by_step": [planar_config.fusion_space == "pixel"] * len(timesteps),
            "denoise_patch_point_counts_by_step": [],
        }
        print(
            f"planar_flux latent={latent_height}x{latent_width} patch="
            f"{layout.patch_height}x{layout.patch_width} patches={layout.num_patches}"
        )

        def to_vae(packed):
            return self._unpack_latents(packed, layout.patch_height, layout.patch_width, 1)

        def to_packed(vae_latents):
            return self._pack_latents(
                vae_latents,
                vae_latents.shape[0],
                vae_latents.shape[1],
                vae_latents.shape[-2],
                vae_latents.shape[-1],
            )

        for step_index, timestep in enumerate(timesteps):
            current_vae_patches = extract_planar_patches(latents, layout)
            current_patches, clean_patches, model_outputs, previous_patches = [], [], [], []
            for patch_index, vae_patch in enumerate(current_vae_patches):
                packed = to_packed(vae_patch.unsqueeze(0))
                prompt_index = int(prompt_indices[patch_index].item())
                model_timestep = timestep.expand(1).to(packed.dtype)
                prediction = self.transformer(
                    hidden_states=packed,
                    timestep=model_timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds[prompt_index : prompt_index + 1],
                    encoder_hidden_states=prompt_embeds[prompt_index : prompt_index + 1],
                    txt_ids=text_ids,
                    img_ids=image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                self.scheduler._step_index = None
                if planar_config.fusion_space == "pixel":
                    clean, _, _ = predict_clean_latents(self.scheduler, prediction, timestep, packed)
                    clean_patches.append(clean)
                    model_outputs.append(prediction)
                    current_patches.append(packed)
                previous = self.scheduler.step(prediction, timestep, packed, return_dict=False)[0]
                previous_patches.append(previous)
            previous_patches = torch.cat(previous_patches)
            if planar_config.fusion_space == "pixel":
                clean_patches = torch.cat(clean_patches)
                model_outputs = torch.cat(model_outputs)
                current_patches = torch.cat(current_patches)
                target_clean = _pixel_clean_targets(
                    vae=self.vae,
                    clean_patches=clean_patches,
                    layout=layout,
                    rgb_layout=rgb_layout,
                    planar_config=planar_config,
                    fusion_config=fusion_config,
                    latent_to_vae_latents=to_vae,
                    vae_latents_to_latent=to_packed,
                )
                corrected_packed = reinject_fused_latents(
                    clean_patches,
                    target_clean,
                    previous_patches,
                    model_outputs,
                    self.scheduler.sigmas[min(step_index + 1, len(self.scheduler.sigmas) - 1)],
                    fusion_config,
                    scheduler=self.scheduler,
                    timestep=timestep,
                    current_latents=current_patches,
                )
            else:
                corrected_packed = previous_patches
            corrected_vae = to_vae(corrected_packed)
            latents = write_back_planar_latents(
                latents,
                corrected_vae,
                layout,
                fusion_config,
                mode=planar_config.latent_writeback_mode,
                owner_map=owner_map,
            )
            self.sphere_diff_run_metadata["denoise_patch_point_counts_by_step"].append(
                [layout.patch_height * layout.patch_width] * layout.num_patches
            )
            print(f"completed_step={step_index + 1}/{num_inference_steps}", flush=True)
        images = _final_patch_decode(self.vae, latents, layout, rgb_layout, fusion_config, self.image_processor)
        self.maybe_free_model_hooks()
        return FluxPipelineOutput(images=images)
