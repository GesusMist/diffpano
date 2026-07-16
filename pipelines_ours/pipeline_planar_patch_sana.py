"""EXPERIMENTAL NO-WARP ABLATION for SANA.

Delete this file, planar_patch_fusion.py, and the dedicated config/smoke test to
remove the experiment. No spherical pipeline imports or calls this class.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.sana import SanaPipeline
from diffusers.pipelines.sana.pipeline_output import SanaPipelineOutput

from .pipeline_spherical_sana import retrieve_timesteps
from .pixel_fusion import (
    apply_configured_random_seed,
    decode_view_latents,
    encode_view_images,
    predict_clean_latents,
    reinject_fused_latents,
)
from .planar_patch_fusion import (
    PlanarPatchFusionConfig,
    blend_planar_patches,
    build_planar_owner_map,
    build_planar_patch_layout,
    extract_planar_patches,
    planar_patch_prompt_indices,
    scale_planar_patch_layout,
    write_back_planar_latents,
)
from .spherical_functions import SphericalFunctions


class PlanarPatchSanaPipeline(SanaPipeline):
    """Generate a normal 2D image through overlapping patches without any warp operation."""

    @torch.no_grad()
    def __call__(
        self,
        prompt_txt_path: str,
        negative_prompt_txt_path: str = "",
        num_inference_steps: int = 20,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int = 1,
        height: int = 2048,
        width: int = 4096,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        clean_caption: bool = False,
        use_resolution_binning: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 300,
        complex_human_instruction: List[str] = [
            "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
            "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
            "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
            "Here are examples of how to transform or refine prompts:",
            "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
            "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
            "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
            "User Prompt: ",
        ],
        planar_fusion_config: Optional[Union[PlanarPatchFusionConfig, Dict[str, Any], str]] = None,
        planar_fusion_config_path: Optional[str] = None,
    ) -> Union[SanaPipelineOutput, Tuple]:
        del eta  # SANA's configured DPM solver does not consume eta.
        device = self._execution_device
        if use_resolution_binning:
            raise ValueError("PlanarPatchSanaPipeline requires use_resolution_binning=False for an exact 2D grid")
        if num_images_per_prompt != 1:
            raise ValueError("PlanarPatchSanaPipeline currently supports num_images_per_prompt=1")
        if height % self.vae_scale_factor or width % self.vae_scale_factor:
            raise ValueError(
                f"Output {(height, width)} must be divisible by VAE scale factor {self.vae_scale_factor}"
            )
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        planar_config = PlanarPatchFusionConfig.from_any(planar_fusion_config_path or planar_fusion_config)
        fusion_config = planar_config.to_pixel_fusion_config()
        generator = apply_configured_random_seed(generator, fusion_config, device=device)

        with open(prompt_txt_path, "r", encoding="utf-8") as handle:
            prompt_raw = [line.strip() for line in handle if line.strip()]
        if len(prompt_raw) != 5:
            raise ValueError("prompt_txt_path must contain exactly 5 non-empty lines, matching SphereDiff")

        prompt, thetas, phis = [], [], []
        for prompt_text, phi_degrees in zip(prompt_raw, (-90, -10, 0, 10, 90)):
            for theta_degrees in (0, 90, 180, 270):
                prompt.append(prompt_text)
                thetas.append(math.radians(theta_degrees))
                phis.append(math.radians(phi_degrees))
        prompt_directions = SphericalFunctions.spherical_to_cartesian(
            torch.tensor(thetas, device=device, dtype=torch.float32),
            torch.tensor(phis, device=device, dtype=torch.float32),
        )

        if negative_prompt_txt_path:
            with open(negative_prompt_txt_path, "r", encoding="utf-8") as handle:
                negative_prompt = handle.read().strip()
        else:
            negative_prompt = ""

        self.check_inputs(
            prompt,
            height,
            width,
            callback_on_step_end_tensor_inputs,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        lora_scale = self.attention_kwargs.get("scale") if self.attention_kwargs is not None else None
        num_prompt = len(prompt)
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
            complex_human_instruction=complex_human_instruction,
            lora_scale=lora_scale,
        )
        if not self.do_classifier_free_guidance:
            raise ValueError("PlanarPatchSanaPipeline requires guidance_scale > 1, matching SphericalSanaPipeline")
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        if self.scheduler.config.get("solver_order", 1) > 1:
            print("Warning: planar patch denoising requires solver_order=1; setting it to 1.")
            self.scheduler.register_to_config(solver_order=1)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
        )
        self._num_timesteps = len(timesteps)

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        latent_channels = self.transformer.config.in_channels
        layout = build_planar_patch_layout(
            latent_height,
            latent_width,
            planar_config.patch_latent_height,
            planar_config.patch_latent_width,
            planar_config.patch_stride_height,
            planar_config.patch_stride_width,
        )
        patch_prompt_indices = planar_patch_prompt_indices(layout, prompt_directions)
        latents = self.prepare_latents(
            1,
            latent_channels,
            height,
            width,
            self.dtype,
            device,
            generator,
            latents,
        )
        owner_map = None
        if planar_config.latent_writeback_mode == "exclusive":
            owner_map = build_planar_owner_map(layout, fusion_config, device=device)
            if not owner_map.covered_mask.all():
                raise AssertionError("Planar exclusive owner map must cover every 2D latent cell")

        self.sphere_diff_run_metadata = {
            "planar_fusion_config": planar_config.to_dict(),
            "generator_initial_seeds": (
                [item.initial_seed() for item in generator]
                if isinstance(generator, list)
                else [generator.initial_seed()]
                if isinstance(generator, torch.Generator)
                else [torch.cuda.initial_seed() if device.type == "cuda" else torch.initial_seed()]
            ),
            "num_denoising_steps": len(timesteps),
            "denoise_timesteps": timesteps.detach().cpu().tolist(),
            "num_dynamic_view_patches_per_step": layout.num_patches,
            "denoise_patch_point_counts_by_step": [],
            "pixel_fusion_applied_by_step": [True] * len(timesteps),
            "planar_latent_shape": list(latents.shape),
            "planar_patch_positions": [list(position) for position in layout.positions],
            "planar_patch_prompt_indices": patch_prompt_indices.detach().cpu().tolist(),
        }
        print(
            f"planar_latent_grid={latent_height}x{latent_width}, "
            f"patch={layout.patch_height}x{layout.patch_width}, patches={layout.num_patches}"
        )

        progress_bar = self.progress_bar(total=len(timesteps) * layout.num_patches)
        for step_index, timestep_value in enumerate(timesteps):
            current_patches = extract_planar_patches(latents, layout)
            clean_patches = []
            model_outputs = []
            previous_patches = []
            sigma_next = None

            for patch_index, current_patch in enumerate(current_patches):
                current_patch = current_patch.unsqueeze(0)
                latent_model_input = torch.cat([current_patch, current_patch], dim=0).to(self.dtype)
                model_timestep = timestep_value.expand(latent_model_input.shape[0]).to(latents.dtype)
                prompt_index = int(patch_prompt_indices[patch_index].item())
                selection = torch.tensor(
                    [prompt_index, prompt_index + num_prompt],
                    device=prompt_embeds.device,
                    dtype=torch.long,
                )
                noise_pred = self.transformer(
                    latent_model_input,
                    encoder_hidden_states=prompt_embeds[selection],
                    encoder_attention_mask=prompt_attention_mask[selection].bool(),
                    timestep=model_timestep,
                    return_dict=False,
                    attention_kwargs=self.attention_kwargs,
                )[0].float()
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                self.scheduler._step_index = None
                clean_patch, _, patch_sigma_next = predict_clean_latents(
                    self.scheduler,
                    noise_pred,
                    timestep_value,
                    current_patch,
                )
                previous_patch = self.scheduler.step(
                    noise_pred,
                    timestep_value,
                    current_patch,
                    return_dict=False,
                )[0]
                clean_patches.append(clean_patch)
                model_outputs.append(noise_pred)
                previous_patches.append(previous_patch)
                sigma_next = patch_sigma_next
                progress_bar.update()
                progress_bar.set_description_str(f"planar step={step_index}, patch={patch_index}")

            clean_patches = torch.cat(clean_patches, dim=0)
            model_outputs = torch.cat(model_outputs, dim=0)
            previous_patches = torch.cat(previous_patches, dim=0)
            decoded_clean_patches = decode_view_latents(self.vae, clean_patches, fusion_config).float()
            rgb_scale_height, remainder_height = divmod(
                decoded_clean_patches.shape[-2],
                layout.patch_height,
            )
            rgb_scale_width, remainder_width = divmod(
                decoded_clean_patches.shape[-1],
                layout.patch_width,
            )
            if remainder_height or remainder_width:
                raise ValueError("VAE decoded patch size is not an integer multiple of the latent patch size")
            rgb_layout = scale_planar_patch_layout(layout, rgb_scale_height, rgb_scale_width)
            if (rgb_layout.canvas_height, rgb_layout.canvas_width) != (height, width):
                raise ValueError(
                    f"Decoded planar canvas {(rgb_layout.canvas_height, rgb_layout.canvas_width)} "
                    f"does not match requested output {(height, width)}"
                )
            fused_rgb = blend_planar_patches(decoded_clean_patches, rgb_layout, fusion_config).fused_values.unsqueeze(0)
            fused_rgb_patches = extract_planar_patches(fused_rgb, rgb_layout)
            fused_clean_patches = encode_view_images(
                self.vae,
                fused_rgb_patches,
                fusion_config,
                generator=generator if isinstance(generator, torch.Generator) else None,
            )
            if fused_clean_patches.shape != clean_patches.shape:
                raise ValueError(
                    f"VAE re-encoded patches have shape {tuple(fused_clean_patches.shape)}, "
                    f"expected {tuple(clean_patches.shape)}"
                )

            next_clean_weight = None
            if planar_config.reinjection_mode == "noise_consistent":
                if hasattr(self.scheduler, "_sigma_to_alpha_sigma_t"):
                    next_clean_weight = self.scheduler._sigma_to_alpha_sigma_t(sigma_next)[0]
                else:
                    next_clean_weight = 1 - sigma_next
            corrected_patches = reinject_fused_latents(
                clean_patches,
                fused_clean_patches,
                previous_patches,
                model_outputs,
                sigma_next,
                fusion_config,
                next_clean_weight=next_clean_weight,
            )
            latents = write_back_planar_latents(
                latents,
                corrected_patches,
                layout,
                fusion_config,
                mode=planar_config.latent_writeback_mode,
                owner_map=owner_map,
            )
            self.sphere_diff_run_metadata["denoise_patch_point_counts_by_step"].append(
                [layout.patch_height * layout.patch_width] * layout.num_patches
            )

            if callback_on_step_end is not None:
                callback_kwargs = {name: locals()[name] for name in callback_on_step_end_tensor_inputs}
                callback_outputs = callback_on_step_end(self, step_index, timestep_value, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)

            del decoded_clean_patches, fused_rgb, fused_rgb_patches, fused_clean_patches, corrected_patches

        progress_bar.close()

        if output_type == "latent":
            image = latents
        else:
            final_latent_patches = extract_planar_patches(latents, layout)
            final_rgb_patches = decode_view_latents(self.vae, final_latent_patches, fusion_config).float()
            final_rgb = blend_planar_patches(final_rgb_patches, rgb_layout, fusion_config).fused_values.unsqueeze(0)
            image = self.image_processor.postprocess(final_rgb, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return SanaPipelineOutput(images=image)
