import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from diffusers.pipelines.sana import SanaPipeline
from diffusers.pipelines.sana.pipeline_output import SanaPipelineOutput
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

from .pixel_fusion import (
    apply_configured_random_seed,
    apply_pixel_space_fusion,
    build_pixel_fusion_config,
    decode_view_latents,
    exclusive_owner_diagnostics,
    get_or_build_exclusive_owner_map,
    predict_clean_latents,
    run_time_travel,
    save_exclusive_owner_diagnostics,
    should_apply_pixel_fusion,
    should_apply_time_travel,
    summarize_patch_geometry,
    temporary_save_fused_clean_erp_debug,
    temporary_save_original_clean_erp_debug,
    write_back_views_exclusive,
    write_back_views_weighted_average,
)
from .spherical_functions import SphericalFunctions

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        # todo
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class SphericalSanaPipeline(SanaPipeline):

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt_txt_path: str = None,  # (modified) SphereDiff
        negative_prompt_txt_path: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        height: int = 1024,
        width: int = 1024,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        clean_caption: bool = False,
        use_resolution_binning: bool = True,
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
        ### Spherical options ###
        n_spherical_points: int = 2600,
        weighted_average_temperature: float = 0.1,
        erp_height: int = 2048,
        erp_width: int = 4096,
        pixel_fusion_config: Optional[Union[Dict[str, Any], str]] = None,
        pixel_fusion_config_path: Optional[str] = None,
    ) -> Union[SanaPipelineOutput, Tuple]:
        """
        Function invoked when calling the SphericalSanaPipeline for spherical panoramic image generation.

        Args:
            prompt_txt_path (`str`, *optional*):
                Path to text file containing prompts.
            negative_prompt_txt_path (`str`, *optional*, defaults to ""):
                Path to text file containing negative prompt. If empty or None, no negative prompt is used.
            num_inference_steps (`int`, *optional*, defaults to 20):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*): 
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For Sana this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.sana.pipeline_output.SanaPipelineOutput`] instead of a plain tuple.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
                the requested resolution. Useful for generating non-square images.
            attention_kwargs (`Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to `300`):
                Maximum sequence length to use with the `prompt`.
            complex_human_instruction (`List[str]`, *optional*):
                Instructions for complex human attention:
                https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.
            n_spherical_points (`int`, *optional*, defaults to 2600):
                Arguments for the sphere sampling function (number of points).
            weighted_average_temperature (`float`, *optional*, defaults to 0.1):
                Method for merging overlapping patches during spherical generation.
            view_dir_vae_fn_args (`List[int]`, *optional*, defaults to (80, 40)):
                Arguments for VAE view direction function during decoding.
            erp_height (`int`, *optional*, defaults to 1024):
                Height of the final equirectangular panoramic image.
            erp_width (`int`, *optional*, defaults to 2048):
                Width of the final equirectangular panoramic image.

        Examples:

        Returns:
            [`~pipelines.sana.pipeline_output.SanaPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.sana.pipeline_output.SanaPipelineOutput`] is returned with
                spherical panoramic images, otherwise a `tuple` is returned where the first element is the generated
                equirectangular panoramic image tensor.
        """
        device = self._execution_device

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # load prompts
        with open(prompt_txt_path, 'r') as f:
            lines = f.readlines()
        prompt_raw = [line.strip() for line in lines]
        assert len(prompt_raw) == 5, 'prompt_txt_path should contain 5 lines'
        prompt, thetas, phis, prompt_fovs = [], [], [], []
        phis_raw = [-90, -10, 0, 10, 90]
        for i in range(len(phis_raw)):
            for theta in [0, 90, 180, 270]:
                prompt.append(prompt_raw[i])
                thetas.append(math.radians(theta))
                phis.append(math.radians(phis_raw[i]))
                prompt_fovs.append((80, 80))
        thetas = torch.tensor(thetas, device=device, dtype=torch.float32)
        phis = torch.tensor(phis, device=device, dtype=torch.float32)
        prompt_dir = SphericalFunctions.spherical_to_cartesian(thetas, phis)

        if negative_prompt_txt_path != '' and negative_prompt_txt_path is not None:
            with open(negative_prompt_txt_path, 'r') as f:
                negative_prompt = f.read().strip('\n')
        else:
            negative_prompt = ''

        # 1. Check inputs. Raise error if not correct
        if use_resolution_binning:
            if self.transformer.config.sample_size == 64:
                aspect_ratio_bin = ASPECT_RATIO_2048_BIN
            elif self.transformer.config.sample_size == 32:
                aspect_ratio_bin = ASPECT_RATIO_1024_BIN
            elif self.transformer.config.sample_size == 16:
                aspect_ratio_bin = ASPECT_RATIO_512_BIN
            else:
                raise ValueError("Invalid sample size")
            orig_height, orig_width = height, width
            height, width = self.image_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

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

        batch_size = 1   # TODO support batch_size > 1
        assert batch_size == 1, 'batch_size should be 1'

        lora_scale = self.attention_kwargs.get("scale", None) if self.attention_kwargs is not None else None

        # 3. Encode input prompt
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
            num_images_per_prompt=num_images_per_prompt,
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

        assert self.do_classifier_free_guidance, 'do_classifier_free_guidance should be True'
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels

        # SphereDiff: sample points on sphere
        spherical_points = SphericalFunctions.fibonacci_sphere(N=n_spherical_points).to(device, dtype=torch.float32)  # (N, 3)
        num_points_on_sphere = spherical_points.shape[0]
        shape = (batch_size, latent_channels, 1, num_points_on_sphere)
        spherical_points = spherical_points.repeat(batch_size, 1, 1, 1)

        # SphereDiff: view directions
        view_dir = SphericalFunctions.horizontal_and_vertical_view_dirs_v3_fov_xy_dense_equator()
        view_dir = view_dir.to(device, dtype=torch.float32)  # (N, 3)
        num_inference_steps_view_dir = len(view_dir)
        multi_prompts_indices_main, fovs_main = SphericalFunctions.get_prompt_indices(view_dir, prompt_dir, prompt_fovs)

        print(f'num_points_on_sphere = {num_points_on_sphere}, num_inference_steps_view_dir = {num_inference_steps_view_dir}')

        pixel_fusion_cfg = build_pixel_fusion_config(pixel_fusion_config, pixel_fusion_config_path)
        generator = apply_configured_random_seed(generator, pixel_fusion_cfg, device=device)
        latents = randn_tensor(shape, generator, device, dtype=self.dtype)
        self.sphere_diff_run_metadata = {
            "pixel_fusion_config": pixel_fusion_cfg.to_dict(),
            "n_spherical_points": int(num_points_on_sphere),
            "num_denoising_steps": len(timesteps),
            "denoise_timesteps": timesteps.detach().cpu().tolist(),
            "num_dynamic_view_patches_per_step": int(num_inference_steps_view_dir),
            "view_directions": view_dir.detach().float().cpu().tolist(),
            "view_fovs": [[float(value) for value in fov] for fov in fovs_main],
            "view_prompt_indices": multi_prompts_indices_main.detach().cpu().tolist(),
            "generator_initial_seeds": (
                [item.initial_seed() for item in generator]
                if isinstance(generator, list)
                else [generator.initial_seed()]
                if isinstance(generator, torch.Generator)
                else [torch.cuda.initial_seed() if device.type == "cuda" else torch.initial_seed()]
            ),
            "denoise_patch_point_counts_by_step": [],
            "pixel_fusion_applied_by_step": [],
            "pixel_fusion_timings_seconds_by_step": [],
            "render_patch_point_counts": [],
        }

        # 7. Denoising loop
        self._num_timesteps = len(timesteps)

        def selected_j_inside(j_inside):  # use it for debugging
            # return j_inside == 2
            # return j_inside in (0, 1, 14, 15, 29, 43, 54, 65, 73, 81, 85)
            return True

        n_total = len(view_dir) * len(timesteps)
        progress_bar = self.progress_bar(total=n_total)

        for i, t in enumerate(timesteps):

            latents_next = torch.zeros_like(latents)
            latents_next_cnt = torch.zeros_like(latents)
            apply_fusion = should_apply_pixel_fusion(i, len(timesteps), pixel_fusion_cfg)
            if should_apply_time_travel(i, len(timesteps), pixel_fusion_cfg):
                run_time_travel()
            fusion_records = []
            # TEMPORARY DEBUG EXPORT: baseline predicted-clean views; remove with the marked helper.
            original_clean_debug_records = []
            save_original_clean_debug = (
                pixel_fusion_cfg.temporary_save_original_clean_erp_per_step
                and not pixel_fusion_cfg.pixel_fusion_enabled
            )
            step_patch_point_counts = []
            step_fusion_timings = None
            exclusive_writeback_result = None

            _view_dir = view_dir
            _multi_prompts_indices = multi_prompts_indices_main

            for j_inside in range(len(_view_dir)):
                if not selected_j_inside(j_inside):
                    progress_bar.update()
                    continue

                cur_view_dir = _view_dir[j_inside].repeat(batch_size, 1)  # (B, 3)
                _fov = fovs_main[j_inside]

                ### Dynamic Latent Sampling ###
                indices_new, weight = SphericalFunctions.dynamic_laetent_sampling(
                    spherical_points, cur_view_dir, num_points_on_sphere, _fov,
                    temperature=weighted_average_temperature, center_first=True,
                )
                step_patch_point_counts.append(int(indices_new.numel()))
                _latents = latents[..., indices_new]  # (B, C, F, N)
                _latents = _latents.squeeze(2)
                cur_latent_height = round(indices_new.shape[-1]**0.5)
                _latents = rearrange(_latents, 'b c (h w) -> b c h w', h=cur_latent_height)

                ### Denoising Step ###
                latent_model_input = torch.cat([_latents] * 2) if self.do_classifier_free_guidance else _latents
                latent_model_input = latent_model_input.to(self.dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)

                # Multi-Prompts: get prompt_embeds and prompt_attention_mask
                _prompt_embeds = prompt_embeds[torch.tensor([_multi_prompts_indices[j_inside], _multi_prompts_indices[j_inside] + num_prompt])]
                _prompt_attention_mask = prompt_attention_mask[torch.tensor([_multi_prompts_indices[j_inside], _multi_prompts_indices[j_inside] + num_prompt])].bool()

                # predict noise model_output
                noise_pred = self.transformer(
                    latent_model_input,
                    encoder_hidden_states=_prompt_embeds,
                    encoder_attention_mask=_prompt_attention_mask,
                    timestep=timestep,
                    return_dict=False,
                    attention_kwargs=self.attention_kwargs,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                else:
                    noise_pred = noise_pred

                # compute previous image: x_t -> x_t-1
                self.scheduler._step_index = None  # ! important
                latents_before_step = _latents
                if apply_fusion or save_original_clean_debug:
                    # Pixel fusion operates on predicted-clean view latents x0 with shape [B, C, H_lat, W_lat].
                    clean_latents, _, _ = predict_clean_latents(self.scheduler, noise_pred, t, latents_before_step)
                _latents = self.scheduler.step(noise_pred, t, latents_before_step, return_dict=False)[0]

                if apply_fusion:
                    fusion_records.append(
                        {
                            "clean_latents": clean_latents,
                            "current_latents": latents_before_step,
                            "model_output": noise_pred,
                            "prev_latents": _latents,
                            "indices": indices_new,
                            "weight": weight,
                            "patch_id": j_inside,
                            "view_dir": cur_view_dir,
                            "fov": _fov,
                        }
                    )
                else:
                    # TEMPORARY DEBUG EXPORT: observe x0 without changing the original write-back path.
                    if save_original_clean_debug:
                        original_clean_debug_records.append(
                            {"clean_latents": clean_latents, "view_dir": cur_view_dir, "fov": _fov}
                        )
                    _latents = rearrange(_latents, 'b c h w -> b c 1 (h w)')
                    for idx_b in range(batch_size):
                        latents_next[idx_b, ..., indices_new] += _latents[idx_b] * weight
                        latents_next_cnt[idx_b, ..., indices_new] += weight

                progress_bar.update()
                progress_bar.set_description_str(f'i: {i}, j: {j_inside}')
                progress_bar.set_postfix_str(f'num points = {len(indices_new)}')

            if apply_fusion and fusion_records:
                target_latent_size = (
                    max(record["clean_latents"].shape[-2] for record in fusion_records),
                    max(record["clean_latents"].shape[-1] for record in fusion_records),
                )

                def resize_view_latents(view_latents, size):
                    if view_latents.shape[-2:] == size:
                        return view_latents
                    return F.interpolate(view_latents, size=size, mode="bilinear", align_corners=True)

                fusion_result = apply_pixel_space_fusion(
                    vae=self.vae,
                    scheduler=self.scheduler,
                    timestep=t,
                    clean_latents=torch.cat(
                        [resize_view_latents(record["clean_latents"], target_latent_size) for record in fusion_records], dim=0
                    ),
                    current_latents=torch.cat(
                        [resize_view_latents(record["current_latents"], target_latent_size) for record in fusion_records], dim=0
                    ),
                    model_output=torch.cat(
                        [resize_view_latents(record["model_output"], target_latent_size) for record in fusion_records], dim=0
                    ),
                    prev_latents=torch.cat(
                        [resize_view_latents(record["prev_latents"], target_latent_size) for record in fusion_records], dim=0
                    ),
                    view_dirs=torch.cat([record["view_dir"] for record in fusion_records], dim=0),
                    fovs=[record["fov"] for record in fusion_records],
                    erp_height=erp_height,
                    erp_width=erp_width,
                    config=pixel_fusion_cfg,
                    generator=generator if isinstance(generator, torch.Generator) else None,
                )
                # TEMPORARY DEBUG EXPORT: remove with the marked helper in pixel_fusion.py.
                temporary_save_fused_clean_erp_debug(
                    fusion_result.fused_erp,
                    step_index=i,
                    timestep=t,
                    config=pixel_fusion_cfg,
                    pipeline_name="sana",
                )
                if pixel_fusion_cfg.measure_performance:
                    print(f"pixel_fusion_timings_seconds={fusion_result.timings}")
                step_fusion_timings = dict(fusion_result.timings)
                corrected_view_latents = []
                for idx_record, record in enumerate(fusion_records):
                    # Convert fused previous-view latents back to SphereDiff's persistent spherical latent layout [B, C, F=1, N].
                    batch_start = idx_record * batch_size
                    batch_end = batch_start + batch_size
                    fused_record = fusion_result.fused_prev_latents[batch_start:batch_end]
                    fused_record = resize_view_latents(fused_record, record["clean_latents"].shape[-2:])
                    fused_latents = rearrange(
                        fused_record,
                        'b c h w -> b c 1 (h w)',
                    )
                    corrected_view_latents.append(fused_latents)

                if pixel_fusion_cfg.spherical_writeback_mode == "weighted_average":
                    # Original SphereDiff weighted write-back retained for ablation B.
                    for corrected_patch, record in zip(corrected_view_latents, fusion_records):
                        for idx_b in range(batch_size):
                            latents_next[idx_b, ..., record["indices"]] += corrected_patch[idx_b] * record["weight"]
                            latents_next_cnt[idx_b, ..., record["indices"]] += record["weight"]
                else:
                    patch_indices = [record["indices"] for record in fusion_records]
                    patch_scores = [record["weight"] for record in fusion_records]
                    patch_ids = [record["patch_id"] for record in fusion_records]
                    patch_view_dirs = [record["view_dir"] for record in fusion_records]
                    patch_fovs = [record["fov"] for record in fusion_records]
                    owner_map, owner_cache_key, owner_map_reused = get_or_build_exclusive_owner_map(
                        num_points_on_sphere,
                        patch_indices,
                        patch_scores,
                        patch_ids,
                        patch_view_dirs,
                        patch_fovs,
                        pixel_fusion_cfg,
                        device=latents.device,
                    )
                    weighted_fallback = None
                    if (
                        pixel_fusion_cfg.exclusive_uncovered_mode == "weighted_average_fallback"
                        and not owner_map.covered_mask.all()
                    ):
                        weighted_fallback = write_back_views_weighted_average(
                            latents,
                            corrected_view_latents,
                            patch_indices,
                            patch_scores,
                        )
                    exclusive_writeback_result = write_back_views_exclusive(
                        latents,
                        corrected_view_latents,
                        patch_indices,
                        patch_ids,
                        owner_map,
                        uncovered_mode=pixel_fusion_cfg.exclusive_uncovered_mode,
                        weighted_average_fallback=weighted_fallback,
                        geometry_summary=summarize_patch_geometry(patch_ids, patch_view_dirs, patch_fovs),
                    )
                    if pixel_fusion_cfg.save_owner_map:
                        owner_diagnostics = exclusive_owner_diagnostics(
                            owner_map,
                            exclusive_writeback_result.exclusive_write_count,
                            patch_ids,
                        )
                        save_exclusive_owner_diagnostics(
                            owner_diagnostics,
                            pixel_fusion_cfg,
                            owner_cache_key,
                            pipeline_name="sana",
                            step_index=i,
                        )
                        print(
                            "exclusive_owner_map "
                            f"reused={owner_map_reused} "
                            f"coverage_min={owner_diagnostics['minimum_coverage_count'].item()} "
                            f"coverage_max={owner_diagnostics['maximum_coverage_count'].item()} "
                            f"coverage_mean={owner_diagnostics['mean_coverage_count'].item():.4f} "
                            f"multiply_covered_percent={owner_diagnostics['multiply_covered_percent'].item():.4f} "
                            f"owner_histogram={owner_diagnostics['owner_patch_histogram'].cpu().tolist()}"
                        )

            self.sphere_diff_run_metadata["denoise_patch_point_counts_by_step"].append(step_patch_point_counts)
            self.sphere_diff_run_metadata["pixel_fusion_applied_by_step"].append(bool(apply_fusion and fusion_records))
            self.sphere_diff_run_metadata["pixel_fusion_timings_seconds_by_step"].append(step_fusion_timings)
            if exclusive_writeback_result is not None:
                latents = exclusive_writeback_result.latents
            else:
                latents_next_cnt[latents_next_cnt == 0] = 1
                latents = latents_next / latents_next_cnt

            # TEMPORARY DEBUG EXPORT START: remove this block with temporary_save_original_clean_erp_debug().
            if save_original_clean_debug:
                temporary_save_original_clean_erp_debug(
                    (
                        (
                            decode_view_latents(self.vae, record["clean_latents"], pixel_fusion_cfg),
                            record["view_dir"],
                            record["fov"],
                        )
                        for record in original_clean_debug_records
                    ),
                    step_index=i,
                    timestep=t,
                    erp_height=erp_height,
                    erp_width=erp_width,
                    weighted_average_temperature=weighted_average_temperature,
                    config=pixel_fusion_cfg,
                    pipeline_name="sana",
                )
            # TEMPORARY DEBUG EXPORT END

        progress_bar.close()

        wb = torch.zeros((batch_size, 3, 1, erp_height, erp_width), device=device, dtype=torch.float)
        wb_cnt = torch.zeros_like(wb)

        with self.progress_bar(total=len(view_dir)) as progress_bar:
            for j_inside in range(len(view_dir)):
                if not selected_j_inside(j_inside):
                    progress_bar.update()
                    continue

                cur_view_dir = view_dir[j_inside].repeat(batch_size, 1)  # (B, 3)
                fov_vae = fovs_main[j_inside]

                ### Dynamic Latent Sampling ###
                indices_new, weight = SphericalFunctions.dynamic_laetent_sampling(
                    spherical_points, cur_view_dir, num_points_on_sphere, _fov,
                    temperature=weighted_average_temperature, center_first=True,
                )
                self.sphere_diff_run_metadata["render_patch_point_counts"].append(int(indices_new.numel()))
                cur_latent_height = round(indices_new.shape[-1]**0.5)

                _latents = latents[..., indices_new]  # (B, C, F, N)
                _latents = rearrange(_latents, 'b c f (h w) -> b c f h w', h=cur_latent_height)

                _latents = _latents.to(self.vae.dtype)
                _latents = _latents[:, :, 0, :, :]  # (B, C, H, W)

                image = self.vae.decode(_latents / self.vae.config.scaling_factor, return_dict=False)[0]
                if use_resolution_binning:
                    image = self.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)

                image = image.unsqueeze(2)  # (B, C, 1, H, W)

                wb, wb_cnt = SphericalFunctions.paste_perspective_to_erp_rectangle(
                    wb, image.to(wb.device, wb.dtype), cur_view_dir.to(wb.device, wb.dtype), fov=fov_vae,
                    add=True, interpolate=True, interpolation_mode='bilinear',
                    panorama_cnt=wb_cnt, return_cnt=True, temperature=weighted_average_temperature,
                )

                progress_bar.update()

        wb_cnt[wb_cnt == 0] = 1
        wb /= wb_cnt

        image = self.image_processor.postprocess(wb[:, :, 0, :, :], output_type='pil')

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return SanaPipelineOutput(images=image)
