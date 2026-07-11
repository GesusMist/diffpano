import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux import FluxPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

from .pixel_fusion import (
    apply_pixel_space_fusion,
    build_pixel_fusion_config,
    decode_view_latents,
    predict_clean_latents,
    run_time_travel,
    should_apply_pixel_fusion,
    should_apply_time_travel,
    temporary_save_fused_clean_erp_debug,
    temporary_save_original_clean_erp_debug,
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


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class SphericalFluxPipeline(FluxPipeline):
    @staticmethod
    def _pack_latents_for_spherical(latents, batch_size, num_channels_latents_for_spherical, height, width):
        latents = latents.permute(0, 2, 1)  # (batch_size, height, width, num_channels_latents_for_spherical)
        latents = latents.reshape(batch_size, height * width, num_channels_latents_for_spherical)
        return latents

    @staticmethod
    def _unpack_latents_for_spherical(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        latents = latents.permute(0, 2, 1)
        latents = latents.reshape(batch_size, channels, height, width)
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt_txt_path: str = None,  # (modified) SphereDiff
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_txt_path: str = "",
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        ### Spherical options ###
        n_spherical_points: int = 26500,
        weighted_average_temperature: float = 0.1,
        erp_height: int = 2048,
        erp_width: int = 4096,
        pixel_fusion_config: Optional[Union[Dict[str, Any], str]] = None,
        pixel_fusion_config_path: Optional[str] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
        device = self._execution_device

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

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        batch_size = 1

        num_prompt = len(prompt)

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        num_channels_latents = num_channels_latents * 4
        spherical_points = SphericalFunctions.fibonacci_sphere(N=n_spherical_points).to(device, dtype=torch.float32)  # (N, 3)
        num_points_on_sphere = spherical_points.shape[0]
        shape = (batch_size, num_channels_latents, 1, num_points_on_sphere)
        spherical_points = spherical_points.repeat(batch_size, 1, 1, 1)

        view_dir = SphericalFunctions.horizontal_and_vertical_view_dirs_v3_fov_xy_dense_equator()
        view_dir = view_dir.to(device, dtype=torch.float32)  # (N, 3)
        num_inference_steps_view_dir = len(view_dir)
        multi_prompts_indices_main, fovs_main = SphericalFunctions.get_prompt_indices(view_dir, prompt_dir, prompt_fovs)

        print(f'num_points_on_sphere = {num_points_on_sphere}, num_inference_steps_view_dir = {num_inference_steps_view_dir}')

        latents = randn_tensor(shape, generator, device, dtype=self.dtype)
        pixel_fusion_cfg = build_pixel_fusion_config(pixel_fusion_config, pixel_fusion_config_path)
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

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        n_total = len(view_dir) * len(timesteps)

        def selected_j_inside(j_inside):  # use it for debugging
            # return j_inside == 2
            # return j_inside in (0, 1, 14, 15, 29, 43, 54, 65, 73, 81, 85)
            return True

        progress_bar = self.progress_bar(total=n_total)
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t

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
                    temperature=weighted_average_temperature, center_first=False,
                )
                step_patch_point_counts.append(int(indices_new.numel()))
                cur_latent_height = round(indices_new.shape[-1]**0.5)
                _latents = latents[..., indices_new]  # (B, C, F, N)
                _latents = _latents.squeeze(2)
                _latents = self._pack_latents_for_spherical(_latents, batch_size, num_channels_latents, cur_latent_height, cur_latent_height)

                ### Denoising Step ###
                latent_model_input = _latents
                latent_model_input = latent_model_input.to(self.dtype)

                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # Multi-Prompts: get prompt_embeds and prompt_attention_mask
                _prompt_embeds = prompt_embeds[_multi_prompts_indices[j_inside]].unsqueeze(dim=0)
                _pooled_prompt_embeds = pooled_prompt_embeds[_multi_prompts_indices[j_inside]].unsqueeze(dim=0)

                latent_image_ids = self._prepare_latent_image_ids(batch_size, cur_latent_height, cur_latent_height, device, latents.dtype)
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=_pooled_prompt_embeds,
                    encoder_hidden_states=_prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                self.scheduler._step_index = None  # ! important
                latents_before_step = _latents
                if apply_fusion or save_original_clean_debug:
                    # FLUX view latents are packed as [B, H_lat * W_lat, C * 4]; adapters below convert to VAE latents.
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
                            "view_dir": cur_view_dir,
                            "fov": _fov,
                            "latent_height": cur_latent_height,
                        }
                    )
                else:
                    # TEMPORARY DEBUG EXPORT: observe x0 without changing the original write-back path.
                    if save_original_clean_debug:
                        original_clean_debug_records.append(
                            {
                                "clean_latents": clean_latents,
                                "view_dir": cur_view_dir,
                                "fov": _fov,
                                "latent_height": cur_latent_height,
                            }
                        )
                    _latents = self._unpack_latents_for_spherical(_latents, cur_latent_height, cur_latent_height, 1)
                    _latents = rearrange(_latents, 'b c h w -> b c 1 (h w)')
                    for idx_b in range(batch_size):
                        latents_next[idx_b, ..., indices_new] += _latents[idx_b] * weight
                        latents_next_cnt[idx_b, ..., indices_new] += weight

                progress_bar.update()
                progress_bar.set_description_str(f'i: {i}, j: {j_inside}')
                progress_bar.set_postfix_str(f'num points = {len(indices_new)}')

            if apply_fusion and fusion_records:
                reference_latent_height = max(record["latent_height"] for record in fusion_records)

                def resize_packed_view_latents(view_latents, source_height, target_height):
                    if source_height == target_height:
                        return view_latents
                    vae_latents = self._unpack_latents(view_latents, source_height * 2, source_height * 2, 1)
                    vae_latents = F.interpolate(
                        vae_latents,
                        size=(target_height * 2, target_height * 2),
                        mode="bilinear",
                        align_corners=True,
                    )
                    return self._pack_latents(
                        vae_latents,
                        vae_latents.shape[0],
                        vae_latents.shape[1],
                        vae_latents.shape[-2],
                        vae_latents.shape[-1],
                    )

                def latent_to_vae_latents(view_latents):
                    # [views, H_lat * W_lat, C * 4] -> [views, C, 2 * H_lat, 2 * W_lat] for the FLUX VAE.
                    return self._unpack_latents(view_latents, reference_latent_height * 2, reference_latent_height * 2, 1)

                def vae_latents_to_latent(vae_latents):
                    # [views, C, 2 * H_lat, 2 * W_lat] -> [views, H_lat * W_lat, C * 4] for scheduler reinjection/write-back.
                    return self._pack_latents(
                        vae_latents,
                        vae_latents.shape[0],
                        vae_latents.shape[1],
                        vae_latents.shape[-2],
                        vae_latents.shape[-1],
                    )

                fusion_result = apply_pixel_space_fusion(
                    vae=self.vae,
                    scheduler=self.scheduler,
                    timestep=t,
                    clean_latents=torch.cat(
                        [
                            resize_packed_view_latents(record["clean_latents"], record["latent_height"], reference_latent_height)
                            for record in fusion_records
                        ],
                        dim=0,
                    ),
                    current_latents=torch.cat(
                        [
                            resize_packed_view_latents(record["current_latents"], record["latent_height"], reference_latent_height)
                            for record in fusion_records
                        ],
                        dim=0,
                    ),
                    model_output=torch.cat(
                        [
                            resize_packed_view_latents(record["model_output"], record["latent_height"], reference_latent_height)
                            for record in fusion_records
                        ],
                        dim=0,
                    ),
                    prev_latents=torch.cat(
                        [
                            resize_packed_view_latents(record["prev_latents"], record["latent_height"], reference_latent_height)
                            for record in fusion_records
                        ],
                        dim=0,
                    ),
                    view_dirs=torch.cat([record["view_dir"] for record in fusion_records], dim=0),
                    fovs=[record["fov"] for record in fusion_records],
                    erp_height=erp_height,
                    erp_width=erp_width,
                    config=pixel_fusion_cfg,
                    latent_to_vae_latents=latent_to_vae_latents,
                    vae_latents_to_latent=vae_latents_to_latent,
                    generator=generator if isinstance(generator, torch.Generator) else None,
                )
                # TEMPORARY DEBUG EXPORT: remove with the marked helper in pixel_fusion.py.
                temporary_save_fused_clean_erp_debug(
                    fusion_result.fused_erp,
                    step_index=i,
                    timestep=t,
                    config=pixel_fusion_cfg,
                    pipeline_name="flux",
                )
                if pixel_fusion_cfg.measure_performance:
                    print(f"pixel_fusion_timings_seconds={fusion_result.timings}")
                step_fusion_timings = dict(fusion_result.timings)
                for idx_record, record in enumerate(fusion_records):
                    batch_start = idx_record * batch_size
                    batch_end = batch_start + batch_size
                    fused_record = resize_packed_view_latents(
                        fusion_result.fused_prev_latents[batch_start:batch_end],
                        reference_latent_height,
                        record["latent_height"],
                    )
                    fused_latents = self._unpack_latents_for_spherical(
                        fused_record,
                        record["latent_height"],
                        record["latent_height"],
                        1,
                    )
                    fused_latents = rearrange(fused_latents, 'b c h w -> b c 1 (h w)')
                    for idx_b in range(batch_size):
                        latents_next[idx_b, ..., record["indices"]] += fused_latents[idx_b] * record["weight"]
                        latents_next_cnt[idx_b, ..., record["indices"]] += record["weight"]

            self.sphere_diff_run_metadata["denoise_patch_point_counts_by_step"].append(step_patch_point_counts)
            self.sphere_diff_run_metadata["pixel_fusion_applied_by_step"].append(bool(apply_fusion and fusion_records))
            self.sphere_diff_run_metadata["pixel_fusion_timings_seconds_by_step"].append(step_fusion_timings)
            latents_next_cnt[latents_next_cnt == 0] = 1
            latents = latents_next / latents_next_cnt

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            # TEMPORARY DEBUG EXPORT START: remove this block with temporary_save_original_clean_erp_debug().
            if save_original_clean_debug:
                temporary_save_original_clean_erp_debug(
                    (
                        (
                            decode_view_latents(
                                self.vae,
                                self._unpack_latents(
                                    record["clean_latents"],
                                    record["latent_height"] * 2,
                                    record["latent_height"] * 2,
                                    1,
                                ),
                                pixel_fusion_cfg,
                            ),
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
                    pipeline_name="flux",
                )
            # TEMPORARY DEBUG EXPORT END

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

        progress_bar.close()

        self._current_timestep = None

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
                    temperature=weighted_average_temperature, center_first=False,
                )
                self.sphere_diff_run_metadata["render_patch_point_counts"].append(int(indices_new.numel()))
                cur_latent_height = round(indices_new.shape[-1]**0.5)

                _latents = latents[..., indices_new].squeeze(2)  # (B, C, F, N)
                _latents = self._unpack_latents(_latents.permute(0, 2, 1), cur_latent_height * 2, cur_latent_height * 2, 1)
                _latents = _latents.unsqueeze(dim=2)

                _latents = _latents.to(self.vae.dtype)
                _latents = _latents[:, :, 0, :, :]  # (B, C, H, W)

                image = self.vae.decode(_latents / self.vae.config.scaling_factor, return_dict=False)[0]

                image = image.unsqueeze(2)  # (B, C, 1, H, W)

                # save image separately
                wb, wb_cnt = SphericalFunctions.paste_perspective_to_erp_rectangle(
                    wb, image.to(wb.device, wb.dtype), cur_view_dir.to(wb.device, wb.dtype), fov=fov_vae,
                    add=True, interpolate=True, interpolation_mode='bilinear',
                    panorama_cnt=wb_cnt, return_cnt=True, temperature=weighted_average_temperature,
                )

                progress_bar.update()

        wb_cnt[wb_cnt == 0] = 1
        wb /= wb_cnt

        image = self.image_processor.postprocess(wb[:, :, 0, :, :], output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, )

        return FluxPipelineOutput(images=image)
