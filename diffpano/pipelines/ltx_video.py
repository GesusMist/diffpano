import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.ltx.pipeline_ltx import LTXPipeline, calculate_shift, retrieve_timesteps
from diffusers.pipelines.ltx.pipeline_output import LTXPipelineOutput
from diffusers.utils import BaseOutput, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

from diffpano.geometry import SphericalFunctions

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        # todo
        ```
"""


@dataclass
class LTXWithLatentsPipelineOutput(BaseOutput):
    r"""
    Output class for LTX pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: torch.Tensor
    latents: torch.Tensor


class SphericalLTXPipeline(LTXPipeline):
    @staticmethod
    def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        assert latents.ndim == 5, f'Expected 5D tensor, got latents.shape={latents.shape}'  # (added) Sanity check
        return LTXPipeline._pack_latents(latents, patch_size, patch_size_t)

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        assert latents.ndim == 3, f'Expected 3D tensor, got latents.shape={latents.shape}'  # (added) Sanity check
        return LTXPipeline._unpack_latents(latents, num_frames, height, width, patch_size, patch_size_t)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt_txt_path: str = None,  # (modified) SphereDiff
        negative_prompt_txt_path: str = "",
        height: int = 512,
        width: int = 512,
        num_frames: int = 121,
        frame_rate: int = 24,
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,  # (modified) (B, HxWxF, C)
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        decode_timestep: Union[float, List[float]] = 0.05,
        decode_noise_scale: Optional[Union[float, List[float]]] = 0.025,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 128,
        ### Spherical options ###
        n_spherical_points: int = 2600,
        weighted_average_temperature: float = 0.1,
        erp_height: int = 1024,
        erp_width: int = 2048,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `512`):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, defaults to `704`):
                The width in pixels of the generated image. This is set to 848 by default for the best results.
            num_frames (`int`, defaults to `161`):
                The number of video frames to generate
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, defaults to `3 `):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
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
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            decode_timestep (`float`, defaults to `0.0`):
                The timestep at which generated video is decoded.
            decode_noise_scale (`float`, defaults to `None`):
                The interpolation factor between random noise and denoised latents at the decode timestep.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ltx.LTXPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
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
            max_sequence_length (`int` defaults to `128 `):
                Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.ltx.LTXPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ltx.LTXPipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
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
        thetas = torch.tensor(thetas, device=device, dtype=self.dtype)
        phis = torch.tensor(phis, device=device, dtype=self.dtype)
        prompt_dir = SphericalFunctions.spherical_to_cartesian(thetas, phis)

        if negative_prompt_txt_path != '' and negative_prompt_txt_path is not None:
            with open(negative_prompt_txt_path, 'r') as f:
                negative_prompt = f.read().strip('\n')
        else:
            negative_prompt = ''

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        batch_size = 1   # TODO support batch_size > 1
        assert batch_size == 1, 'batch_size should be 1'

        device = self._execution_device

        # 3. Prepare text embeddings
        num_prompt = len(prompt)
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels

        # (added) Spherical: sample points on sphere
        spherical_points = SphericalFunctions.fibonacci_sphere(N=n_spherical_points).to(device, dtype=self.dtype)  # (N, 3)
        num_points_on_sphere = spherical_points.shape[0]
        shape = (batch_size, num_channels_latents, (num_frames - 1) // self.vae_temporal_compression_ratio + 1, num_points_on_sphere)  # ? (added) Spherical
        spherical_points = spherical_points.repeat(batch_size, (num_frames - 1) // self.vae_temporal_compression_ratio + 1, 1, 1)

        # (added) Spherical: view directions
        view_dir = SphericalFunctions.horizontal_and_vertical_view_dirs_v3_fov_xy_dense_equator()
        view_dir = view_dir.to(device, dtype=self.dtype)  # (N, 3)
        num_inference_steps_view_dir = len(view_dir)
        multi_prompts_indices_main, fovs_main = SphericalFunctions.get_prompt_indices(view_dir, prompt_dir, prompt_fovs)

        print(f'num_points_on_sphere = {num_points_on_sphere}, num_inference_steps_view_dir = {num_inference_steps_view_dir}')

        latents = randn_tensor(shape, generator, device, dtype=self.dtype)

        # _pack => b c f h w => b c f*h*w
        # _unpack => b c f*h*w => b c f h w

        # 4. Prepare timesteps
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        aspect_ratio = fovs_main[0][1] / fovs_main[0][0]
        A_FOV_ratio = 4 * math.sin(fovs_main[0][1] / 180 * math.pi / 2) * math.sin(fovs_main[0][0] / 180 * math.pi / 2) / (4 * math.pi)  # TODO: approximated
        latent_height = round(math.sqrt(num_points_on_sphere * A_FOV_ratio / aspect_ratio))
        latent_width = round(aspect_ratio * latent_height)
        video_sequence_length = latent_num_frames * latent_height * latent_width  # (modified) Flexible
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            video_sequence_length,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Prepare micro-conditions
        latent_frame_rate = frame_rate / self.vae_temporal_compression_ratio
        rope_interpolation_scale = (
            1 / latent_frame_rate,
            self.vae_spatial_compression_ratio,
            self.vae_spatial_compression_ratio,
        )

        def selected_j_inside(j_inside):  # use it for debugging
            # return j_inside == 2
            # return j_inside in (0, 1, 14, 15, 29, 43, 54, 65, 73, 81, 85)
            return True

        n_total = len(view_dir) * len(timesteps)
        progress_bar = self.progress_bar(total=n_total)
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            latents_next = torch.zeros_like(latents)
            latents_next_cnt = torch.zeros_like(latents)

            _view_dir = view_dir
            _multi_prompts_indices = multi_prompts_indices_main

            for j_inside in range(len(_view_dir)):
                if not selected_j_inside(j_inside):
                    progress_bar.update()
                    continue

                cur_view_dir = _view_dir[j_inside].repeat(batch_size, 1)  # (B, 3)
                _fov = fovs_main[j_inside]

                ### Dynamic Latent Sampling ###
                indices_new, weight = SphericalFunctions.dynamic_latent_sampling(
                    spherical_points, cur_view_dir, num_points_on_sphere, _fov,
                    temperature=weighted_average_temperature, center_first=True,
                )
                cur_latent_height = round(indices_new.shape[-1]**0.5)

                _latents = latents[..., indices_new]  # (B, C, F, N)
                _latents = rearrange(_latents, 'b c f n -> b (f n) c')

                ### Denoising Step ###
                latent_model_input = torch.cat([_latents] * 2) if self.do_classifier_free_guidance else _latents
                latent_model_input = latent_model_input.to(self.dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # Multi-Prompts: get prompt_embeds and prompt_attention_mask
                _prompt_embeds = prompt_embeds[torch.tensor([_multi_prompts_indices[j_inside], _multi_prompts_indices[j_inside] + num_prompt])]
                _prompt_attention_mask = prompt_attention_mask[torch.tensor([_multi_prompts_indices[j_inside], _multi_prompts_indices[j_inside] + num_prompt])].bool()

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=_prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=_prompt_attention_mask,
                    num_frames=latent_num_frames,
                    height=cur_latent_height,
                    width=cur_latent_height,
                    rope_interpolation_scale=rope_interpolation_scale,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                self.scheduler._step_index = None  # ! important
                _latents = self.scheduler.step(noise_pred, t, _latents, return_dict=False)[0]

                _latents = rearrange(_latents, 'b (f n) c -> b c f n', f=latent_num_frames)

                for idx_b in range(batch_size):
                    latents_next[idx_b, ..., indices_new] += _latents[idx_b] * weight
                    latents_next_cnt[idx_b, ..., indices_new] += weight

                # ? (added) Spherical temporal update

                progress_bar.update()
                progress_bar.set_description_str(f'i: {i}, j: {j_inside}')
                progress_bar.set_postfix_str(f'num points = {len(indices_new)}')

            latents_next_cnt[latents_next_cnt == 0] = 1
            latents = latents_next / latents_next_cnt

            # call the callback, if provided
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            if XLA_AVAILABLE:
                xm.mark_step()

        wb = torch.zeros((batch_size, 3, num_frames, erp_height, erp_width), device=device, dtype=torch.float)
        wb_cnt = torch.zeros_like(wb)

        with self.progress_bar(total=len(view_dir)) as progress_bar:  # (added) ERP
            for j_inside in range(len(view_dir)):
                if not selected_j_inside(j_inside):
                    progress_bar.update()
                    continue

                cur_view_dir = view_dir[j_inside].repeat(batch_size, 1)  # (B, 3)
                fov_vae = fovs_main[j_inside]

                ### Dynamic Latent Sampling ###
                indices_new, weight = SphericalFunctions.dynamic_latent_sampling(
                    spherical_points, cur_view_dir, num_points_on_sphere, _fov,
                    temperature=weighted_average_temperature, center_first=True,
                )
                cur_latent_height = round(indices_new.shape[-1]**0.5)

                _latents = latents[..., indices_new]  # (B, C, F, N)
                _latents = rearrange(_latents, 'b c f (h w) -> b c f h w', h=cur_latent_height)

                _latents = self._denormalize_latents(
                    _latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
                )
                _latents = _latents.to(prompt_embeds.dtype)

                if not self.vae.config.timestep_conditioning:
                    timestep = None
                else:
                    noise = torch.randn(_latents.shape, generator=generator, device=device, dtype=_latents.dtype)
                    if not isinstance(decode_timestep, list):
                        decode_timestep = [decode_timestep] * batch_size
                    if decode_noise_scale is None:
                        decode_noise_scale = decode_timestep
                    elif not isinstance(decode_noise_scale, list):
                        decode_noise_scale = [decode_noise_scale] * batch_size

                    timestep = torch.tensor(decode_timestep, device=device, dtype=_latents.dtype)
                    decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=_latents.dtype)[
                        :, None, None, None, None
                    ]
                    _latents = (1 - decode_noise_scale) * _latents + decode_noise_scale * noise

                ### (added) Spatial Loop: add margin to latents and remove it to video (start) ###
                video = self.vae.decode(_latents, timestep, return_dict=False)[0]

                wb, wb_cnt = SphericalFunctions.paste_perspective_to_erp_rectangle(
                    wb, video.to(wb.device, wb.dtype), cur_view_dir.to(wb.device, wb.dtype), fov=fov_vae,
                    add=True, interpolate=True, interpolation_mode='bilinear',
                    panorama_cnt=wb_cnt, return_cnt=True, temperature=weighted_average_temperature,
                )

                progress_bar.update()

        wb_cnt[wb_cnt == 0] = 1
        wb /= wb_cnt

        _output_type = output_type if output_type != 'np_with_latent' else 'np'
        video = self.video_processor.postprocess_video(wb, output_type=_output_type)

        if output_type == 'np_with_latent':
            if not return_dict:
                return (video, latents)
            return LTXWithLatentsPipelineOutput(frames=video, latents=latents)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return LTXPipelineOutput(frames=video)
