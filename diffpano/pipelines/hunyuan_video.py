import math
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jsonlines
import numpy as np
import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import HunyuanVideoPipeline, retrieve_timesteps
from diffusers.pipelines.hunyuan_video.pipeline_output import HunyuanVideoPipelineOutput
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
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


DEFAULT_PROMPT_TEMPLATE = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}


class SphericalHunyuanVideoPipeline(HunyuanVideoPipeline):

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt_txt_path: str = None,  # (modified) SphereDiff
        negative_prompt_txt_path: str = "",
        height: int = 720,
        width: int = 720,
        num_frames: int = 129,
        num_inference_steps: int = 20,
        sigmas: List[float] = None,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        max_sequence_length: int = 256,
        ### Spherical options ###
        n_spherical_points: int = 14400,
        weighted_average_temperature: float = 0.1,
        erp_height: int = 2048,
        erp_width: int = 4096,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            height (`int`, defaults to `720`):
                The height in pixels of the generated image.
            width (`int`, defaults to `1280`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `129`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, defaults to `6.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality. Note that the only available HunyuanVideo model is
                CFG-distilled, which means that traditional guidance between unconditional and conditional latent is
                not applied.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HunyuanVideoPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned, otherwise a `tuple` is returned
                where the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
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
        prompt_2 = None

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds,
            callback_on_step_end_tensor_inputs,
            prompt_template,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # # 2. Define call parameters
        batch_size = 1

        # 3. Encode input prompt
        num_prompt = len(prompt)
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_template=prompt_template,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # (added) Spherical: sample points on sphere
        num_channels_latents = num_channels_latents * 4
        spherical_points = SphericalFunctions.fibonacci_sphere(N=n_spherical_points).to(device, dtype=self.dtype)  # (N, 3)
        num_points_on_sphere = spherical_points.shape[0]
        shape = (batch_size, num_channels_latents, num_latent_frames, num_points_on_sphere)
        spherical_points = spherical_points.repeat(batch_size, num_latent_frames, 1, 1)

        # (added) Spherical: view directions
        view_dir = SphericalFunctions.horizontal_and_vertical_view_dirs_v3_fov_xy_dense_equator()
        view_dir = view_dir.to(device, dtype=self.dtype)  # (N, 3)
        num_inference_steps_view_dir = len(view_dir)
        multi_prompts_indices_main, fovs_main = SphericalFunctions.get_prompt_indices(view_dir, prompt_dir, prompt_fovs)

        print(f'num_points_on_sphere = {num_points_on_sphere}, num_inference_steps_view_dir = {num_inference_steps_view_dir}')

        latents = randn_tensor(shape, generator, device, dtype=self.dtype)  # ? (added) Spherical

        # 4. Prepare timesteps
        aspect_ratio = fovs_main[0][1] / fovs_main[0][0]
        A_FOV_ratio = 4 * math.sin(fovs_main[0][1] / 180 * math.pi / 2) * math.sin(fovs_main[0][0] / 180 * math.pi / 2) / (4 * math.pi)  # TODO: approximated
        latent_height = round(math.sqrt(num_points_on_sphere * A_FOV_ratio / aspect_ratio))
        latent_width = round(aspect_ratio * latent_height)

        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )

        # 6. Prepare guidance condition
        guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

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
                    temperature=weighted_average_temperature, center_first=False,
                )
                cur_latent_height = round(indices_new.shape[-1]**0.5)
                _latents = latents[..., indices_new]  # (B, C, F, N)
                _latents = rearrange(_latents, 'b c f (h w) -> b c f h w', h=cur_latent_height, w=cur_latent_height)
                _latents = rearrange(_latents, 'b (c ph pw) f h w -> b c f (h ph) (w pw)', ph=2, pw=2)

                ### Denoising Step ###
                latent_model_input = _latents.to(transformer_dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(_latents.shape[0]).to(_latents.dtype)

                # (added) Multi-Prompts: get prompt_embeds and prompt_attention_mask
                _prompt_embeds = prompt_embeds[_multi_prompts_indices[j_inside]][None]
                _prompt_attention_mask = prompt_attention_mask[_multi_prompts_indices[j_inside]][None].bool()
                if pooled_prompt_embeds is not None:
                    _pooled_prompt_embeds = pooled_prompt_embeds[_multi_prompts_indices[j_inside]][None]
                else:
                    _pooled_prompt_embeds = None

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=_prompt_embeds,
                    encoder_attention_mask=_prompt_attention_mask,
                    pooled_projections=_pooled_prompt_embeds,
                    guidance=guidance,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                self.scheduler._step_index = None  # ! important
                # latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                _latents = self.scheduler.step(noise_pred, t, _latents, return_dict=False)[0]

                _latents = rearrange(_latents, 'b c f (h ph) (w pw) -> b (c ph pw) f h w', ph=2, pw=2)  # ! hunyuan
                _latents = rearrange(_latents, 'b c f h w -> b c f (h w)')

                for idx_b in range(batch_size):
                    latents_next[idx_b, ..., indices_new] += _latents[idx_b] * weight
                    latents_next_cnt[idx_b, ..., indices_new] += weight

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

        wb = torch.zeros((batch_size, 3, num_frames, erp_height, erp_width), device=device, dtype=torch.float)
        wb_cnt = torch.zeros_like(wb)

        with self.progress_bar(total=num_inference_steps_view_dir) as progress_bar:  # (added) ERP
            for j_inside in range(num_inference_steps_view_dir):
                if not selected_j_inside(j_inside):
                    progress_bar.update()
                    continue

                cur_view_dir = view_dir[j_inside].repeat(batch_size, 1)  # (B, 3)
                fov_vae = fovs_main[j_inside]

                ### Dynamic Latent Sampling ###
                indices_new, weight = SphericalFunctions.dynamic_latent_sampling(
                    spherical_points, cur_view_dir, num_points_on_sphere, _fov,
                    temperature=weighted_average_temperature, center_first=False,
                )
                cur_latent_height = round(indices_new.shape[-1]**0.5)

                _latents = latents[..., indices_new]  # (B, C, F, N)
                _latents = rearrange(_latents, 'b c f (h w) -> b c f h w', h=cur_latent_height)
                _latents = rearrange(_latents, 'b (c ph pw) f h w -> b c f (h ph) (w pw)', ph=2, pw=2)

                _latents = _latents.to(self.vae.dtype) / self.vae.config.scaling_factor
                video = self.vae.decode(_latents, return_dict=False)[0]

                wb, wb_cnt = SphericalFunctions.paste_perspective_to_erp_rectangle(
                    wb, video.to(wb.device, wb.dtype), cur_view_dir.to(wb.device, wb.dtype), fov=fov_vae,
                    add=True, interpolate=True, interpolation_mode='bilinear',
                    panorama_cnt=wb_cnt, return_cnt=True, temperature=weighted_average_temperature,
                )

                progress_bar.update()

        wb_cnt[wb_cnt == 0] = 1
        wb /= wb_cnt

        video = self.video_processor.postprocess_video(wb, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HunyuanVideoPipelineOutput(frames=video)
