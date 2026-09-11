"""Stable Diffusion 3.5: official SD3 components, raw BCHW flow state."""
import time
from dataclasses import dataclass

import torch

from diffpano.conditioning import camera_prompt_indices, expand_directional_prompts, expanded_prompt_indices
from diffpano.pipelines.base import ViewDenoiser, ensure_first_order_scheduler, release_prompt_encoders, reset_scheduler_step_state
from diffpano.pipelines.native_state import NativeStateMixin
from diffpano.pipelines.clean_prediction import flow_add_noise, flow_predicted_clean
from diffpano.pipelines.endpoints import flow_bounds, flow_endpoints
from diffpano.vae import encode_view_images, decode_view_latents


@dataclass
class SD35PromptBank:
    prompt_directions: torch.Tensor
    positive: torch.Tensor
    pooled: torch.Tensor
    negative: torch.Tensor = None
    negative_pooled: torch.Tensor = None


class SD35ViewDenoiser(NativeStateMixin, ViewDenoiser):
    """Native state and guided velocity are float32, as in SD2/SANA adapters.

    The transformer and VAE retain the configured model dtype. Both ordinary
    scheduler integration and endpoint reconstruction receive the same float32
    CFG prediction. No packed state, SLG, guidance embedding or extra forward.
    """
    def __init__(self, pipeline, *, guidance_scale, vae_chunk_size=1, measure_performance=False):
        self.pipeline=pipeline
        self.guidance_scale=guidance_scale
        self.vae_chunk_size=vae_chunk_size
        self.measure_performance=measure_performance
        self.last_timings={}
        self._timesteps=torch.empty(0)
        self._view_size=None

    @classmethod
    def from_pretrained(cls, source, *, guidance_scale, vae_chunk_size=1, measure_performance=False, **kwargs):
        from diffusers import StableDiffusion3Pipeline
        return cls(StableDiffusion3Pipeline.from_pretrained(source, **kwargs),
                   guidance_scale=guidance_scale, vae_chunk_size=vae_chunk_size,
                   measure_performance=measure_performance)

    @property
    def device(self):return torch.device(self.pipeline._execution_device)
    @property
    def dtype(self):return self.pipeline.transformer.dtype
    @property
    def timesteps(self):return self._timesteps
    @property
    def native_channels(self):return int(self.pipeline.transformer.config.in_channels)
    @property
    def native_spatial_factor(self):return int(self.pipeline.vae_scale_factor)
    @property
    def native_initial_noise_sigma(self):return 1.0

    def to(self,*args,**kwargs):
        self.pipeline.to(*args,**kwargs)
        return self

    def enable_model_cpu_offload(self):self.pipeline.enable_model_cpu_offload()

    def _timed(self,name,operation):
        if self.measure_performance and self.device.type=='cuda':torch.cuda.synchronize(self.device)
        started=time.perf_counter();value=operation()
        if self.measure_performance and self.device.type=='cuda':torch.cuda.synchronize(self.device)
        self.last_timings[name]=self.last_timings.get(name,0.)+time.perf_counter()-started
        return value

    def prepare(self, *, num_steps, view_height, view_width):
        from diffusers import FlowMatchEulerDiscreteScheduler
        scheduler=self.pipeline.scheduler
        if type(scheduler) is not FlowMatchEulerDiscreteScheduler:
            raise ValueError('SD3.5 controls require the checkpoint FlowMatchEulerDiscreteScheduler')
        ensure_first_order_scheduler(scheduler)
        factor=self.native_spatial_factor*int(self.pipeline.transformer.config.patch_size)
        if min(view_height,view_width)<factor or view_height%factor or view_width%factor:
            raise ValueError('SD3.5 resolution must align with VAE and transformer patch factors')
        if self.native_channels!=int(self.pipeline.vae.config.latent_channels):
            raise ValueError('SD3.5 transformer and VAE raw latent channels differ')
        self._view_size=(view_height,view_width)
        self.scheduler_image_seq_len=(view_height//factor)*(view_width//factor)
        self.scheduler_shift_mu=None
        kwargs={}
        if scheduler.config.use_dynamic_shifting:
            # Match the installed SD3 pipeline's dynamic branch if a checkpoint uses it.
            from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift
            cfg=scheduler.config
            self.scheduler_shift_mu=calculate_shift(self.scheduler_image_seq_len,cfg.base_image_seq_len,
                cfg.max_image_seq_len,cfg.base_shift,cfg.max_shift)
            kwargs['mu']=self.scheduler_shift_mu
        scheduler.set_timesteps(num_steps,device=self.device,**kwargs)
        self._timesteps=scheduler.timesteps
        self.pipeline._guidance_scale=self.guidance_scale
        self.pipeline._clip_skip=None
        self.pipeline._joint_attention_kwargs=None

    @torch.no_grad()
    def prepare_prompt_conditioning(self,prompts,negative_prompt=''):
        directional=expand_directional_prompts(prompts)
        unique=list(dict.fromkeys(directional.prompts))
        values=[]
        # Encode each unique prompt once; global control uses one, not 20 T5 copies.
        for prompt in unique:
            values.append(self.pipeline.encode_prompt(prompt=prompt,prompt_2=None,prompt_3=None,
                negative_prompt=negative_prompt,do_classifier_free_guidance=self.guidance_scale>1,
                device=self.device,num_images_per_prompt=1,clip_skip=None,max_sequence_length=256))
        indices=torch.tensor([unique.index(p) for p in directional.prompts],device=self.device)
        def stack(index):
            return None if values[0][index] is None else torch.cat([v[index] for v in values])[indices]
        bank=SD35PromptBank(directional.directions,stack(0),stack(2),stack(1),stack(3))
        self.pipeline.text_encoder_3=None
        release_prompt_encoders(self.pipeline)
        return bank

    def conditioning_for_prompt_indices(self,bank,prompt_indices,*,batch_size):
        indices=expanded_prompt_indices(prompt_indices,batch_size=batch_size,
            num_prompts=bank.positive.shape[0],device=self.device)
        embeds,pooled=bank.positive[indices],bank.pooled[indices]
        if bank.negative is not None:
            embeds=torch.cat([bank.negative[indices],embeds])
            pooled=torch.cat([bank.negative_pooled[indices],pooled])
        return dict(embeds=embeds,pooled=pooled)

    def conditioning_for_cameras(self,bank,cameras,*,batch_size):
        return self.conditioning_for_prompt_indices(bank,camera_prompt_indices(cameras,bank.prompt_directions),batch_size=batch_size)

    def _predict_flow(self,state,timestep,conditioning):
        if state.ndim!=4 or state.shape[1]!=self.native_channels:
            raise ValueError('SD3.5 expects raw BCHW native latents')
        if self.rgb_spatial_shape_for_native(*state.shape[-2:])!=self._view_size:
            raise ValueError('SD3.5 native patch differs from the prepared model resolution')
        self.record_guided_prediction()
        model_input=torch.cat([state,state]) if self.guidance_scale>1 else state
        # SD3 passes the actual scheduler timestep, not FLUX's t/1000.
        time_batch=torch.as_tensor(timestep,device=self.device).expand(model_input.shape[0])
        prediction=self._timed('model_forward',lambda:self.pipeline.transformer(
            hidden_states=model_input.to(self.dtype),timestep=time_batch,
            encoder_hidden_states=conditioning['embeds'],pooled_projections=conditioning['pooled'],
            joint_attention_kwargs=None,return_dict=False)[0]).float()
        if self.guidance_scale>1:
            negative,positive=prediction.chunk(2)
            prediction=negative+self.guidance_scale*(positive-negative)
        return prediction

    @torch.no_grad()
    def denoise_native_step(self,state,timestep,conditioning):
        self.last_timings={}
        prediction=self._predict_flow(state,timestep,conditioning)
        self.last_model_prediction=prediction.detach()
        reset_scheduler_step_state(self.pipeline.scheduler)
        return self.pipeline.scheduler.step(prediction,timestep,state.float(),return_dict=False)[0].float()

    def predict_clean_native(self,state,timestep,conditioning):
        self.last_timings={}
        self.last_model_prediction=self._predict_flow(state,timestep,conditioning).detach()
        return flow_predicted_clean(self.pipeline.scheduler,state,self.last_model_prediction,timestep)

    def _endpoints_from_last_prediction(self,state,timestep,clean=None):
        return flow_endpoints(state,self.last_model_prediction,*flow_bounds(self.pipeline.scheduler,timestep,state),clean=clean)

    def encode_clean(self,rgb):return encode_view_images(self.pipeline.vae,rgb.float(),chunk_size=self.vae_chunk_size)
    def decode_clean(self,state):return decode_view_latents(self.pipeline.vae,state.float(),chunk_size=self.vae_chunk_size).float()

    def sample_fixed_noise(self,*,batch_size,height,width,generator):
        h,w=self.native_spatial_shape_for_rgb(height,width)
        return torch.randn(batch_size,self.native_channels,h,w,generator=generator,device=generator.device,dtype=torch.float32)

    def make_initial_noisy_state(self,fixed_noise,timestep):return self.initialize_native_state(fixed_noise)
    def add_fixed_noise(self,clean_state,fixed_noise,timestep):return flow_add_noise(self.pipeline.scheduler,clean_state,fixed_noise,timestep)

    @torch.no_grad()
    def denoise_step(self,rgb_view,timestep,conditioning):
        return self.decode_clean(self.denoise_native_step(self.encode_clean(rgb_view),timestep,conditioning))

    @torch.no_grad()
    def sample_native_rgb(self,*,batch_size,height,width,generator):
        return self.decode_clean(self.sample_fixed_noise(batch_size=batch_size,height=height,width=width,generator=generator).to(self.device))

    def backend_metadata(self):
        import diffusers
        return dict(pipeline_class=type(self.pipeline).__name__,transformer_class=type(self.pipeline.transformer).__name__,
            transformer_config=dict(self.pipeline.transformer.config),vae_class=type(self.pipeline.vae).__name__,
            vae_config=dict(self.pipeline.vae.config),scheduler_config={k:sorted(v) if k=="_use_default_values" else v for k,v in self.pipeline.scheduler.config.items()},
            prediction_semantics='flow velocity v=x1-x0; scheduler step x+(sigma_next-sigma)*v',
            prediction_type_config=self.pipeline.scheduler.config.get('prediction_type',None),
            network_dtype=str(self.dtype),vae_dtype=str(self.pipeline.vae.dtype),native_state_dtype='torch.float32',
            guided_prediction_dtype='torch.float32',cfg_arithmetic='float32 after network prediction',
            native_scheduler_precision='float32 prediction and state; ordinary unchanged scheduler.step',
            packed_native_state=False,guidance_embedding=False,skip_layer_guidance=False,
            clip_skip=None,max_sequence_length=256,native_channels=self.native_channels,
            native_spatial_factor=self.native_spatial_factor,local_rgb_shape=self._view_size,
            scheduler_shift_mu=self.scheduler_shift_mu,scheduler_image_seq_len=self.scheduler_image_seq_len,
            diffusers_version=diffusers.__version__)
