"""Ordinary single-image FLUX native sampling; no panorama pipeline is invoked."""
import contextlib,time
import numpy as np
import torch
from studies.perspective24.common import *
from studies.perspective24.configuration import configuration,namespace

def tensor_hash(value):
    value=value.detach().cpu().contiguous()
    h=hashlib.sha256(str((tuple(value.shape),str(value.dtype))).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes());return h.hexdigest()

def conditioning_hash(conditioning):
    return digest({k:tensor_hash(v) for k,v in sorted(conditioning.items())})

def schedule_record(b):
    c=dict(b.pipeline.scheduler.config)
    c={k:sorted(v) if isinstance(v,set) else v for k,v in c.items()}
    if '_use_default_values' in c:c['_use_default_values']=sorted(c['_use_default_values'])
    return dict(class_name=type(b.pipeline.scheduler).__name__,config=c,
        timesteps=b.timesteps.detach().cpu().tolist(),sigmas=b.pipeline.scheduler.sigmas.detach().cpu().tolist())

def reset_schedule(b,row,expected):
    # Reconstruct mutable scheduler state, preserving the audited class/config.
    previous=b.pipeline.scheduler
    b.pipeline.scheduler=type(previous).from_config(previous.config)
    b.prepare(num_steps=row['steps'],view_height=row['height'],view_width=row['width'])
    current=schedule_record(b)
    assert current==expected,(current,expected)
    assert len(b.timesteps)==20 and b.pipeline.scheduler.step_index is None
    return current

@contextlib.contextmanager
def count_calls(b):
    counters=dict(denoiser=0,encode=0,decode=0,initialize=0);saved=[]
    def wrap(obj,name,key):
        original=getattr(obj,name)
        def call(*args,**kwargs):
            counters[key]+=1
            if key=='encode':raise AssertionError('VAE image encoding is forbidden in standalone text-to-image')
            return original(*args,**kwargs)
        setattr(obj,name,call);saved.append((obj,name,original))
    wrap(b.pipeline.transformer,'forward','denoiser');wrap(b.pipeline.vae,'encode','encode')
    wrap(b.pipeline.vae,'decode','decode');wrap(b,'initialize_native_state','initialize')
    try:yield counters
    finally:
        for obj,name,original in reversed(saved):setattr(obj,name,original)

@torch.no_grad()
def encode_exact_lines(b,rows,negative):
    """Encode every assigned line before releasing one-shot encoders once."""
    from diffpano.pipelines.base import release_prompt_encoders
    assert b.pipeline.text_encoder is not None and b.pipeline.text_encoder_2 is not None
    result={}
    for row in rows:
        assert hashlib.sha256(row['line_text'].encode('utf-8')).hexdigest()==row['line_sha256']
        positive,pooled,ids=b.pipeline.encode_prompt(prompt=row['line_text'],prompt_2=None,device=b.device,
            num_images_per_prompt=1,max_sequence_length=512)
        value=dict(embeds=positive,pooled=pooled,text_ids=ids)
        if b.true_cfg_scale>1.0 and negative:
            neg,npooled,nids=b.pipeline.encode_prompt(prompt=negative,prompt_2=None,device=b.device,num_images_per_prompt=1,max_sequence_length=512)
            value.update(negative=neg,negative_pooled=npooled,negative_text_ids=nids)
        assert value['embeds'].shape[0]==value['pooled'].shape[0]==1
        result[row['index']]={k:v.detach().cpu().contiguous() for k,v in value.items()}
    release_prompt_encoders(b.pipeline)
    assert b.pipeline.text_encoder is None and b.pipeline.text_encoder_2 is None
    return result

@torch.no_grad()
def load_backend(rows):
    from diffpano.pipelines import build_view_denoiser
    from scripts.generate import _configure_denoiser
    from diffpano.initialization import set_random_seed
    import diffusers,transformers
    local,expected,negative=configuration();c=namespace(local);start=time.perf_counter()
    set_random_seed(0);b=build_view_denoiser(c);_configure_denoiser(c,b)
    encoded=encode_exact_lines(b,rows,negative)
    return b,encoded,dict(seconds=time.perf_counter()-start,encoded_cases=len(rows),encoder_lifecycle='encode exact assigned lines at batch one, cache on CPU, release once',
        model_checkpoint=c.model.path or c.model.id,model_revision=c.model.revision,precision=c.model.precision,
        guidance_scale=c.generation.guidance_scale,true_cfg_scale=c.generation.true_cfg_scale,negative_prompt=negative,
        max_sequence_length=512,vae_tiling=c.model.vae_tiling,vae_slicing=c.model.vae_slicing,cpu_offload=c.model.cpu_offload,
        adapter_class=type(b).__module__+'.'+type(b).__name__,pipeline_class=type(b.pipeline).__module__+'.'+type(b.pipeline).__name__,
        transformer_dtype=str(b.dtype),vae_dtype=str(b.pipeline.vae.dtype),versions=dict(torch=torch.__version__,diffusers=diffusers.__version__,transformers=transformers.__version__))

@torch.no_grad()
def sample(b,row,conditioning,expected,progress=None):
    """One fresh Gaussian, twenty native steps, one terminal decode."""
    schedule=reset_schedule(b,row,expected)
    generator=torch.Generator(device=b.device).manual_seed(row['seed'])
    before=getattr(b,'guided_prediction_count',0)
    with count_calls(b) as counts:
        state=b.sample_initial_native_state(batch_size=1,native_height=128,native_width=128,generator=generator)
        initial=dict(method='independent standard native Gaussian',seed=row['seed'],generator_device=str(generator.device),
            fresh_generator_per_image=True,initial_native_sha256=tensor_hash(state),native_shape=list(state.shape),dtype=str(state.dtype),native_scale=float(b.native_initial_noise_sigma))
        assert b.native_initial_noise_sigma==1.
        initial['initial_noise_sha256']=initial['initial_native_sha256']
        assert state.shape==(1,16,128,128) and state.dtype==torch.float32 and bool(torch.isfinite(state).all())
        for index,timestep in enumerate(b.timesteps):
            state=b.denoise_native_step(state,timestep,conditioning)
            assert bool(torch.isfinite(state).all()),index
            if progress:progress(index+1,len(b.timesteps))
        rgb=b.decode_native_canvas(state)
    guided=getattr(b,'guided_prediction_count',0)-before
    assert counts==COUNTS and guided==row['steps']==20,(counts,guided)
    assert rgb.shape==(1,3,1024,1024) and bool(torch.isfinite(rgb).all())
    return rgb,dict(counts=counts,guided_steps=guided,native_intervals=20,initialization=initial,schedule=schedule,schedule_sha256=digest(schedule))

@torch.no_grad()
def generate(b,row,conditioning,directory,setup):
    from diffpano.diagnostics import tensor_to_pil
    condition={k:v.to(b.device) for k,v in conditioning.items()}
    local,expected,negative=configuration()
    config=dict(**row['semantic'],conditioning_sha256=conditioning_hash(condition),line_text=row['line_text'],line_sha256=row['line_sha256'])
    write(directory/'config.json',config)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    def progress(step,total):write(directory/'progress.json',dict(step=step,total=total,elapsed_seconds=time.perf_counter()-start))
    rgb,record=sample(b,row,condition,expected,progress)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    staging=directory/('final.tmp-'+os.environ['SLURM_JOB_ID']+'.png')
    tensor_to_pil(rgb[0].detach().cpu()).save(staging);os.replace(str(staging),str(directory/'final.png'))
    return dict(**record,line_text=row['line_text'],line_index=row['line_index'],line_sha256=row['line_sha256'],
        conditioning_sha256=config['conditioning_sha256'],model=setup,generation_seconds=elapsed,shared_setup_seconds=setup['seconds'],
        generation_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,gpu=torch.cuda.get_device_name(),
        raw_decoder_statistics=dict(minimum=float(rgb.min()),maximum=float(rgb.max()),out_of_range_fraction=float(((rgb<-1)|(rgb>1)).float().mean())),
        preprocessing=dict(prompt='exact unmodified line',prompt_2=None,max_sequence_length=512,image='none',output='tensor_to_pil only'),
        sampler='FluxViewDenoiser.denoise_native_step',decoder='FluxViewDenoiser.decode_native_canvas',
        layout=None,camera_count=None,panorama_trajectory=False,warp_calls=0,fusion_stages=0,poisson_solves=0,
        artifacts={name:sha(directory/name) for name in ('config.json','final.png')})
