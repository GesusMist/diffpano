#!/usr/bin/env python3
"""Cache official SD3.5 Medium, then verify its unmodified Diffusers pipeline."""
import argparse
import json
import os
import platform
import time
from pathlib import Path

MODEL = 'stabilityai/stable-diffusion-3.5-medium'


def existing_token():
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    for root in (Path(os.environ.get('HF_HOME', '/scratch/user/shig/diffpano/hf_cache')),
                 Path.home()/'.cache/huggingface'):
        if not token and (root/'token').is_file():
            token = (root/'token').read_text().strip()
    return token


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-only',action='store_true')
    args=parser.parse_args()
    from huggingface_hub import HfApi,snapshot_download
    token=existing_token()
    index=Path('outputs/native-controls/sd35-checkpoint.json')
    if args.download_only:
        info=HfApi().model_info(MODEL,token=token)
        snapshot=snapshot_download(MODEL,revision=info.sha,token=token,
            allow_patterns=['model_index.json','scheduler/*','transformer/*','vae/*',
                            'text_encoder/*','text_encoder_2/*','text_encoder_3/*',
                            'tokenizer/*','tokenizer_2/*','tokenizer_3/*'],
            ignore_patterns=['*.bin','*.onnx','*.msgpack'],max_workers=4)
        index.parent.mkdir(parents=True,exist_ok=True)
        index.write_text(json.dumps(dict(model=MODEL,revision=info.sha,snapshot=snapshot),indent=2)+'\n')
        print(index.read_text(),flush=True)
        return
    import torch
    import diffusers
    from diffusers import StableDiffusion3Pipeline
    checkpoint=json.loads(index.read_text())
    folder=Path('outputs/native-controls/sd35-official-sanity')/os.environ.get('SLURM_JOB_ID','local')
    folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    pipe=StableDiffusion3Pipeline.from_pretrained(checkpoint['snapshot'],torch_dtype=torch.bfloat16,local_files_only=True)
    from collections import Counter
    def parameter_dtypes():
        return {name:dict(Counter(str(p.dtype) for p in getattr(pipe,name).parameters()))
                for name in ('text_encoder','text_encoder_2','text_encoder_3','transformer','vae')}
    before_dtypes=parameter_dtypes()
    pipe.to('cuda',dtype=torch.bfloat16)
    after_dtypes=parameter_dtypes()
    (folder/'loaded_dtypes.json').write_text(json.dumps(dict(before=before_dtypes,after=after_dtypes),indent=2)+'\n')
    # Use the same non-tiled local VAE for all future SD3.5 controls.
    from diffpano.initialization import load_directional_prompts
    prompt=load_directional_prompts('prompts/native_control.txt')[2]
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();sample_started=time.perf_counter()
    image=pipe(prompt,negative_prompt='',height=1024,width=1024,num_inference_steps=40,
        guidance_scale=4.5,max_sequence_length=256,
        generator=torch.Generator(device='cpu').manual_seed(0)).images[0]
    torch.cuda.synchronize();sampling=time.perf_counter()-sample_started
    image.save(folder/'result.png')
    data=dict(checkpoint=checkpoint,experiment='official sanity (prerequisite to A)',
        prompt=prompt,negative_prompt='',seed=0,resolution=[1024,1024],steps=40,guidance=4.5,
        precision='bf16',skip_layer_guidance=False,packed_native_state=False,
        scheduler_class=type(pipe.scheduler).__name__,scheduler_config=dict(pipe.scheduler.config),
        scheduler_timesteps=pipe.scheduler.timesteps.cpu().tolist(),scheduler_sigmas=pipe.scheduler.sigmas.cpu().tolist(),
        vae_class=type(pipe.vae).__name__,vae_config=dict(pipe.vae.config),
        transformer_config=dict(pipe.transformer.config),spatial_factor=pipe.vae_scale_factor,
        sampling_seconds=sampling,total_seconds=time.perf_counter()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        job=os.environ.get('SLURM_JOB_ID'),node=platform.node(),gpu=torch.cuda.get_device_name(),
        diffusers=diffusers.__version__,torch=str(torch.__version__),parameter_dtypes=after_dtypes)
    (folder/'metadata.json').write_text(json.dumps(data,indent=2)+'\n')
    print('Official sanity complete:',folder,flush=True)


if __name__=='__main__':main()
