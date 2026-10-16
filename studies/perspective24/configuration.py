"""Select local settings from saved audited FLUX20; no camera construction."""
from types import SimpleNamespace
from studies.perspective24.common import *

BASELINE=ROOT/'outputs/gwtf-erp4k/20260923/flux/metadata.json'
WEIGHTS=ROOT/'outputs/metrics-evaluation/seed0-v1/checkpoint_weight_audit.json'

def configuration():
    old=read(BASELINE);c=old['config']
    assert (c['model']['pipeline'],c['model']['id'],c['model']['revision'])==('flux','ModelsLab/flux.1-dev','fa45a9eb6808ba8fdfc7cc2756f7f1a16e0921f4')
    assert c['generation']['num_inference_steps']==20 and c['generation']['batch_size']==1
    assert c['view']['height']==c['view']['width']==1024 and c['experiment']['seed']==0
    assert c['generation']['guidance_scale']==3.5 and c['generation']['true_cfg_scale']==1.
    # Only local loader/conditioning settings are exposed to the native runner.
    keys=('model','generation','performance','debug','view')
    local={k:c[k] for k in keys}
    negative_path=c['prompt']['negative_path'];negative=''
    if negative_path:negative=(ROOT/negative_path).read_text().strip()
    expected=old['prepared_schedule']
    return local,expected,negative

def namespace(local):return SimpleNamespace(**{k:SimpleNamespace(**v) for k,v in local.items()})

def semantic(row):
    local,schedule,negative=configuration()
    return dict(case=dict(row),local_configuration=local,schedule=schedule,negative_prompt=negative,
        max_sequence_length=512,prompt_2=None,num_images_per_prompt=1,
        conditioning='encode exact one line with FluxPipeline.encode_prompt; no directional expansion',
        initialization=dict(method='independent standard native Gaussian',generator_device='cuda',dtype='torch.float32',seed=0,fresh_generator_per_image=True,scaling_applications=1),
        native_resolution=[128,128],native_channels=16,native_arithmetic_dtype='torch.float32',
        sampler='FluxViewDenoiser.denoise_native_step',terminal_decode='FluxViewDenoiser.decode_native_canvas',
        schedule_reset='FluxViewDenoiser.prepare for each image; fresh scheduler from immutable config',
        output_conversion='[-1,1] RGB clamp then [0,255] rounded PNG using tensor_to_pil',
        panorama_processing=False,bridge=False,vae_image_encode=False,terminal_decode_only=True,
        weight_audit_sha256=sha(WEIGHTS),baseline_metadata=str(BASELINE),baseline_metadata_sha256=sha(BASELINE))
