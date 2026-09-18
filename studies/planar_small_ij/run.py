"""Run the unchanged I/J pipeline with only aligned patch/stride changes."""
import argparse
import copy
import json
import os
import subprocess
from dataclasses import asdict
from types import SimpleNamespace
from pathlib import Path

import torch
from diffpano.implied_endpoint_consensus import PlanarImpliedEndpointConsensusPipeline
from diffpano.initialization import set_random_seed
from diffpano.metadata import save_run_metadata
from diffpano.native_multidiffusion import prepare_native_backend
from diffpano.pipelines import build_view_denoiser
from diffpano.trajectory import conditioning_digest
from diffpano.vae_residual import identity_vae_preflight
from scripts.generate import _configure_denoiser
from scripts.paired_rgb_endpoint import digest, timed, save_result
from studies.planar_small_ij.common import FACTORS, read, write, validated_row


def check_runtime(runtime, old, backend):
    for key in ('model','scheduler','environment'):
        if key=='environment':
            assert all(runtime[key][k]==old[key][k] for k in ('python','torch','cuda'))
        else:
            assert runtime[key]==old[key], key
    # FLUX prepares its dynamic shift from local image token count.
    if backend!='flux':
        for key in ('scheduler_timesteps','scheduler_sigmas'):
            assert runtime[key]==old[key], key
    a,b=copy.deepcopy(runtime['backend_details']),copy.deepcopy(old['backend_details'])
    if a is not None and b is not None:
        for key in ('local_rgb_shape','scheduler_image_seq_len','scheduler_shift_mu'):
            a.pop(key,None); b.pop(key,None)
        assert a==b, 'backend_details'
    assert runtime['native_geometry']['local_rgb_shape']==[256,256]


def run(name, method):
    row,c=validated_row(name,method)
    folder=Path(row['output']); folder.mkdir(parents=True,exist_ok=False)
    write(folder/'spec.json',row)
    write(folder/'repository.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        status=subprocess.check_output(['git','status','--short'],text=True)))
    assert torch.cuda.get_device_name()=='NVIDIA A40'
    set_random_seed(c.experiment.seed)
    backend=build_view_denoiser(c); _configure_denoiser(c,backend)
    prepared=prepare_native_backend(c,backend)
    conditioning=backend.conditioning_for_prompt_indices(prepared,[8],batch_size=c.generation.batch_size)
    assert conditioning_digest(conditioning)==row['conditioning_sha256']
    assert backend.native_spatial_factor==FACTORS[name]
    save_run_metadata(str(folder/'runtime_preflight.json'),c,backend,SimpleNamespace(steps=[]),'')
    runtime=read(folder/'runtime_preflight.json'); old=read(row['baseline_metadata'])
    check_runtime(runtime,old,name)
    initial=torch.load(row['initial_native'],map_location='cpu',weights_only=True)
    initial_hash=digest(initial)
    n=c.native_multidiffusion
    pipe=PlanarImpliedEndpointConsensusPipeline(native_config=n,backend=backend,
        residual_correction=method=='I',roundtrip_diagnostics=method=='I',
        fusion_config=c.fusion,transition_mode=c.consensus_transition.mode,flow_transition=name!='sd2')
    assert json.loads(json.dumps(asdict(pipe.native_layout)))==row['native_geometry']
    shape=(initial.shape[0],backend.native_channels,n.patch_size,n.patch_size)
    with torch.no_grad():
        sample=initial[:,:,:n.patch_size,:n.patch_size].to(backend.device)
        rgb=backend.decode_clean(sample); encoded=backend.encode_clean(rgb)
        assert tuple(rgb.shape[-2:])==(256,256) and tuple(encoded.shape)==shape
        assert torch.isfinite(rgb).all() and torch.isfinite(encoded).all()
        if method=='I':
            write(folder/'real_vae_identity_preflight.json',identity_vae_preflight(backend,shape))
        del sample,rgb,encoded
    local=pipe.initialize_local_states(initial.to(backend.device))
    del initial
    before=getattr(backend,'guided_prediction_count',0)
    print('Starting',name,method,'patches',pipe.native_layout.num_patches,'steps',len(backend.timesteps),flush=True)
    result,timing=timed(lambda:pipe.run(local,prepared))
    count=backend.guided_prediction_count-before
    assert count==row['expected_guided_predictions']==result.audit['guided_predictions']
    assert conditioning_digest(conditioning)==row['conditioning_sha256']
    result.audit.update(initial_local_states_equal_global_native_crops=True,
        consensus_transition=c.consensus_transition.mode,vae_residual_correction=method=='I',
        own_rgb_roundtrip_diagnostic=method=='I')
    write(folder/'transition_patches.json',result.transition_diagnostics)
    save_result(folder/'generation',c,backend,result,timing)
    final=read(folder/'generation/metadata.json')
    check_runtime(final,old,name)
    for key in ('scheduler_timesteps','scheduler_sigmas','native_geometry','backend_details'):
        assert runtime[key]==final[key], 'Runtime changed: '+key
    write(folder/'complete.json',dict(backend=name,method=method,job=os.environ.get('SLURM_JOB_ID'),
        initial_native_sha256=initial_hash,conditioning_sha256=row['conditioning_sha256'],
        guided_predictions=count,timing=timing,baseline_metadata=row['baseline_metadata'],
        schedule_changed_from_historical=runtime['scheduler_sigmas']!=old['scheduler_sigmas']))
    print('COMPLETE',name,method,folder,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--backend',required=True,choices=list(FACTORS)); p.add_argument('--method',required=True,choices=['I','J'])
    a=p.parse_args(); run(a.backend,a.method)
