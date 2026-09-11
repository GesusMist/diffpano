#!/usr/bin/env python3
"""Fair native MultiDiffusion/RGB endpoint pairs plus latent roundtrip controls."""
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from diffpano.config import load_experiment_config
from diffpano.diagnostics import tensor_to_pil
from diffpano.implied_endpoint_consensus import PlanarImpliedEndpointConsensusPipeline
from diffpano.initialization import load_directional_prompts, set_random_seed
from diffpano.metadata import save_run_metadata
from diffpano.native_multidiffusion import NativeMultiDiffusionPipeline, prepare_native_backend
from diffpano.pipelines import build_view_denoiser
from diffpano.pipelines.base import reset_scheduler_step_state
from diffpano.planar_pipeline import _negative_prompt
from diffpano.trajectory import conditioning_digest, difference_stats
from scripts.generate import _configure_denoiser, _run_directory


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def timed(operation):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    value=operation()
    torch.cuda.synchronize()
    return value,dict(runtime_seconds=time.perf_counter()-started,
        peak_gpu_memory_gib=dict(allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                                 reserved_gib=torch.cuda.max_memory_reserved()/1024**3))


def save_result(folder, config, backend, result, timing):
    folder.mkdir()
    tensor_to_pil(result.canvas_rgb[0]).save(folder/'result.png')
    if getattr(result,'fused_clean_rgb',None) is not None:
        tensor_to_pil(result.fused_clean_rgb[0]).save(folder/'final_fused_clean.png')
    for key,value in timing.items(): setattr(result,key,value)
    save_run_metadata(str(folder/'metadata.json'),config,backend,result,str(folder/'result.png'))
    steps=[dict(step=s.step_index,timestep=s.scheduler_timestep,**s.state_statistics) for s in result.steps]
    fields=list(dict.fromkeys(k for s in steps for k in s))
    with (folder/'steps.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(steps)


@torch.no_grad()
def no_roundtrip(backend, initial, conditioning):
    state=initial.clone()
    scheduler=getattr(getattr(backend,'pipeline',None),'scheduler',None)
    before=getattr(backend,'guided_prediction_count',0)
    for t in backend.timesteps:
        if scheduler is not None: reset_scheduler_step_state(scheduler)
        state=backend.predict_clean_and_endpoint(state,t,conditioning).reconstruct_next()
    if backend.guided_prediction_count-before != len(backend.timesteps):
        raise AssertionError('No-roundtrip control evaluation mismatch')
    return backend.decode_native_canvas(state)


def paired_config_check(config):
    native=load_experiment_config('configs/experiments/native_multidiffusion/'+config.model.pipeline+'.yaml')
    a,b=native.to_dict(),config.to_dict()
    if native.experiment.seed != config.experiment.seed:
        raise AssertionError('Seed differs from native reference')
    for key in ('experiment','output','global_pipeline'):
        del a[key]; del b[key]
    if a != b or config.global_pipeline.mode != 'implied_endpoint_consensus':
        raise AssertionError('Paired configuration differs from native reference settings')
    return native


def run(config):
    config.validate(); native_config=paired_config_check(config)
    snapshot=config.to_dict()
    set_random_seed(config.experiment.seed)
    destination=_run_directory(config);destination.mkdir(parents=True,exist_ok=False)
    OmegaConf.save(OmegaConf.create(snapshot),destination/'config.yaml')
    started=time.perf_counter()
    backend=build_view_denoiser(config);_configure_denoiser(config,backend)
    prepared=prepare_native_backend(config,backend)
    setup_seconds=time.perf_counter()-started
    conditioning=backend.conditioning_for_prompt_indices(prepared,[8],batch_size=config.generation.batch_size)
    condition_hash=conditioning_digest(conditioning)
    schedule=backend.timesteps.clone()
    scheduler=getattr(getattr(backend,'pipeline',None),'scheduler',None)
    coefficients=[]
    for obj,names in ((scheduler,('sigmas','alphas_cumprod')), (getattr(backend,'solver',None),('schedule',))):
        for name in names:
            value=getattr(obj,name,None)
            if isinstance(value,torch.Tensor): coefficients.append((obj,name,value.clone()))
    n=config.native_multidiffusion
    initial=backend.sample_initial_native_state(batch_size=config.generation.batch_size,
        native_height=n.canvas_height,native_width=n.canvas_width,
        generator=torch.Generator(device=backend.device).manual_seed(config.experiment.seed))
    initial_hash=digest(initial);torch.save(initial.cpu(),destination/'initial_native.pt')
    pipeline=PlanarImpliedEndpointConsensusPipeline(native_config=n,backend=backend,pixel_native=config.model.pipeline=='pixeldit')
    locals_=pipeline.initialize_local_states(initial)
    geometry=dict(native=asdict(pipeline.native_layout),rgb=asdict(pipeline.rgb_layout),factor=backend.native_spatial_factor)
    initial_shape=list(initial.shape)
    before=getattr(backend,'guided_prediction_count',0)
    native,nt=timed(lambda: NativeMultiDiffusionPipeline(native_config=n,backend=backend,overlap_disagreement=True).run(initial,prepared))
    native_count=backend.guided_prediction_count-before
    if digest(initial)!=initial_hash: raise AssertionError('Native reference mutated initialization')
    save_result(destination/'native_multidiffusion',native_config,backend,native,nt)
    native_image=native.canvas_rgb.cpu()
    del initial,native
    consensus,ct=timed(lambda: pipeline.run(locals_,prepared))
    expected=len(schedule)*pipeline.native_layout.num_patches
    if native_count!=expected or consensus.audit['guided_predictions']!=expected:
        raise AssertionError('Paired model evaluation counts differ')
    consensus.audit['initial_local_states_equal_global_native_crops']=True
    save_result(destination/'implied_endpoint_consensus',config,backend,consensus,ct)
    pair_difference=difference_stats(native_image,consensus.canvas_rgb.cpu(),'final_rgb_native_vs_consensus')
    del consensus,locals_,native_image
    controls=None
    if config.model.pipeline!='pixeldit':
        # Same CPU Gaussian convention as the validated single-patch controls.
        p=n.patch_size
        epsilon=torch.randn(config.generation.batch_size,backend.native_channels,p,p,
            generator=torch.Generator(device='cpu').manual_seed(config.experiment.seed)).to(backend.device)
        one_initial=backend.initialize_native_state(epsilon)
        folder=destination/'one_patch_roundtrip';folder.mkdir()
        torch.save(epsilon.cpu(),folder/'initial_epsilon.pt')
        without,wt=timed(lambda: no_roundtrip(backend,one_initial,conditioning))
        tensor_to_pil(without[0]).save(folder/'no_roundtrip.png')
        geometry_one=copy.deepcopy(n);geometry_one.canvas_height=p;geometry_one.canvas_width=p
        onepipe=PlanarImpliedEndpointConsensusPipeline(native_config=geometry_one,backend=backend)
        withrt,rt=timed(lambda: onepipe.run([one_initial.clone()],prepared))
        tensor_to_pil(withrt.canvas_rgb[0]).save(folder/'roundtrip.png')
        tensor_to_pil(withrt.fused_clean_rgb[0]).save(folder/'roundtrip_final_fused_clean.png')
        controls=dict(no_roundtrip=wt,roundtrip=rt,epsilon_sha256=digest(epsilon),
            same_initial_state=True,model_evaluations=dict(no_roundtrip=len(schedule),roundtrip=withrt.audit['guided_predictions']),
            differences=difference_stats(without,withrt.canvas_rgb,'final_rgb'),
            steps=[asdict(s) for s in withrt.steps])
        (folder/'control.json').write_text(json.dumps(controls,indent=2)+'\n')
    if config.to_dict()!=snapshot or condition_hash!=conditioning_digest(conditioning) or not torch.equal(schedule,backend.timesteps):
        raise AssertionError('Shared configuration, conditioning, or schedule changed')
    for obj,name,value in coefficients:
        if not torch.equal(getattr(obj,name).to(value.device),value):
            raise AssertionError('Scheduler coefficients changed')
    data=dict(schema_version=1,gpu_name=torch.cuda.get_device_name(),gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/1024**3,slurm_job_id=os.environ.get('SLURM_JOB_ID'),node=platform.node(),
        config=snapshot,geometry=geometry,initial_native_sha256=initial_hash,initial_native_shape=initial_shape,
        initial_generator_device=str(backend.device),initial_noise_sigma=backend.native_initial_noise_sigma,
        fairness=dict(same_initial_global_field=True,local_crops_bit_identical=True,same_conditioning=True,
            conditioning_sha256=condition_hash,same_checkpoint_precision_guidance_geometry=True,same_schedule=True),
        prompt=load_directional_prompts(config.prompt.path)[2],negative_prompt=_negative_prompt(config) or getattr(backend,'negative_prompt',''),
        actual_precision=str(backend.dtype),setup_seconds=setup_seconds,total_seconds=time.perf_counter()-started,
        methods=dict(native_multidiffusion=nt,implied_endpoint_consensus=ct),
        model_evaluations=dict(native_multidiffusion=native_count,implied_endpoint_consensus=expected),
        diagnostic_extra_model_evaluations=0,final_rgb_differences=pair_difference,one_patch_control=controls is not None)
    (destination/'comparison.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(destination=str(destination),counts=data['model_evaluations']),indent=2),flush=True)
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    run(load_experiment_config(parser.parse_args().config))
