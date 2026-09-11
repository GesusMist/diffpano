#!/usr/bin/env python3
"""Run one validated A-G stage using the existing native/endpoint/residual loops."""
import argparse
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch

from diffpano.config import load_experiment_config
from diffpano.diagnostics import tensor_to_pil
from diffpano.initialization import load_directional_prompts, set_random_seed
from diffpano.implied_endpoint_consensus import PlanarImpliedEndpointConsensusPipeline
from diffpano.native_multidiffusion import NativeMultiDiffusionPipeline, prepare_native_backend
from diffpano.pipelines import build_view_denoiser
from diffpano.trajectory import conditioning_digest, difference_stats
from diffpano.vae_residual import identity_vae_preflight, single_patch_residual_trajectory
from scripts.generate import _configure_denoiser
from scripts.paired_rgb_endpoint import digest, timed, save_result
from scripts.vae_residual_experiment import write_json, save_steps


def read(path):return json.loads(Path(path).read_text())
def control_path(protocol,phase):return Path(protocol['folders'][phase])/(phase+'_control.json')


def check_configs(protocol):
    configs={phase:load_experiment_config(path) for phase,path in protocol['configs'].items()}
    if set(configs)!=set('ABCDEFG'):raise ValueError('A-G configs are required')
    for labels in ('ABCD','EFG'):
        snapshots=[]
        for phase in labels:
            snapshot=configs[phase].to_dict()
            for key in ('experiment','output','global_pipeline'):del snapshot[key]
            snapshots.append(snapshot)
        if any(s!=snapshots[0] for s in snapshots):raise AssertionError('Paired settings differ')
    for phase,c in configs.items():
        if c.model.pipeline!=protocol['backend']:raise AssertionError('Backend mismatch')
        if c.global_pipeline.mode!=('implied_endpoint_consensus' if phase in 'FG' else 'native_multidiffusion'):
            raise ValueError('Wrong control mode')
        if c.performance.view_batch_size!=1:raise ValueError('Control ladder requires individual patches')
    return configs


def prerequisites(protocol,phase):
    if not (Path(protocol['official_sanity'])/'metadata.json').is_file():
        raise AssertionError('Official sanity run must finish before A-G')
    previous={'A':None,'B':'A','C':'B','D':'C','E':'D','F':'E','G':'F'}[phase]
    if previous is not None and not control_path(protocol,previous).is_file():
        raise AssertionError('Required preceding control has not completed: '+previous)
    if control_path(protocol,phase).exists():raise FileExistsError('Completed control already exists')


def setup(protocol,phase,config):
    if torch.cuda.get_device_name()!=protocol['gpu']:raise AssertionError('Use the paired GPU model')
    set_random_seed(config.experiment.seed)
    backend=build_view_denoiser(config);_configure_denoiser(config,backend)
    prepared=prepare_native_backend(config,backend)
    conditioning=backend.conditioning_for_prompt_indices(prepared,[8],batch_size=config.generation.batch_size)
    shared=dict(checkpoint=config.model.id,revision=config.model.revision,precision=config.model.precision,
        prompt=load_directional_prompts(config.prompt.path)[2],negative_prompt='',seed=config.experiment.seed,
        conditioning_sha256=conditioning_digest(conditioning),timesteps=backend.timesteps.cpu().tolist(),
        sigmas=backend.pipeline.scheduler.sigmas.cpu().tolist(),guidance=config.generation.guidance_scale,
        backend_details=backend.backend_metadata())
    shared=json.loads(json.dumps(shared))
    if phase!='A':
        previous=read(control_path(protocol,'A'))
        if shared!=previous['shared']:raise AssertionError('Checkpoint/conditioning/schedule/precision changed')
    return backend,prepared,conditioning,shared


@torch.no_grad()
def single_native_or_implied(backend,conditioning,epsilon,phase,native_states=None):
    state=backend.initialize_native_state(epsilon.to(backend.device));records=[];states=[]
    for index,t in enumerate(backend.timesteps):
        if phase=='A':
            state,pair=backend.native_step_with_endpoints(state,t,conditioning)
            implied=pair.reconstruct_next()
            torch.testing.assert_close(implied,state,atol=2e-6,rtol=2e-6)
            extra=difference_stats(state,implied,'native_vs_implied')
            states.append(state.cpu())
        else:
            pair=backend.predict_clean_and_endpoint(state,t,conditioning)
            state=pair.reconstruct_next()
            extra=difference_stats(native_states[index].to(state),state,'native_vs_implied')
        if not bool(torch.isfinite(state).all()):raise ValueError('Non-finite native trajectory')
        records.append(dict(step=index,timestep=float(t),**pair.schedule_stats(),**extra))
    return backend.decode_native_canvas(state),records,states


def run(protocol,phase):
    config=check_configs(protocol)[phase];prerequisites(protocol,phase)
    started=time.perf_counter();backend,prepared,conditioning,shared=setup(protocol,phase,config)
    folder=Path(protocol['folders'][phase]);folder.parent.mkdir(parents=True,exist_ok=True)
    before=backend.guided_prediction_count if hasattr(backend,'guided_prediction_count') else 0
    extra={};single=phase in 'ABCD';n=config.native_multidiffusion
    if single:
        folder.mkdir(exist_ok=phase=='D')
        a_folder=Path(protocol['folders']['A'])
        if phase=='A':
            epsilon=torch.randn(config.generation.batch_size,backend.native_channels,n.patch_size,n.patch_size,
                generator=torch.Generator().manual_seed(config.experiment.seed))
            torch.save(epsilon,a_folder/'initial_epsilon.pt')
        else:epsilon=torch.load(a_folder/'initial_epsilon.pt',map_location='cpu',weights_only=True)
        extra['initial_epsilon_sha256']=digest(epsilon)
        if phase in 'AB':
            reference=None if phase=='A' else torch.load(a_folder/'native_states.pt',map_location='cpu',weights_only=True)
            result,timing=timed(lambda:single_native_or_implied(backend,conditioning,epsilon,phase,reference))
            image,steps,states=result
            if phase=='A':torch.save(states,folder/'native_states.pt')
            else:
                baseline=torch.load(a_folder/'final_rgb.pt',map_location=backend.device,weights_only=True)
                extra.update(difference_stats(baseline,image,'final_rgb_native_vs_implied'))
            name='native_final.png' if phase=='A' else 'implied_endpoint_final.png'
        else:
            write_json(folder/(phase+'_real_vae_identity_preflight.json'),identity_vae_preflight(backend,tuple(epsilon.shape)))
            initial=backend.initialize_native_state(epsilon.to(backend.device))
            result,timing=timed(lambda:single_patch_residual_trajectory(backend,initial,conditioning,correction=phase=='D'))
            image,steps,count=result
            if count!=len(backend.timesteps):raise AssertionError('Unexpected C/D evaluations')
            baseline=torch.load(Path(protocol['folders']['B'])/'final_rgb.pt',map_location=backend.device,weights_only=True)
            extra.update(difference_stats(image,baseline,'final_rgb_vs_B'))
            for row in steps:
                for suffix in ('mae','rmse','max_abs'):
                    row['roundtrip_'+suffix]=row['roundtrip_residual_'+suffix]
                    if phase=='D':row['corrected_vs_original_'+suffix]=row['correction_recovery_'+suffix]
            name=phase+'_final.png'
        if not bool(torch.isfinite(image).all()):raise ValueError('Non-finite decoded image')
        tensor_to_pil(image[0]).save(folder/name)
        torch.save(image.cpu(),folder/(phase+'_final_rgb.pt' if phase in 'CD' else 'final_rgb.pt'))
        save_steps(folder,phase,steps)
        expected=len(backend.timesteps)
    else:
        pair_folder=Path(protocol['pair_folder']);pair_folder.mkdir(parents=True,exist_ok=True)
        initial_path=pair_folder/'initial_native.pt'
        if phase=='E':
            initial=backend.sample_initial_native_state(batch_size=config.generation.batch_size,
                native_height=n.canvas_height,native_width=n.canvas_width,
                generator=torch.Generator(device='cpu').manual_seed(config.experiment.seed)).cpu()
            torch.save(initial,initial_path)
        else:initial=torch.load(initial_path,map_location='cpu',weights_only=True)
        extra['initial_native_sha256']=digest(initial)
        if phase!='E' and extra['initial_native_sha256']!=read(control_path(protocol,'E'))['initial_native_sha256']:
            raise AssertionError('Global initialization differs')
        pipe=PlanarImpliedEndpointConsensusPipeline(native_config=n,backend=backend,residual_correction=phase=='G')
        extra['geometry']=dict(native=asdict(pipe.native_layout),rgb=asdict(pipe.rgb_layout),factor=backend.native_spatial_factor)
        expected=pipe.native_layout.num_patches*len(backend.timesteps)
        if phase=='E':
            result,timing=timed(lambda:NativeMultiDiffusionPipeline(native_config=n,backend=backend,
                overlap_disagreement=True,clean_rgb_overlap_disagreement=True).run(initial.to(backend.device),prepared))
        else:
            locals_=pipe.initialize_local_states(initial.to(backend.device))
            extra['local_initial_crops_bit_identical']=True
            del initial
            if phase=='G':
                folder.parent.mkdir(parents=True,exist_ok=True)
                write_json(folder.parent/'real_vae_identity_preflight.json',identity_vae_preflight(backend,
                    (config.generation.batch_size,backend.native_channels,n.patch_size,n.patch_size)))
            result,timing=timed(lambda:pipe.run(locals_,prepared))
        save_result(folder,config,backend,result,timing)
        name='result.png';extra['audit']=getattr(result,'audit',None)
    count=backend.guided_prediction_count-before
    if count!=expected:raise AssertionError('Model evaluation count mismatch')
    if conditioning_digest(conditioning)!=shared['conditioning_sha256'] or backend.timesteps.cpu().tolist()!=shared['timesteps'] or backend.pipeline.scheduler.sigmas.cpu().tolist()!=shared['sigmas']:
        raise AssertionError('Shared schedule/conditioning mutated')
    data=dict(experiment=phase,backend=config.model.pipeline,config=config.to_dict(),shared=shared,
        training_free=True,correction=phase in 'DG',guided_predictions=count,extra_denoiser_calls=0,
        timing=timing,total_seconds=time.perf_counter()-started,job=os.environ.get('SLURM_JOB_ID'),
        node=platform.node(),gpu=torch.cuda.get_device_name(),output=str(folder/name),**extra)
    write_json(control_path(protocol,phase),data)
    print('Completed',phase,folder,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',default='configs/experiments/trajectory/sd35-ladder.json')
    parser.add_argument('--phase',required=True,choices=list('ABCDEFG'))
    args=parser.parse_args();run(read(args.protocol),args.phase)


if __name__=='__main__':main()
