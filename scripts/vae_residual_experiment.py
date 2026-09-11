#!/usr/bin/env python3
"""Training-free LookingGlass residual controls; never creates/trains a network."""
import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import torch

from diffpano.config import load_experiment_config
from diffpano.diagnostics import tensor_to_pil
from diffpano.initialization import set_random_seed
from diffpano.native_multidiffusion import prepare_native_backend
from diffpano.pipelines import build_view_denoiser
from diffpano.trajectory import conditioning_digest,difference_stats
from diffpano.vae_residual import identity_vae_preflight,single_patch_residual_trajectory
from scripts.generate import _configure_denoiser
from scripts.paired_rgb_endpoint import digest,no_roundtrip,paired_config_check,timed,save_result


def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def save_steps(folder,label,steps):
    write_json(folder/(label+'_steps.json'),steps)
    with (folder/(label+'_steps.csv')).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in steps for k in row)))
        writer.writeheader();writer.writerows(steps)


def setup(spec):
    config=load_experiment_config(spec['base_config']);paired_config_check(config)
    previous=Path(spec['previous_pair']);old=json.loads((previous/'comparison.json').read_text())
    if config.to_dict()!=old['config']:raise AssertionError('Reference model/settings/config differ')
    if spec['backend']!=config.model.pipeline or config.model.pipeline=='pixeldit':raise ValueError('Latent backend required')
    if torch.cuda.get_device_name()!=old['gpu_name']:raise AssertionError('Use the same GPU model as the saved paired controls')
    set_random_seed(config.experiment.seed)
    backend=build_view_denoiser(config);_configure_denoiser(config,backend)
    prepared=prepare_native_backend(config,backend)
    conditioning=backend.conditioning_for_prompt_indices(prepared,[8],batch_size=config.generation.batch_size)
    historical=json.loads((previous/'implied_endpoint_consensus/metadata.json').read_text())
    scheduler=backend.pipeline.scheduler
    if backend.timesteps.cpu().tolist()!=historical['scheduler_timesteps']:raise AssertionError('Timestep mismatch')
    if hasattr(scheduler,'sigmas') and scheduler.sigmas.cpu().tolist()!=historical['scheduler_sigmas']:raise AssertionError('Shifted sigma mismatch')
    condition_hash=conditioning_digest(conditioning)
    if condition_hash!=old['fairness']['conditioning_sha256']:raise AssertionError('Conditioning differs from historical control')
    return config,previous,old,backend,prepared,conditioning


def run_cd(spec,folder):
    config,previous,old,backend,prepared,conditioning=setup(spec)
    epsilon=torch.load(previous/'one_patch_roundtrip/initial_epsilon.pt',map_location='cpu',weights_only=True)
    preflight=identity_vae_preflight(backend,tuple(epsilon.shape))
    write_json(folder/'real_vae_identity_preflight.json',preflight)
    initial=backend.initialize_native_state(epsilon.to(backend.device));torch.save(epsilon,folder/'initial_epsilon.pt')
    images={};records={};counts={};timing={}
    # B is rerun only as a matched oracle for accumulated D/B numerical drift.
    before=getattr(backend,'guided_prediction_count',0)
    images['B'],timing['B']=timed(lambda:no_roundtrip(backend,initial.clone(),conditioning))
    counts['B']=backend.guided_prediction_count-before
    for label,correction in (('C',False),('D',True)):
        result,timing[label]=timed(lambda:single_patch_residual_trajectory(backend,initial.clone(),conditioning,correction=correction))
        images[label],records[label],counts[label]=result
        save_steps(folder,label,records[label])
    if set(counts.values())!={len(backend.timesteps)}:raise AssertionError('B/C/D evaluation counts differ')
    for label,image in images.items():tensor_to_pil(image[0]).save(folder/(label+'_final.png'))
    write_json(folder/'comparison.json',dict(config=config.to_dict(),counts=counts,timings=timing,
        training_free=True,extra_diagnostic_denoiser_calls=0,epsilon_sha256=digest(epsilon),
        conditioning_sha256=conditioning_digest(conditioning),same_initial_model_settings_schedule=True,
        C_vs_B=difference_stats(images['C'],images['B'],'final_rgb'),
        D_vs_B=difference_stats(images['D'],images['B'],'final_rgb'),
        C_matches_previous_png=(folder/'C_final.png').read_bytes()==(previous/'one_patch_roundtrip/roundtrip.png').read_bytes(),
        B_matches_previous_png=(folder/'B_final.png').read_bytes()==(previous/'one_patch_roundtrip/no_roundtrip.png').read_bytes()))



def run_g(spec, folder):
    # CD is a prerequisite; no image experiment proceeds after a failed identity oracle.
    cd = Path(spec['output'])/'CD'
    control = json.loads((cd/'comparison.json').read_text())
    if not control['training_free'] or not control['same_initial_model_settings_schedule']:
        raise AssertionError('Missing validated C/D controls')
    preflight = json.loads((cd/'real_vae_identity_preflight.json').read_text())
    if any(row['correction_recovery_max_abs'] > 2e-6 for row in preflight):
        raise AssertionError('C/D real VAE identity prerequisite failed')
    from diffpano.implied_endpoint_consensus import PlanarImpliedEndpointConsensusPipeline
    config, previous, old, backend, prepared, conditioning = setup(spec)
    initial = torch.load(previous/'initial_native.pt', map_location='cpu', weights_only=True)
    initial_hash = digest(initial)
    if initial_hash != old['initial_native_sha256']:
        raise AssertionError('Historical F initialization hash mismatch')
    pipeline = PlanarImpliedEndpointConsensusPipeline(native_config=config.native_multidiffusion,
        backend=backend, residual_correction=True)
    # Recheck the real VAE on this job before its first denoiser evaluation.
    shape=(initial.shape[0], backend.native_channels, pipeline.native_layout.patch_size, pipeline.native_layout.patch_size)
    write_json(folder/'real_vae_identity_preflight.json', identity_vae_preflight(backend, shape))
    local = pipeline.initialize_local_states(initial.to(backend.device))
    torch.save(initial, folder/'initial_native.pt')
    del initial
    result, timing = timed(lambda: pipeline.run(local, prepared))
    expected = old['model_evaluations']['implied_endpoint_consensus']
    if result.audit['guided_predictions'] != expected:
        raise AssertionError('G/F denoiser counts differ')
    result.audit['initial_local_states_equal_global_native_crops'] = True
    save_result(folder/'generation', config, backend, result, timing)
    write_json(folder/'comparison.json', dict(config=config.to_dict(), training_free=True,
        same_initial_model_settings_schedule=True, initial_native_sha256=initial_hash,
        conditioning_sha256=conditioning_digest(conditioning), previous_pair=str(previous),
        guided_predictions=expected, extra_diagnostic_denoiser_calls=0, timing=timing, audit=result.audit))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',required=True);parser.add_argument('--phase',choices=['CD','G'],required=True)
    args=parser.parse_args();spec=json.loads(Path(args.spec).read_text())
    folder=Path(spec['output'])/args.phase;folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'spec.json',spec);started=time.perf_counter()
    (run_cd if args.phase == 'CD' else run_g)(spec,folder)
    torch.cuda.synchronize()
    write_json(folder/'runtime.json',dict(runtime_seconds=time.perf_counter()-started,job=os.environ.get('SLURM_JOB_ID'),
        node=platform.node(),gpu=torch.cuda.get_device_name(),torch_version=str(torch.__version__)))
    print('complete',args.phase,spec['backend'],folder,flush=True)


if __name__=='__main__':main()
