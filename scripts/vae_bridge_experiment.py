#!/usr/bin/env python3
"""Reproducible C, dataset collection, bridge training/validation, D and G phases."""
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from diffpano.bridge_trajectory import run_roundtrip_trajectory
from diffpano.config import load_experiment_config
from diffpano.diagnostics import tensor_to_pil
from diffpano.implied_endpoint_consensus import PlanarImpliedEndpointConsensusPipeline
from diffpano.initialization import load_directional_prompts, set_random_seed
from diffpano.native_multidiffusion import prepare_native_backend
from diffpano.pipelines import build_view_denoiser
from diffpano.planar_pipeline import _negative_prompt
from diffpano.trajectory import conditioning_digest
from diffpano.vae_bridge import LatentBridge, load_bridge, projection_metrics, save_bridge
from scripts.generate import _configure_denoiser
from scripts.paired_rgb_endpoint import digest, paired_config_check, save_result


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def file_hash(path):
    value=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def save_steps(folder,steps):
    write_json(folder/'steps.json',steps)
    with (folder/'steps.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for s in steps for k in s)))
        writer.writeheader();writer.writerows(steps)


def environment():
    return dict(job=os.environ.get('SLURM_JOB_ID'),node=platform.node(),gpu=torch.cuda.get_device_name(),
                torch_version=str(torch.__version__))


def check_splits(protocol,evaluation_prompt,evaluation_seed):
    a,b=set(protocol['train_prompts']),set(protocol['validation_prompts'])
    sa,sb=set(protocol['train_seeds']),set(protocol['validation_seeds'])
    if not a or not b or a&b or sa&sb or evaluation_prompt in a|b or evaluation_seed in sa|sb:
        raise AssertionError('Training, validation and final evaluation prompts/seeds must be disjoint')


def make_backend(config):
    set_random_seed(config.experiment.seed)
    backend=build_view_denoiser(config);_configure_denoiser(config,backend)
    prepared=prepare_native_backend(config,backend)
    return backend,prepared


def configuration_audit(config,previous):
    current=config.to_dict(); saved=json.loads((previous/'comparison.json').read_text())
    if current!=saved['config']:
        raise AssertionError('Model, precision, guidance, geometry, seed or other reference config changed')
    return saved


@torch.no_grad()
def evaluation_phase(phase,spec,protocol,config,folder):
    previous=Path(spec['previous_pair']);old=configuration_audit(config,previous)
    backend,prepared=make_backend(config)
    conditioning=backend.conditioning_for_prompt_indices(prepared,[8],batch_size=config.generation.batch_size)
    before_condition=conditioning_digest(conditioning)
    scheduler=getattr(backend.pipeline,'scheduler',None)
    saved_schedule=json.loads((previous/'implied_endpoint_consensus/metadata.json').read_text())
    if backend.timesteps.cpu().tolist()!=saved_schedule['scheduler_timesteps']:
        raise AssertionError('Actual schedule differs from F/B/C controls')
    if hasattr(scheduler,'sigmas') and scheduler.sigmas.cpu().tolist()!=saved_schedule['scheduler_sigmas']:
        raise AssertionError('Actual shifted sigmas differ from controls')
    bridge=None;checkpoint=None
    if phase in ('D','G'):
        checkpoint=Path(spec['output'])/'training/best.pt'
        bridge,metadata=load_bridge(checkpoint,device=backend.device,expected_backend=spec['backend'])
        if metadata['config']!=config.to_dict():raise AssertionError('Bridge trained under different model settings')
    if phase in ('C','D'):
        epsilon=torch.load(previous/'one_patch_roundtrip/initial_epsilon.pt',map_location='cpu',weights_only=True)
        state=backend.initialize_native_state(epsilon.to(backend.device))
        result=run_roundtrip_trajectory(backend,state,conditioning,bridge)
        tensor_to_pil(result.final_rgb[0]).save(folder/'result.png')
        save_steps(folder,result.steps)
        counts=result.model_evaluations
        initialization=dict(epsilon_sha256=digest(epsilon),source=str(previous/'one_patch_roundtrip/initial_epsilon.pt'))
        # Retain the frozen historical reference, without rerunning B.
        reference=previous/'one_patch_roundtrip'/('roundtrip.png' if phase=='C' else 'no_roundtrip.png')
        reference_identical=reference.read_bytes()==(folder/'result.png').read_bytes()
    else:
        initial=torch.load(previous/'initial_native.pt',map_location='cpu',weights_only=True).to(backend.device)
        if digest(initial)!=old['initial_native_sha256']:raise AssertionError('G initialization differs from F')
        pipe=PlanarImpliedEndpointConsensusPipeline(native_config=config.native_multidiffusion,backend=backend,bridge=bridge)
        states=pipe.initialize_local_states(initial);del initial
        result=pipe.run(states,prepared)
        # Existing F output convention is retained exactly.
        save_result(folder/'generation',config,backend,result,{})
        counts=result.audit['guided_predictions'];reference_identical=None
        initialization=dict(initial_native_sha256=old['initial_native_sha256'],source=str(previous/'initial_native.pt'))
    if before_condition!=conditioning_digest(conditioning):raise AssertionError('Shared conditioning changed')
    if counts != len(backend.timesteps)*(3 if phase=='G' else 1):raise AssertionError('Evaluation count mismatch')
    write_json(folder/'experiment.json',dict(phase=phase,config=config.to_dict(),reference=str(previous),
        checkpoint=str(checkpoint) if checkpoint else None,checkpoint_sha256=file_hash(checkpoint) if checkpoint else None,
        initialization=initialization,model_evaluations=counts,extra_denoiser_calls=0,
        reference_png_bit_identical=reference_identical,conditioning_sha256=before_condition,
        actual_precision=str(backend.dtype),scheduler_timesteps=backend.timesteps.cpu().tolist(),
        same_model_config_and_schedule=True,**environment()))


@torch.no_grad()
def collect(spec,protocol,config,folder):
    backend,_=make_backend(config)
    evaluation_prompt=load_directional_prompts(config.prompt.path)[2]
    check_splits(protocol,evaluation_prompt,config.experiment.seed)
    p=config.native_multidiffusion.patch_size
    rows=[]
    for split,prompts,seeds in (('train',protocol['train_prompts'],protocol['train_seeds']),
                                ('validation',protocol['validation_prompts'],protocol['validation_seeds'])):
        for prompt_index,prompt in enumerate(prompts):
            prepared=backend.prepare_prompt_conditioning([prompt]*5,_negative_prompt(config))
            conditioning=backend.conditioning_for_prompt_indices(prepared,[8],batch_size=1)
            for seed in seeds:
                epsilon=torch.randn(1,backend.native_channels,p,p,generator=torch.Generator().manual_seed(seed)).to(backend.device)
                state=backend.initialize_native_state(epsilon)
                inputs=[];targets=[];stats=[]
                before=getattr(backend,'guided_prediction_count',0)
                for i,t in enumerate(backend.timesteps):
                    state,pair=backend.native_step_with_endpoints(state,t,conditioning)
                    projected=backend.encode_clean(backend.decode_clean(pair.clean))
                    if not bool(torch.isfinite(projected).all()) or not bool(torch.isfinite(pair.clean).all()):
                        raise ValueError('Nonfinite bridge training pair')
                    inputs.append(projected.cpu());targets.append(pair.clean.cpu())
                    stats.append(dict(step=i,timestep=float(t),**pair.schedule_stats(),**projection_metrics(pair.clean,projected)))
                calls=backend.guided_prediction_count-before
                if calls!=len(backend.timesteps):raise AssertionError('Collection made extra denoiser calls')
                path=folder/('{}-p{}-s{}.pt'.format(split,prompt_index,seed))
                torch.save(dict(input=torch.cat(inputs),target=torch.cat(targets),timesteps=backend.timesteps.cpu()),path)
                rows.append(dict(split=split,prompt=prompt,seed=seed,path=path.name,sha256=file_hash(path),
                    model_evaluations=calls,steps=stats))
                print('collected',spec['backend'],split,prompt_index,seed,flush=True)
    manifest=dict(backend=spec['backend'],channels=backend.native_channels,config=config.to_dict(),protocol=protocol,
        evaluation_prompt=evaluation_prompt,evaluation_seed=config.experiment.seed,
        teacher='ordinary native trajectory, same cached prediction supplies clean target; no roundtrip reinjection',
        rows=rows,**environment())
    write_json(folder/'manifest.json',manifest)


def read_dataset(path):
    manifest=json.loads((path/'manifest.json').read_text());splits={}
    check_splits(manifest['protocol'],manifest['evaluation_prompt'],manifest['evaluation_seed'])
    for split in ('train','validation'):
        inputs=[];targets=[];times=[]
        for row in manifest['rows']:
            if row['split']!=split:continue
            file=path/row['path']
            if file_hash(file)!=row['sha256']:raise AssertionError('Dataset tensor file changed')
            value=torch.load(file,map_location='cpu',weights_only=True)
            inputs.append(value['input']);targets.append(value['target']);times.append(value['timesteps'])
        splits[split]=(torch.cat(inputs),torch.cat(targets),torch.cat(times))
    return manifest,splits


@torch.no_grad()
def validate_bridge(bridge,tensors,l1_weight=1.,mse_weight=1.):
    bridge.eval();device=next(bridge.parameters()).device
    rows=[]
    for x,y,t in zip(*tensors):
        x,y=x[None].to(device),y[None].to(device)
        pred=bridge(x,t.to(device))
        stats=projection_metrics(y,x,pred)
        stats['loss']=l1_weight*stats['bridge_mae']+mse_weight*stats['bridge_rmse']**2
        rows.append(stats)
    keys=('roundtrip_mae','roundtrip_rmse','normalized_roundtrip_mae','bridge_mae','bridge_rmse','normalized_bridge_mae','loss')
    summary={k:sum(r[k] for r in rows)/len(rows) for k in keys}
    summary['error_reduction_percent']=100*(summary['roundtrip_mae']-summary['bridge_mae'])/max(summary['roundtrip_mae'],1e-8)
    return summary,rows


def train(spec,protocol,config,folder):
    dataset=Path(spec['output'])/'dataset'
    manifest,splits=read_dataset(dataset)
    if manifest['config']!=config.to_dict():raise AssertionError('Dataset configuration mismatch')
    tc=protocol['training'];torch.manual_seed(tc['seed'])
    bridge=LatentBridge(manifest['channels'],**protocol['architecture']).cuda()
    optimizer=torch.optim.Adam(bridge.parameters(),lr=tc['learning_rate'])
    generator=torch.Generator().manual_seed(tc['seed'])
    x,y,t=splits['train'];crop=min(tc['crop_size'],x.shape[-1],x.shape[-2])
    meta=dict(backend=spec['backend'],config=config.to_dict(),protocol=protocol,
        dataset_manifest=str(dataset/'manifest.json'),manifest_sha256=file_hash(dataset/'manifest.json'),
        train_examples=len(x),validation_examples=len(splits['validation'][0]),
        parameter_count=sum(p.numel() for p in bridge.parameters()),**environment())
    best,initial_rows=validate_bridge(bridge,splits['validation'],tc['l1_weight'],tc['mse_weight'])
    history=[dict(step=0,train_loss=None,validation=best)]
    best_loss=best['loss'];best_step=0
    save_bridge(folder/'best.pt',bridge,dict(meta,best_step=0,validation=best))
    for step in range(1,tc['steps']+1):
        bridge.train();indices=torch.randint(len(x),(tc['batch_size'],),generator=generator)
        bx=[];by=[]
        for i in indices:
            top=int(torch.randint(x.shape[-2]-crop+1,(1,),generator=generator))
            left=int(torch.randint(x.shape[-1]-crop+1,(1,),generator=generator))
            bx.append(x[i,:,top:top+crop,left:left+crop]);by.append(y[i,:,top:top+crop,left:left+crop])
        inputs=torch.stack(bx).cuda();targets=torch.stack(by).cuda();times=t[indices].cuda()
        prediction=bridge(inputs,times)
        loss=tc['l1_weight']*F.l1_loss(prediction,targets)+tc['mse_weight']*F.mse_loss(prediction,targets)
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite bridge training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(bridge.parameters(),1.)
        optimizer.step()
        if step%tc['validate_every']==0 or step==tc['steps']:
            val,_=validate_bridge(bridge,splits['validation'],tc['l1_weight'],tc['mse_weight'])
            history.append(dict(step=step,train_loss=float(loss),validation=val))
            if val['loss']<best_loss:
                best_loss=val['loss'];best_step=step
                save_bridge(folder/'best.pt',bridge,dict(meta,best_step=step,validation=val))
            write_json(folder/'history.json',history)
            print(spec['backend'],'step',step,'train',float(loss),'validation',val['loss'],flush=True)
    save_bridge(folder/'last.pt',bridge,dict(meta,step=tc['steps']))
    bridge,best_metadata=load_bridge(folder/'best.pt',device='cuda',expected_backend=spec['backend'])
    validation,per_example=validate_bridge(bridge,splits['validation'],tc['l1_weight'],tc['mse_weight'])
    training,_=validate_bridge(bridge,splits['train'],tc['l1_weight'],tc['mse_weight'])
    write_json(folder/'validation.json',dict(metadata=meta,best_step=best_step,identity=history[0]['validation'],
        training=training,validation=validation,per_example=per_example,checkpoint_sha256=file_hash(folder/'best.pt')))


def standalone_validation(spec,protocol,folder):
    _,splits=read_dataset(Path(spec['output'])/'dataset')
    bridge,metadata=load_bridge(Path(spec['output'])/'training/best.pt',device='cuda',expected_backend=spec['backend'])
    values,rows=validate_bridge(bridge,splits['validation'],protocol['training']['l1_weight'],protocol['training']['mse_weight'])
    write_json(folder/'validation.json',dict(metadata=metadata,validation=values,per_example=rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True);parser.add_argument('--phase',required=True,choices=['C','collect','train','validate','D','G'])
    args=parser.parse_args();spec=json.loads(Path(args.spec).read_text());protocol=json.loads(Path(spec['protocol']).read_text())
    config=load_experiment_config(spec['base_config']);paired_config_check(config)
    if config.model.pipeline=='pixeldit':raise ValueError('No PixelDiT bridge in this experiment')
    if spec['backend']!=config.model.pipeline:raise ValueError('Backend/spec mismatch')
    phase_dir={'collect':'dataset','train':'training'}.get(args.phase,args.phase)
    folder=Path(spec['output'])/phase_dir;folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'spec.json',spec);write_json(folder/'protocol.json',protocol)
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    if args.phase in ('C','D','G'):evaluation_phase(args.phase,spec,protocol,config,folder)
    elif args.phase=='collect':collect(spec,protocol,config,folder)
    elif args.phase=='train':train(spec,protocol,config,folder)
    else:standalone_validation(spec,protocol,folder)
    torch.cuda.synchronize()
    write_json(folder/'runtime.json',dict(runtime_seconds=time.perf_counter()-started,
        allocated_gib=torch.cuda.max_memory_allocated()/1024**3,reserved_gib=torch.cuda.max_memory_reserved()/1024**3,**environment()))
    print('complete',args.phase,spec['backend'],folder,flush=True)


if __name__=='__main__':main()
