"""Launch unchanged original SphereDiff classes; capture outputs/provenance only."""
import argparse
import collections
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scripts.bridge_factorial_common import ROOT,SPHERE,SPHERE_COMMIT,require_validation,write_new,source_hashes,sha,verify_sources


@torch.no_grad()
def run(name):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('GPU Slurm allocation required')
    manifest=require_validation();spec=manifest['references'][name];folder=ROOT/'references'/name
    if folder.exists():raise FileExistsError(folder)
    sys.path.insert(0,str(SPHERE));import pipelines_ours
    cls=getattr(pipelines_ours,spec['pipeline']);source_before=verify_sources()
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    started=time.perf_counter()
    pipe=cls.from_pretrained(spec['model_source'],revision=spec['revision'],variant=spec['variant'],torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.to(torch.device('cuda'),dtype=torch.bfloat16)
    original_order=pipe.scheduler.config.get('solver_order',1)
    if original_order>1:pipe.scheduler.config.solver_order=1  # Exact original static launcher behavior.
    calls=[0];shapes=collections.Counter();forward=pipe.transformer.forward
    def counted(*args,**kwargs):
        calls[0]+=1
        hidden=kwargs.get('hidden_states',args[0] if args else None)
        if hidden is not None:shapes[str(list(hidden.shape))]+=1
        return forward(*args,**kwargs)
    pipe.transformer.forward=counted
    generator=torch.Generator(device='cuda').manual_seed(0)
    kwargs=dict(spec['call'],prompt_txt_path=str(SPHERE/'data/prompts/ruins.txt'),generator=generator)
    def progress(pipeline,step,timestep,callback_kwargs):
        print('SphereDiff',name,step+1,'/',spec['call']['num_inference_steps'],'forwards',calls[0],flush=True)
        return callback_kwargs
    kwargs['callback_on_step_end']=progress
    effective={k:v.default for k,v in inspect.signature(cls.__call__).parameters.items() if v.default is not inspect.Parameter.empty}
    effective.update(spec['call']);effective['prompt_txt_path']=str(SPHERE/'data/prompts/ruins.txt');effective['generator']='torch.Generator(cuda), seed=0';effective['callback_on_step_end']='logging-only; returns unchanged callback kwargs'
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
    output=pipe(**kwargs)
    torch.cuda.synchronize();runtime=time.perf_counter()-t;pipe.transformer.forward=forward
    im=output.images[0]
    if im.size!=(4096,2048):raise AssertionError('Original reference output dimensions changed')
    if verify_sources()!=source_before:raise AssertionError('Original SphereDiff source modified')
    scheduler=dict(class_name=type(pipe.scheduler).__name__,config=dict(pipe.scheduler.config),timesteps=pipe.scheduler.timesteps.detach().cpu().tolist())
    if hasattr(pipe.scheduler,'sigmas'):scheduler['sigmas']=pipe.scheduler.sigmas.detach().cpu().tolist()
    for k,v in scheduler['config'].items():
        if isinstance(v,set):scheduler['config'][k]=sorted(v)
    metadata=dict(study=manifest['study'],external_reference=True,in_factorial_contrasts=False,backend=name,source_commit=SPHERE_COMMIT,
        original_source=str(SPHERE),source_hashes=source_before,wrapper_source_hashes=source_hashes(),spec=spec,effective_call=effective,prompt=manifest['prompt'],seed=0,
        generator_device='cuda',initialization_paired_with_DiffPano=False,representation='original persistent spherical native latent state',
        original_launcher_solver_order=original_order,effective_solver_order=pipe.scheduler.config.get('solver_order',1),prepared_schedule=scheduler,
        transformer_forwards=calls[0],transformer_input_shape_histogram=dict(shapes),output_resolution=[2048,4096],
        runtime_seconds=runtime,total_seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,host_max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        job=os.environ['SLURM_JOB_ID'],node=platform.node(),gpu=torch.cuda.get_device_name(),
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version(),diffusers=importlib.metadata.version('diffusers'),transformers=importlib.metadata.version('transformers')),
        limitations=['External method reference, not an isolated factorial intervention.','Different model sources/revisions for FLUX; see checkpoint provenance audit.','Native output preserved; compare at common perspective resolution.'])
    folder.mkdir(parents=True,exist_ok=False);im.save(folder/'final_result.png');write_new(folder/'metadata.json',metadata)
    print('COMPLETED ORIGINAL SPHEREDIFF',name,'seconds',runtime,'forwards',calls[0],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=['flux','sana']);run(p.parse_args().backend)
