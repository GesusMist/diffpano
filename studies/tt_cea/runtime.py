"""Fresh backend setup and audits using the historical, unmodified helpers."""
import time
import torch
from diffpano.initialization import set_random_seed
from diffpano.pipelines import build_view_denoiser
from scripts.generate import _configure_denoiser
from scripts.dense_erp_experiment import load_directional_prompts,canonical_schedule,schedule_metadata
from diffpano.planar_pipeline import _negative_prompt
from diffpano.dense_consensus import prepare_camera_conditioning
from diffpano.erp_noise_initialization import native_cameras,primary_noise_size,states_digest
from diffpano.gwtf_noise_initialization import initialize_gwtf_shared_noise,GWTFlowNoiseConfig
from studies.gwtf_noise.run import count_calls,runtime_audit,environment
from studies.gwtf_erp4k.common import differences
from studies.tt_cea.common import *
from studies.tt_cea.cameras import load_cover
from studies.tt_cea.pipeline import ExperimentalPipeline
from studies.tt_cea.schedule import prepare_interval_table


def prepare_schedule(b,c,steps):
    b.prepare(num_steps=steps,view_height=c.view.height,view_width=c.view.width)
    return prepare_interval_table(b)


def setup(name,prompt,case,covers=None):
    c,_,_,old,row=resolved(name,prompt,case)
    set_random_seed(0);b=build_view_denoiser(c);_configure_denoiser(c,b)
    table=prepare_schedule(b,c,row['ordinary_steps'])
    bank=b.prepare_prompt_conditioning(load_directional_prompts(c.prompt.path),_negative_prompt(c))
    groups={}
    for cover in covers or [row['cameras']]:
        cams,ids=load_cover(cover,c.view.height,c.view.width)
        conditions,slots=prepare_camera_conditioning(b,bank,cams,c.dense_consensus.prompt_assignment,c.generation.batch_size)
        groups[cover]=dict(cameras=cams,ids=ids,conditions=conditions,slots=slots)
    del bank
    return c,b,old,row,table,groups


def check_runtime(c,b,p,group,old,case):
    provenance,audit=runtime_audit(c,b,p,group['conditions'],group['slots'],old)
    allowed=set()
    if case=='TB':allowed.add('prepared_schedule')
    if CASES[case]['cameras']=='ea89':allowed.update(('camera_sha256','camera_geometry_sha256','conditioning_sha256','prompt_indices'))
    unexpected=set(audit['differences'])-allowed
    assert not unexpected,(unexpected,audit['differences'])
    config_diff=differences(old['config'],c.to_dict())
    assert {d['path'] for d in config_diff}==({'generation.num_inference_steps'} if case=='TB' else set()),config_diff
    audit.update(allowed_runtime_differences=sorted(allowed),resolved_config_differences=config_diff,accepted=True)
    return provenance,audit


@torch.no_grad()
def initialize(b,name,group,old=None,execution_order=None):
    h,w=primary_noise_size(native_cameras(b,group['cameras']))
    rng=torch.random.get_rng_state().clone();start=time.perf_counter()
    with count_calls(b,name,forbid=True) as calls:
        states,record=initialize_gwtf_shared_noise(b,group['cameras'],GWTFlowNoiseConfig(h,w,0),1,
            camera_slots=group['ids'],execution_order=execution_order)
    assert calls==dict(denoiser=0,encode=0,decode=0,initialize=89)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert states_digest(states)==record['initial_local_sha256']
    assert all(s.dtype==torch.float32 and s.device.type=='cpu' and bool(torch.isfinite(s).all()) for s in states)
    if old is not None:assert record['initial_local_sha256']==old['initialization']['initial_local_sha256']
    return states,record,dict(seconds=time.perf_counter()-start,calls=calls)
