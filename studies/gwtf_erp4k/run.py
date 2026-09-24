"""Reuse the frozen GWTFlow generation function with a new ERP grid/output root."""
import argparse
import importlib.util
import os
import time
from pathlib import Path
import torch
from studies.gwtf_noise.statistics import state_statistics
from studies.gwtf_erp4k.common import *


def runner():
    spec=importlib.util.spec_from_file_location('_gwtf_erp4k_frozen_generation',Path('studies/gwtf_noise/run.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.ROOT=ROOT;mod.require_validation=require_validation;mod.source_hashes=source_hashes
    saved_write=mod.write
    def write_metadata(path,value):
        if Path(path).name=='metadata.json':
            old=read(BASE/value['backend']/'metadata.json')
            assert value['initialization']['initial_local_sha256']==old['initialization']['initial_local_sha256']
            value.pop('reused_current_control',None)
            value.update(baseline_gwtf_output=str(BASE/value['backend']),
                scientific_setting_diff=differences(old['config'],value['config']),
                clean_RGB_ERP_resolution=list(ERP_SIZE),camera_count=89,FOV_x=80,FOV_y=80,
                initial_local_states_identical_to_baseline=True,
                limitations=['Single ruins prompt and seed 0','ERP pixel-frequency metrics at different raster resolutions are not directly comparable'])
        saved_write(path,value)
    mod.write=write_metadata
    return mod


def spatial_probe(p,c,b):
    rgb=torch.full((1,3,c.view.height,c.view.width),.2,device=b.device)
    acc=p._accumulator(1)
    for cam in p.cameras:p._accumulate(acc,rgb,cam,{})
    result=p._finalize(acc,{})
    assert result.erp_rgb.shape==(1,3,*ERP_SIZE) and bool(torch.isfinite(result.erp_rgb).all())
    count=result.contributor_count
    assert bool((count>0).all()),'Uncovered 4K ERP pixels'
    returned=p.diagnostic_operator.erp_to_perspective(result.erp_rgb,p.cameras[7])
    assert returned.shape==rgb.shape and bool(torch.isfinite(returned).all())
    error=float((returned-rgb).abs().max());assert error<1e-5,error
    return dict(coverage_percent=100.,minimum_contributors=int(count.min()),constant_error_max=error,
        ERP_resolution=list(result.erp_rgb.shape[-2:]),local_RGB_resolution=list(rgb.shape[-2:]))


@torch.no_grad()
def run(name):
    gate=require_validation();start=time.perf_counter();r=runner()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    folder=ROOT/name
    if folder.exists():raise FileExistsError(folder)
    c,stub,cams,old,_=geometry(name)
    assert c.to_dict()==gate['models'][name]['config']
    r.set_random_seed(0);b=r.build_view_denoiser(c);r._configure_denoiser(c,b)
    p=r.BridgeFactorialPipeline(backend=b,cameras=cams,erp_size=ERP_SIZE,warp_operator=r.make_operator(c),backend_name=name)
    conditions,slots=r.prepare(c,b,p)
    provenance,audit=r.runtime_audit(c,b,p,conditions,slots,old)
    assert audit['matched'],audit['differences']
    assert not audit['full_config_match']
    audit['expected_config_changes']=differences(old['config'],c.to_dict())
    assert {x['path'] for x in audit['expected_config_changes']}=={'erp.height','erp.width'}
    rng=torch.random.get_rng_state().clone()
    with r.count_calls(b,name,forbid=True) as calls:
        t=time.perf_counter()
        states,init=r.initialize_gwtf_shared_noise(b,cams,r.GWTFlowNoiseConfig(*provenance['noise_grid'],0),
            c.generation.batch_size,camera_slots=list(range(89)))
        init_seconds=time.perf_counter()-t
        assert init['initial_local_sha256']==old['initialization']['initial_local_sha256']
        exact=state_statistics(states,b,cams)
        spatial=spatial_probe(p,c,b)
    assert calls==dict(denoiser=0,encode=0,decode=0,initialize=89)
    assert torch.equal(rng,torch.random.get_rng_state())
    write(folder/'gpu-preflight.json',dict(passed=True,backend=name,environment=r.environment(),
        provenance=provenance,baseline_audit=audit,spatial_probe=spatial,calls=calls,
        initial_local_states_identical_to_baseline=True,initialization=init,statistics=exact,source_hashes=source_hashes()))
    print('4K PREFLIGHT PASSED',name,'identical GWTFlow initialization; full ERP coverage',flush=True)
    r.generate(name,'gwtf',c,b,p,conditions,states,init,provenance,audit,init_seconds,exact,start)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=BACKENDS);run(p.parse_args().backend)
