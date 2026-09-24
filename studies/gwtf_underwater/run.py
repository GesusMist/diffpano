import argparse
import importlib.util
import os
import time
import torch
from studies.gwtf_noise.statistics import state_statistics
from studies.gwtf_erp4k.run import spatial_probe
from studies.gwtf_underwater.common import *

def runner():
    spec=importlib.util.spec_from_file_location('_underwater_frozen_generation',Path('studies/gwtf_noise/run.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.ROOT=ROOT;mod.require_validation=require_validation;mod.source_hashes=source_hashes
    saved_write=mod.write
    def write_metadata(path,value):
        if Path(path).name=='metadata.json':
            value.pop('reused_current_control',None)
            value.update(prompt_sha256=sha(PROMPT),clean_RGB_ERP_resolution=list(ERP_SIZE),camera_count=89,FOV_x=80,FOV_y=80,
                fresh_underwater_control=True,initial_states_match_validated_seed0=True,
                limitations=['One underwater prompt set and seed 0','Matched-ray metrics do not establish full 3D coherence','High-frequency energy is not perceptual quality'])
        saved_write(path,value)
    mod.write=write_metadata
    return mod

@torch.no_grad()
def run(name,method):
    gate=require_validation();start=time.perf_counter();r=runner()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    out=folder(name,method)
    if (out/'metadata.json').exists() or (out/'gpu-preflight.json').exists():raise FileExistsError(out)
    c,_,cams,old,_=geometry(name)
    assert c.to_dict()==gate['models'][name]['config']
    r.set_random_seed(0);b=r.build_view_denoiser(c);r._configure_denoiser(c,b)
    p=r.BridgeFactorialPipeline(backend=b,cameras=cams,erp_size=ERP_SIZE,warp_operator=r.make_operator(c),backend_name=name)
    conditions,slots=r.prepare(c,b,p)
    provenance,audit=r.runtime_audit(c,b,p,conditions,slots,old)
    assert set(audit['differences'])=={'conditioning_sha256'},audit['differences']
    assert not audit['full_config_match']
    audit.update(expected_config_changes=differences(old['config'],c.to_dict()),
        expected_conditioning_change=True,all_other_runtime_settings_match=True,baseline=str(baseline.ROOT/name))
    assert {x['path'] for x in audit['expected_config_changes']}=={'prompt.path'}
    rng=torch.random.get_rng_state().clone()
    with r.count_calls(b,name,forbid=True) as calls:
        t=time.perf_counter()
        if method=='gwtf':
            states,init=r.initialize_gwtf_shared_noise(b,cams,r.GWTFlowNoiseConfig(*provenance['noise_grid'],0),
                c.generation.batch_size,camera_slots=list(range(89)))
        else:
            states,init=r.initialize_erp_noise(b,cams,r.ERPNoiseConfig('V-shared-erp',*provenance['noise_grid'],0),c.generation.batch_size)
        init_seconds=time.perf_counter()-t
        assert init['initial_local_sha256']==gate['models'][name]['expected_initial_sha256'][method]
        exact=state_statistics(states,b,cams);spatial=spatial_probe(p,c,b)
    assert calls==dict(denoiser=0,encode=0,decode=0,initialize=89)
    assert torch.equal(rng,torch.random.get_rng_state())
    write(out/'gpu-preflight.json',dict(passed=True,backend=name,method=method,environment=r.environment(),
        provenance=provenance,baseline_audit=audit,spatial_probe=spatial,calls=calls,initialization=init,
        statistics=exact,prompt_sha256=sha(PROMPT),source_hashes=source_hashes()))
    print('UNDERWATER PREFLIGHT PASSED',name,method,flush=True)
    r.generate(name,method,c,b,p,conditions,states,init,provenance,audit,init_seconds,exact,start)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=BACKENDS);p.add_argument('method',choices=METHODS)
    a=p.parse_args();run(a.backend,a.method)
