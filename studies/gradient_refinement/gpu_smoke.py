"""Bounded 4K feasibility test with actual saved clean-proposal sufficient statistics."""
import os,time
from dataclasses import replace
import torch
from diffpano.gradient_fusion import GradientSettings,edges,reconstruct,GuidanceAccumulator
from diffpano.poisson_reference import ConnectivityCache
from studies.gradient_refinement.common import *

@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    start=time.perf_counter();sources=source_hashes();records=[]
    path=ROOT/'outputs/10.6gradient-blending/seed0-v1/cases/ruins/rgb/intermediate/step-10/sufficient-statistics.pt'
    data=torch.load(path,map_location='cuda',weights_only=False)
    print('STATISTICS_KEYS',list(data),flush=True)
    reference=data['reference']
    # This is saved per-source confidence-select guidance, never claimed to be
    # max guidance. Max source construction is validated independently on CPU.
    guidance=data['selected'];support=data['support']
    reference=reference.float();guidance=tuple(x.float() for x in guidance);support=tuple(m.bool() for m in support)
    assert reference.shape==(1,3,2048,4096)
    cache=ConnectivityCache();cache.check(support)
    for mode in ('screened','global_mean','single_pixel','coarse_color'):
        settings=GradientSettings('poisson_select',reference_mode=mode)
        anchor=dict(erp_y=1024,erp_x=2048,color=reference[...,1024,2048],camera_id='synthetic test anchor only')
        torch.cuda.reset_peak_memory_stats()
        try:
            result,record=reconstruct(reference,guidance,support,settings,anchor=anchor,connectivity=cache)
            record['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30;record['test']='actual saved guidance';records.append(record)
            print('GPU_SOLVE',record,flush=True)
            del result
        except Exception as error:
            write(OUT/'gpu-smoke.json',dict(passed=False,source_hashes=sources,records=records,failure=getattr(error,'diagnostics',str(error)),job=os.environ['SLURM_JOB_ID']))
            raise
    assert sources==source_hashes(),'Source changed during solver smoke'
    write(OUT/'gpu-smoke.json',dict(passed=True,source_hashes=sources,records=records,elapsed_seconds=time.perf_counter()-start,
        input_path=str(path),input_sha256=sha(path),production_shape=[1,3,2048,4096],job=os.environ['SLURM_JOB_ID']))
if __name__=='__main__':main()
