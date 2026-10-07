"""Bounded full-resolution solver smoke test, without loading a generative model."""
import os
from dataclasses import replace
import torch
from diffpano.gradient_fusion import GradientSettings,edges,reconstruct
from studies.gradient_blending.common import *

@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    gate=read(OUT/'validation.json');assert gate['passed'] and gate['core_hashes']==core_hashes()
    torch.manual_seed(45);device='cuda'
    reference=torch.rand(1,3,2048,4096,device=device)*.5
    target=reference+.05*torch.rand_like(reference)
    guidance=edges(target);support=tuple(torch.ones_like(v[:,:1],dtype=torch.bool) for v in guidance)
    support[0][...,100:110]=False;support[1][...,600:610,:]=False
    torch.cuda.reset_peak_memory_stats();records=[]
    for lam in [.01,.1,1.]:
        result,record=reconstruct(reference,guidance,support,GradientSettings('poisson_select',lambda_color=lam))
        assert record['converged'];records.append(record)
        print('FULL_RESOLUTION_SOLVE',record,flush=True)
        del result
    write(OUT/'gpu-smoke.json',dict(passed=True,core_hashes=core_hashes(),records=records,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,job=os.environ['SLURM_JOB_ID']))
if __name__=='__main__':main()
