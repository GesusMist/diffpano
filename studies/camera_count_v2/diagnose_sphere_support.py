"""Read-only isolation of the original RGB assembly's nonfinite weights."""
import math
import os
import torch
import torch.nn.functional as F
from .common import *
from .cameras import build
from .spherediff_adapter import reference_geometry,directions

@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    sf=reference_geometry();cams=build('old',70);dirs=directions(cams).cuda().bfloat16().float()
    canvas=torch.zeros((1,1,1,2048,4096),device='cuda');den=torch.zeros_like(canvas)
    rgb=torch.ones((1,1,1,1024,1024),device='cuda');prior=0;records=[];grids=[]
    original=F.grid_sample
    def sample(input,grid,*a,**kw):
        result=original(input,grid,*a,**kw)
        bad=~torch.isfinite(result)
        if bool(bad.any()):
            ix=torch.nonzero(bad)[:8]
            flat=grid.reshape(-1,2)
            grids.append(dict(padding=kw.get('padding_mode'),nonfinite_output=int(bad.sum()),
                nonfinite_grid=int((~torch.isfinite(grid)).any(-1).sum()),
                grid_examples=flat[(~torch.isfinite(flat)).any(-1)][:8].cpu().tolist(),
                output_indices=ix.cpu().tolist()))
        return result
    F.grid_sample=sample
    try:
        for i,d in enumerate(dirs):
            grids.clear()
            canvas,den=sf.paste_perspective_to_erp_rectangle(canvas,rgb,d[None],(80,80),
                panorama_cnt=den,return_cnt=True,temperature=.1)
            bad=~torch.isfinite(den);count=int(bad.sum())
            if count>prior or grids:
                locations=torch.nonzero(bad[0,0,0])[:12].cpu().tolist()
                records.append(dict(index=i,canonical_yaw=cams[i].yaw,canonical_pitch=cams[i].pitch,
                    runtime_direction=d.cpu().tolist(),cumulative_nonfinite=count,
                    erp_pixel_examples_yx=locations,grid_sample=list(grids)))
            prior=count
    finally:F.grid_sample=original
    record=dict(job=os.environ['SLURM_JOB_ID'],geometry_identity=geometry_identity(),
        nonfinite_erp_pixels=int((~torch.isfinite(den)).sum()),zero_denominator=int((den==0).sum()),
        examples=records,original_helper_unchanged=True,
        decision='Block SphereDiff old70 only; preserve original division/grid_sample behavior and exact prescribed cameras')
    atomic(ROOT/'layouts/old_n70-spherediff-failure.json',record)
    emit('SPHEREDIFF_SUPPORT_FAILURE_ISOLATED',**record)
if __name__=='__main__':main()
