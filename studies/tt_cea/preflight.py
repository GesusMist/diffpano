"""Chunked model-free geometry and isolated actual-backend numerical gates."""
import math
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path
import torch
import torch.nn.functional as F
from diffpano.fusion import create_view_weight_map
from diffpano.geometry import erp_world_directions
from diffpano.projection import perspective_world_rays
from studies.tt_cea.common import *
from studies.tt_cea.cameras import load_cover
from studies.tt_cea.canvas import cea_rays,project_rays,CanvasOperator,CanvasSpec


def stats(count,weight,strongest,bands):
    def summary(mask):
        c=count[mask].float();w=weight[mask];s=strongest[mask]
        if not c.numel():return dict(samples=0)
        q=torch.tensor([.01,.05,.5,.95,.99])
        return dict(samples=c.numel(),uncovered_fraction=float((c==0).float().mean()),
            contributors=dict(min=int(c.min()),median=float(c.median()),max=int(c.max())),
            weight_sum=dict(min=float(w.min()),percentiles=torch.quantile(w,q).tolist(),max=float(w.max())),
            strongest_view_weight=dict(min=float(s.min()),median=float(s.median()),max=float(s.max())),
            finite_positive_denominators=bool(torch.isfinite(w).all() and (w>0).all()))
    result=summary(torch.ones(count.shape,dtype=torch.bool));result['percentile_levels']=[1,5,50,95,99]
    result['latitude_bins']={str(bounds):summary(bands==i) for i,bounds in enumerate(([-90,-60],[-60,-30],[-30,0],[0,30],[30,60],[60,90]))}
    result['passed']=result['uncovered_fraction']==0 and result['finite_positive_denominators']
    return result


def raster_chunks(projection,h,w,device):
    for start in range(0,h,64):
        stop=min(h,start+64)
        if projection=='cea':rays=cea_rays(h,w,device,row_start=start,row_stop=stop)
        else:
            y=torch.arange(start,stop,device=device,dtype=torch.float32)
            x=torch.arange(w,device=device,dtype=torch.float32)
            phi=(.5-(y+.5)/h)*math.pi;lon=((x+.5)/w-.5)*2*math.pi
            pp,ll=torch.meshgrid(phi,lon,indexing='ij')
            rays=torch.stack((pp.cos()*ll.sin(),pp.sin(),pp.cos()*ll.cos()),-1)
        yield rays.reshape(-1,1,3)


def coverage(cams,chunks,device):
    counts=[];weights=[];strong=[];bands=[]
    wm=create_view_weight_map(cams[0].height,cams[0].width,'spherediff_center',temperature=.1,device=device)
    for rays in chunks:
        n=rays.shape[0];count=torch.zeros(n,device=device,dtype=torch.int16)
        den=torch.zeros(n,device=device);maximum=torch.zeros_like(den)
        for camera in cams:
            grid,mask=project_rays(rays,camera)
            w=F.grid_sample(wm,grid,mode='bilinear',padding_mode='border',align_corners=False).flatten()*mask.flatten()
            count.add_((mask.flatten()>0).to(count.dtype));den.add_(w);maximum=torch.maximum(maximum,w)
        counts.append(count.cpu());weights.append(den.cpu());strong.append(maximum.cpu())
        latitude=torch.asin(rays[:,0,1].clamp(-1,1))*180/math.pi
        bands.append(torch.bucketize(latitude,latitude.new_tensor([-60,-30,0,30,60])).to(torch.int8).cpu())
    return stats(torch.cat(counts),torch.cat(weights),torch.cat(strong),torch.cat(bands))


def analytic(rays):
    # Independent continuous world-coordinate patterns: low frequency + great circle.
    x,y,z=rays.double().unbind(-1)
    return torch.stack((.45*x+.25*torch.sin(3*y),.5*y+.2*z,torch.exp(-30*(x+.3*z).square())-.5),0).float()[None]


def stress_tests(device,covers=('old89','ea89')):
    from PIL import Image,ImageDraw
    from diffpano.diagnostics import tensor_to_pil
    c,_,_,_=base_config('sd2','ruins');rows=[];sheet=Image.new('RGB',(4*256,4*156),'white');draw=ImageDraw.Draw(sheet)
    erp_truth=analytic(erp_world_directions(128,256,device=device))
    for row,(cover,projection) in enumerate((cover,projection) for cover in covers for projection in ('erp','cea')):
        cams,_=load_cover(cover,64,64);op=CanvasOperator(CanvasSpec(projection,128,256),c.warp,c.fusion,device)
        truth=[analytic(perspective_world_rays(v,device=device)).cpu() for v in cams]
        views=[v.clone() for v in truth];steps=[]
        sheet.paste(tensor_to_pil(erp_truth[0].cpu()),(0,row*156+24));draw.text((4,row*156+3),cover+' '+projection+' | truth',fill='black')
        for step in range(1,21):
            acc=op.make_accumulator(1)
            for v,cam in zip(views,cams):op.accumulate(acc,v.to(device),cam)
            canvas=op.finalize(acc);views=[op.sample_view(canvas,cam).cpu() for cam in cams]
            if step in (1,5,20):
                erp=op.export_erp(canvas,128,256);error=(erp-erp_truth).abs();rays=erp_world_directions(128,256,device=device)
                latitude=torch.asin(rays[...,1].clamp(-1,1))*180/math.pi
                steps.append(dict(round_trips=step,erp_mae=float(error.mean()),erp_max=float(error.max()),
                    poles_mae=float(error[:,:,latitude.abs()>=60].mean()),equator_mae=float(error[:,:,latitude.abs()<=30].mean()),
                    view_mae=sum(float((a-b).abs().mean()) for a,b in zip(views,truth))/89,
                    finite=bool(torch.isfinite(erp).all()),minimum_weight_sum=float(canvas.weight_sum.min())))
                col=(1,5,20).index(step)+1;sheet.paste(tensor_to_pil(erp[0].cpu()),(col*256,row*156+24))
                draw.text((col*256+4,row*156+3),'round trips '+str(step),fill='black')
        rows.append(dict(cover=cover,projection=projection,canvas_size=[128,256],view_size=[64,64],steps=steps));op.clear_cache()
    path=ROOT/'geometry/resampling-stress.png';sheet.save(path)
    result=dict(passed=all(s['finite'] and s['minimum_weight_sum']>0 for r in rows for s in r['steps']),rows=rows,
        figure=str(path),note='Reduced-raster model-free stress diagnostic; production coverage is measured separately at full raster. No CEA-superiority requirement.')
    atomic(ROOT/'geometry/operator_tests.json',result)
    return result


@torch.no_grad()
def geometry_phase():
    require_gate();assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    device=torch.device('cuda');started=time.perf_counter();records={}
    generator=torch.Generator(device='cpu').manual_seed(19570925)
    z=2*torch.rand(1000000,generator=generator)-1;lon=2*math.pi*torch.rand(1000000,generator=generator)-math.pi
    r=(1-z.square()).sqrt();probes=torch.stack((r*lon.sin(),z,r*lon.cos()),-1)
    lat=torch.linspace(-math.pi/2,math.pi/2,257)
    special=[torch.tensor([[0.,1.,0.],[0.,-1.,0.]])]
    for longitude in (-math.pi+1e-6,math.pi-1e-6):
        special.append(torch.stack((lat.cos()*math.sin(longitude),lat.sin(),lat.cos()*math.cos(longitude)),-1))
    special=torch.cat(special)
    for cover in ('old89','ea89'):
        record={}
        for raster in (512,1024):
            cams,_=load_cover(cover,raster,raster);entries={}
            for projection in ('erp','cea'):
                entries[projection]=coverage(cams,raster_chunks(projection,2048,4096,device),device)
                print('COVERAGE',cover,raster,projection,entries[projection]['passed'],flush=True)
            entries['independent_sphere']=coverage(cams,(v.to(device).reshape(-1,1,3) for v in probes.split(262144)),device)
            entries['poles_and_seam']=coverage(cams,[special.to(device).reshape(-1,1,3)],device)
            record[str(raster)]=entries
        records[cover]=dict(rasters=record,passed=all(v['passed'] for e in record.values() for v in e.values()))
    coverage_result=dict(passed=all(r['passed'] for r in records.values()),covers=records,
        probe_seed=19570925,independent_probe_count=1000000,canvas_size=[2048,4096],source_hashes=study_hashes(),
        job=os.environ['SLURM_JOB_ID'],seconds=time.perf_counter()-started)
    atomic(ROOT/'geometry/coverage.json',coverage_result)
    # A failed proposed cover blocks affected geometry cases, not the TT branch.
    if any(r['passed'] for r in records.values()):stress_tests(device,tuple(k for k,v in records.items() if v['passed']))
    else:atomic(ROOT/'geometry/operator_tests.json',dict(passed=False,reason='Cover failed production geometry; stress tests not run'))
    print('GEOMETRY PREFLIGHT RECORDED',coverage_result['passed'],flush=True)


def backend_phase(name):
    from studies.tt_cea.backend_preflight import run
    run(name)

if __name__=='__main__':
    p=parser(__doc__);p.add_argument('--phase',required=True,choices=['geometry','backend']);p.add_argument('--backend',choices=BACKENDS)
    a=p.parse_args();load_settings(a.config)
    if a.phase=='geometry':geometry_phase()
    elif a.backend:backend_phase(a.backend)
    else:p.error('--backend required')
