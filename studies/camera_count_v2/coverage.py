"""Frozen finite-sample geometry and actual production RGB-support gates."""
import argparse
import gc
import math
import os
import time
import numpy as np
import torch
from studies.camera_patching.coverage import target_rays, ray_statistics, percentiles
from .common import *
from .cameras import build, layout_record, restore

PROBES = 1000000
HEIGHT, WIDTH = 2048,4096
CHUNK = 32768

def special_rays(device='cpu'):
    # Exact poles plus both representations/sides of the longitude wrap.
    lat=torch.linspace(-math.pi/2,math.pi/2,2049,device=device)
    angles=(-math.pi, math.pi, -math.pi+1e-6, math.pi-1e-6)
    parts=[torch.stack((lat.cos()*math.sin(a),lat.sin(),lat.cos()*math.cos(a)),-1) for a in angles]
    parts.append(torch.tensor([[0.,1.,0.],[0.,-1.,0.]],device=device))
    return torch.cat(parts)

@torch.no_grad()
def quick_pass(cameras,probes,device):
    rotations=torch.stack([c.rotation(torch.device(device)) for c in cameras])
    for start in range(0,probes,CHUNK):
        rays=target_rays('probes',start,min(probes,start+CHUNK),HEIGHT,WIDTH,probes,device)
        xyz=torch.einsum('rj,mji->rmi',rays,rotations)
        z=xyz[...,2];t=math.tan(math.radians(40))
        visible=(z>0)&(xyz[...,0].abs()<=z*t)&(xyz[...,1].abs()<=z*t)
        if not bool(visible.any(1).all()):return False
    return True

def summary(values,*,erp=False):
    count,total,maxw,dist=values
    good=np.isfinite(total)&np.isfinite(dist)
    d=dict(samples=len(count),uncovered=int(np.count_nonzero(count==0)),
           zero_weight=int(np.count_nonzero(total<=0)),finite=bool(good.all()),
           contributor_count=percentiles(count),total_center_weight=percentiles(total),
           maximum_single_view_weight=percentiles(maxw),nearest_center_degrees=percentiles(dist))
    d['passed']=d['uncovered']==d['zero_weight']==0 and d['finite']
    if erp:
        row_area=np.cos(np.arange(HEIGHT)*math.pi/HEIGHT)-np.cos(np.arange(1,HEIGHT+1)*math.pi/HEIGHT)
        row_area/=row_area.sum()
        d['spherical_area_weighted']={
            'uncovered_fraction':float(((count.reshape(HEIGHT,WIDTH)==0).mean(1)*row_area).sum()),
            'zero_weight_fraction':float(((total.reshape(HEIGHT,WIDTH)<=0).mean(1)*row_area).sum())}
        for name,a in zip(('contributor_count','total_weight','maximum_weight','nearest_center_degrees'),values):
            d['spherical_area_weighted'][name+'_mean']=float((a.reshape(HEIGHT,WIDTH).mean(1)*row_area).sum())
        d['percentile_convention']='texel percentiles; spherical-area weighted means and coverage fractions reported separately'
    return d

@torch.no_grad()
def scan(cameras,device):
    result={}
    for kind,total in [('probes',PROBES),('erp',HEIGHT*WIDTH),('special',8198)]:
        extras=special_rays(device) if kind=='special' else None
        if extras is not None:total=len(extras)
        values=[np.empty(total,dtype=np.float32) for _ in range(4)]
        bad=[]
        for start in range(0,total,CHUNK):
            stop=min(total,start+CHUNK)
            rays=extras[start:stop] if extras is not None else target_rays(kind,start,stop,HEIGHT,WIDTH,PROBES,device)
            stats=ray_statistics(cameras,rays)
            for dst,src in zip(values,stats):dst[start:stop]=src.cpu().numpy()
            if len(bad)<8:
                ix=torch.nonzero(stats[0]==0).flatten()[:8-len(bad)]
                bad.extend(dict(index=start+int(i),ray=rays[i].cpu().tolist()) for i in ix)
        result[kind]=summary(values,erp=kind=='erp');result[kind]['uncovered_examples']=bad
        emit('GEOMETRY_DOMAIN',domain=kind,summary=result[kind])
        del values
        if not result[kind]['passed']:return dict(passed=False,domains=result)
    return dict(passed=True,domains=result,guarantee='finite numerical tests, not continuous mathematical proof',
                probe_count=PROBES,wrap_and_poles=True)

@torch.no_grad()
def diffpano_support(cameras,device='cuda'):
    from studies.all_prompts.audit import configuration
    from studies.tt_cea.canvas import CanvasOperator,CanvasSpec
    from diffpano.projection import ProjectionCache
    c,*_=configuration('flux')
    op=CanvasOperator(CanvasSpec('erp',HEIGHT,WIDTH),c.warp,c.fusion,device,
                      cache=ProjectionCache(max_entries=1,cpu_fallback=False))
    acc=op.make_accumulator(1)
    view=torch.ones((1,3,1024,1024),device=device)
    for camera in cameras:op.accumulate(acc,view,camera)
    raw=acc['main'].ordinary_den
    v=dict(passed=bool((raw>0).all()) and bool(torch.isfinite(raw).all()),
           zero_denominator=int((raw<=0).sum()),minimum_denominator=float(raw.min()),
           weight_percentiles=percentiles(raw.flatten().cpu().numpy()),source_rgb_size=[1024,1024],
           arithmetic='actual CanvasOperator / StandardWarpOperator / RGBFusionAccumulator; bilinear weights, FP32',
           local_reprojection='unchanged nearest ERP sampling; full positive ERP support required')
    if v['passed']:
        final=op.finalize(acc);v['constant_reconstruction_error']=float((final.rgb-1).abs().max())
        assert v['constant_reconstruction_error']<2e-6
    op.clear_cache();del acc,op,raw,view;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return v

@torch.no_grad()
def spherediff_support(cameras,device='cuda'):
    from .spherediff_adapter import reference_geometry,directions,geometry_audit
    sf=reference_geometry()
    pose=geometry_audit(cameras,device)
    dirs=directions(cameras).to(device,dtype=torch.bfloat16).float()
    result=dict(adapter=pose,backends={})
    # SANA officially resizes decoded patches to 1024; FLUX's 59 packed
    # latent sites across decode to 59*16 = 944 RGB pixels.
    for backend,size in [('sana',1024),('flux',sf.get_height_width_from_fov((80,80),26500)[0]*16)]:
        rgb=torch.ones((1,1,1,size,size),device=device)
        canvas=torch.zeros((1,1,1,HEIGHT,WIDTH),device=device);den=torch.zeros_like(canvas)
        for d in dirs:
            canvas,den=sf.paste_perspective_to_erp_rectangle(canvas,rgb,d[None],(80,80),
                panorama_cnt=den,return_cnt=True,temperature=.1)
        passed=bool((den>0).all()) and bool(torch.isfinite(den).all())
        result['backends'][backend]=dict(passed=passed,zero_denominator=int((den<=0).sum()),
            minimum_denominator=float(den.min()),weight_percentiles=percentiles(den.flatten().cpu().numpy()),
            local_rgb_size=[size,size],dtype='BF16 direction cast to FP32 in official final paste',
            constant_reconstruction_error=float((canvas[den>0]/den[den>0]-1).abs().max()) if bool((den>0).any()) else None)
        del rgb,canvas,den;gc.collect();torch.cuda.empty_cache()
    result['passed']=all(v['passed'] for v in result['backends'].values()) and pose['passed']
    return result

def freeze_fixed(strategy,n,device='cuda'):
    if strategy not in ('old','fibonacci'):raise ValueError(strategy)
    path=layout_path(strategy,n)
    if path.exists():
        v=read(path);assert v['geometry_identity']==geometry_identity();return v
    cams=build(strategy,n);v=layout_record(strategy,n)
    geom=scan(cams,device)
    dp=diffpano_support(cams,device) if geom['passed'] else dict(passed=False,reason='geometric coverage failed')
    v['coverage']=dict(geometric=geom,diffpano=dict(passed=geom['passed'] and dp['passed'],runtime=dp))
    if strategy=='old':
        sd=spherediff_support(cams,device) if geom['passed'] else dict(passed=False,reason='geometric coverage failed')
        v['coverage']['spherediff']=sd
    v.update(geometry_identity=geometry_identity(),created=now(),job=os.environ['SLURM_JOB_ID'])
    immutable(path,v)
    emit('LAYOUT_FROZEN',path=str(path),coverage=v['coverage'])
    return v

def main():
    p=argparse.ArgumentParser();p.add_argument('--strategy',choices=('old','fibonacci'),required=True)
    p.add_argument('--camera-count',type=int,choices=COUNTS,required=True);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    assert_preserved()
    with lock('geometry-'+layout_key(a.strategy,a.camera_count)):freeze_fixed(a.strategy,a.camera_count)
if __name__=='__main__':main()
