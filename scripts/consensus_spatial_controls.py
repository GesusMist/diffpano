"""Q: coherent-input tests of frozen historical L/N/O operators; no models."""
import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from diffpano.camera import spherediff_camera_cover
from diffpano.config import load_experiment_config
from diffpano.consensus_audit import image_metrics, view_metrics, camera_slots, montage, thumbnail
from diffpano.dense_geometry import contributor_stats
from diffpano.erp_local_consensus import camera_digest
from diffpano.fusion import RGBFusionAccumulator
from diffpano.geometry import erp_world_directions
from diffpano.projection import ProjectionCache, perspective_world_rays
from diffpano.warp import StandardWarpOperator, LaplacianPyramidWarpOperator

ROOT=Path('outputs/vae-residual-controls/20260916-diagnostics-pt')
NATURAL=Path('outputs/vae-residual-controls/20260910-lookingglass-v1/sana/K/final_erp.png')


def require_validation():
    gate=json.loads((ROOT/'validation.json').read_text())
    if not gate['passed']:
        raise AssertionError('Full regression gate has not passed')
    for path,digest in gate['source_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise AssertionError('Source changed after validation: '+path)
    return gate


def signal(rays):
    """Continuous, pole-safe functions, evaluated at actual rays, never from ERP."""
    x,y,z=rays.unbind(-1)
    envelope=torch.exp(-4*y.square())
    waves=torch.stack([.35*torch.sin(k*math.pi*x)*envelope for k in (8,32,96)])
    smooth=torch.stack([.15*z,.15*y,.15*x])
    # Markers at the wrap and near the north pole, with different colors.
    wrap=torch.exp(-((x/.075).square()+(y/.075).square()))*(z<0)
    pole=torch.exp(-((x/.09).square()+(z/.09).square()))*(y>0)
    return (waves+smooth+torch.stack([.5*wrap,.5*pole,.25*wrap]))[None]


def assemble(operator,cameras,size,source,repeat=1):
    previous=torch.zeros(1,3,*size,device=source(cameras[0]).device)
    accumulator=operator.create_fusion_accumulator(previous)
    if accumulator is None:accumulator=RGBFusionAccumulator(previous,operator.fusion_config)
    for _ in range(repeat):
        for c in cameras:
            view=source(c)
            if isinstance(operator,LaplacianPyramidWarpOperator):accumulator.accumulate(view,c)
            else:accumulator.accumulate(operator.perspective_to_erp(view,c,size))
    return accumulator.finalize()


def frequency_retention(value):
    # Fit each known sinusoid jointly with its smooth/marker/DC nuisance terms.
    # Unlike total channel contrast, this measures the specified angular band.
    rays=erp_world_directions(*value.shape[-2:],device=value.device)
    x,y,z=rays.unbind(-1);envelope=torch.exp(-4*y.square())
    wrap=torch.exp(-((x/.075).square()+(y/.075).square()))*(z<0)
    pole=torch.exp(-((x/.09).square()+(z/.09).square()))*(y>0)
    area=torch.sqrt((1-y.square()).clamp_min(0)).flatten()
    result={}
    for channel,(k,coordinate,marker) in enumerate(zip((8,32,96),(z,y,x),(wrap,pole,wrap))):
        basis=torch.stack([torch.sin(k*math.pi*x)*envelope,coordinate,marker,torch.ones_like(x)]).flatten(1)
        weighted=basis*area[None]
        gram=(weighted@basis.T).double();rhs=(weighted@value[0,channel].flatten()).double()
        fit=torch.linalg.solve(gram,rhs)
        result[str(k)]=dict(fitted_amplitude=float(fit[0]),retention=float(fit[0]/.35),reference_amplitude=.35)
    return result


def marker_alignment(value,reference):
    rays=erp_world_directions(*value.shape[-2:],device=value.device)
    result={}
    for name,channel,direction in [('wrap',0,[0.,0.,-1.]),('north',1,[0.,1.,0.])]:
        d=rays.new_tensor(direction);cap=(rays*d).sum(-1)>math.cos(.16)
        def peak(image):
            index=int(torch.where(cap,image[0,channel],torch.full_like(image[0,channel],-float('inf'))).flatten().argmax())
            return rays.reshape(-1,3)[index]
        r=peak(reference);v=peak(value)
        result[name]=dict(peak_shift_degrees=float(torch.rad2deg(torch.acos((r*v).sum().clamp(-1,1)))),
                          reference_peak_offset_degrees=float(torch.rad2deg(torch.acos((r*d).sum().clamp(-1,1)))))
    return result


def metrics(value,reference):
    h,w=value.shape[-2:]
    result=image_metrics(value,reference,spherical=True)
    result['equator']=image_metrics(value[...,h//3:2*h//3,:],reference[...,h//3:2*h//3,:],spherical=False)
    result['polar']=dict(north=image_metrics(value[...,:h//6,:],reference[...,:h//6,:]),south=image_metrics(value[...,-h//6:,:],reference[...,-h//6:,:]))
    result['wrap_mae']=float((value-reference).abs()[...,[0,1,w-2,w-1]].mean())
    result['wrap_gradient']=float((value[...,0]-value[...,-1]).abs().mean())
    result['channel_retained_amplitudes']=[image_metrics(value[:,i:i+1],reference[:,i:i+1],spherical=True)['retained_amplitude'] for i in range(3)]
    return result


@torch.no_grad()
def run(device,quick=False):
    if not quick:require_validation()
    folder=ROOT/('Q-smoke' if quick else 'Q')
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True)
    started=time.perf_counter();records=[];figures={'synthetic':[],'sampled':[]};duplicates=[];geometry=[]
    heights=(64,128) if quick else (1024,2048)
    config=load_experiment_config('configs/experiments/erp_later/flux-l.yaml')
    if quick:config.view.height=config.view.width=64
    cameras=spherediff_camera_cover(config.view)
    poses=camera_digest(cameras);slots=camera_slots(cameras)
    sampled=torch.from_numpy(np.array(Image.open(NATURAL).convert('RGB'),copy=True)).permute(2,0,1)[None].float().to(device)/127.5-1 if NATURAL.exists() else None
    for h in heights:
        size=(h,2*h);rays=erp_world_directions(*size,device=device);reference=signal(rays)
        for label in ('L','N','O'):
            c=load_experiment_config('configs/experiments/erp_later/flux-'+label.lower()+'.yaml')
            historical_root=Path('outputs/vae-residual-controls')/('20260915-dense-lm' if label=='L' else '20260916-dense-no')
            historical=json.loads((historical_root/label/'flux/metadata.json').read_text())
            if c.to_dict()!=historical['config']:
                raise AssertionError('Q operator config differs from actual historical '+label)
            cache=ProjectionCache(max_entries=2,cpu_fallback=True)
            op=(LaplacianPyramidWarpOperator(c.warp,c.fusion,cache,periodic_reconstruction=True) if label=='O' else StandardWarpOperator(c.warp,c.fusion,cache))
            direct=lambda camera:signal(perspective_world_rays(camera,device=device))
            direct_result=assemble(op,cameras,size,direct)
            duplicated=assemble(op,cameras,size,direct,repeat=2)
            error=float((duplicated.erp_rgb-direct_result.erp_rgb).abs().max())
            duplicates.append(dict(operator=label,erp_size=size,max_abs=error,passed=error<2e-5))
            constant=assemble(op,cameras,size,lambda camera:torch.full((1,3,camera.height,camera.width),.3,device=device))
            constant_error=float((constant.erp_rgb-.3).abs().max())
            geometry.append(dict(operator=label,erp_size=list(size),view_size=[cameras[0].height,cameras[0].width],camera_sha256=poses,constant_max_error=constant_error,**contributor_stats(direct_result.contributor_count)))
            record=dict(operator=label,erp_size=size,warp=asdict(c.warp),fusion=asdict(c.fusion),direct=metrics(direct_result.erp_rgb,reference),direct_markers=marker_alignment(direct_result.erp_rgb,reference),direct_frequency=frequency_retention(direct_result.erp_rgb),single_camera=[])
            for i in slots:
                camera=cameras[i];view=direct(camera)
                single=assemble(op,[camera],size,lambda _:view)
                returned=op.erp_to_perspective(single.erp_rgb,camera)
                record['single_camera'].append(dict(slot=i,**view_metrics(returned,view)))
            record['cycles']={}
            for name,ref in [('synthetic',reference)]+([('sampled',torch.nn.functional.interpolate(sampled,size=size,mode='bilinear',align_corners=False))] if sampled is not None else []):
                value=ref.clone();cycle_records=[];cells=[('reference',thumbnail(ref,(384,192)))]
                for cycle in range(1,21):
                    value=assemble(op,cameras,size,lambda camera:op.erp_to_perspective(value,camera)).erp_rgb
                    if cycle in (1,5,20):
                        cycle_records.append(dict(cycle=cycle,**metrics(value,ref)))
                        if name=='synthetic':
                            cycle_records[-1]['markers']=marker_alignment(value,ref)
                            cycle_records[-1]['frequency']=frequency_retention(value)
                        cells.append((f'cycle {cycle}',thumbnail(value,(384,192))))
                        print('Q',label,size,name,'cycle',cycle,'rmse',cycle_records[-1]['rmse'],flush=True)
                record['cycles'][name]=cycle_records
                figures[name].append((f'{label} ERP {h}x{2*h}',cells))
            records.append(record)
            # Checkpoint only the single compact metrics artifact; no cycle tensors.
            payload=dict(experiment='Q',completed=len(records)==6,job=os.environ.get('SLURM_JOB_ID'),quick=quick,records=records,duplication=duplicates,
                geometry=geometry,camera_poses=[asdict(c) for c in cameras],
                signal='continuous Cartesian smooth components + localized sin(k*pi*x), k=8/32/96, wrap and polar Gaussian markers',
                sampling='Production maximum phase gradient 96*pi radians/radian is below ERP equatorial and local perspective Nyquist; quick smoke high bands intentionally alias and are not scientific results.',
                sampled_reference=str(NATURAL) if sampled is not None else None,
                sampled_reference_sha256=hashlib.sha256(NATURAL.read_bytes()).hexdigest() if sampled is not None else None,
                natural_reference_limitation='No photographic ERP asset found. Existing SANA K generated ERP is a sampled-image stress test, not geometric ground truth; its stitching is inherited.',
                runtime_seconds=time.perf_counter()-started,denoiser_calls=0,vae_calls=0,
                passed=all(d['passed'] for d in duplicates) and all(g['constant_max_error']<2e-5 and g['coverage_percent']==100 for g in geometry))
            (folder/'metrics.json').write_text(json.dumps(payload,indent=2)+'\n')
            del op,cache,constant,direct_result,duplicated,value
    for name,rows in figures.items():
        if rows:montage(rows,folder/(name+'.png'),cell=(384,192))
    if not payload['passed']:raise AssertionError('Q normalization/coverage gate failed; review before model runs')
    print('Q COMPLETED',folder,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',default='cuda');parser.add_argument('--quick',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2);run(torch.device(args.device),args.quick)
