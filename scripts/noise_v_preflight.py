"""Model-free map/covariance preflight. Streams all cameras; saves no maps/noise."""
import hashlib
import math
import os
import resource
import time
from dataclasses import asdict,replace
import torch
from diffpano.camera import spherediff_camera_cover
from diffpano.consensus_audit import camera_slots
from diffpano.erp_local_consensus import camera_digest
from diffpano.erp_noise_initialization import NearestERPIndexProjector, native_cameras, primary_noise_size, map_statistics
from diffpano.projection import perspective_world_rays
from scripts.noise_v_common import ROOT,s_config,shape_backend,read,require_validation


def representative_slots(cameras):
    selected=camera_slots(cameras)
    selected += [min(range(len(cameras)),key=lambda i:abs(cameras[i].pitch+math.pi/4)+abs(cameras[i].yaw)*.01),
                 min(range(len(cameras)),key=lambda i:cameras[i].pitch)]
    return list(dict.fromkeys(selected))


def matched_ray_pairs(a,b,samples=24):
    """Map sampled world rays to the closest of a 3x3 neighborhood in camera b.

    Angular separation is measured from actual unit world directions, never
    inferred by matching array positions. No full pairwise ray matrix is built.
    """
    ys=torch.linspace(0,a.height-1,min(samples,a.height)).round().long()
    xs=torch.linspace(0,a.width-1,min(samples,a.width)).round().long()
    y,x=torch.meshgrid(ys,xs,indexing='ij');indices_a=(y*a.width+x).flatten()
    rays=perspective_world_rays(a,device=torch.device('cpu')).reshape(-1,3)[indices_a]
    local=rays@b.rotation();z=local[:,2]
    u=(local[:,0]/z/math.tan(math.radians(b.fov_x)/2)+1)*b.width/2-.5
    v=(1-local[:,1]/z/math.tan(math.radians(b.fov_y)/2))*b.height/2-.5
    valid=(z>0)&(u>=-.5)&(u<b.width-.5)&(v>=-.5)&(v<b.height-.5)
    indices_a=indices_a[valid];rays=rays[valid];u=u[valid];v=v[valid]
    if not len(indices_a):raise AssertionError('Selected cameras do not overlap')
    bx=u.round().long();by=v.round().long();candidates=[];scores=[]
    brays=perspective_world_rays(b,device=torch.device('cpu')).reshape(-1,3)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            ids=(by+dy).clamp(0,b.height-1)*b.width+(bx+dx).clamp(0,b.width-1)
            candidates.append(ids);scores.append((rays*brays[ids]).sum(-1))
    choices=torch.stack(scores).argmax(0)
    indices_b=torch.stack(candidates).gather(0,choices[None])[0]
    # FP64 angle avoids a float32 acos floor when nearly identical rays match.
    ra=rays.double();rb=brays[indices_b].double();ra=ra/ra.norm(dim=-1,keepdim=True);rb=rb/rb.norm(dim=-1,keepdim=True)
    angle=torch.rad2deg(torch.acos((ra*rb).sum(-1).clamp(-1,1)))
    return indices_a,indices_b,angle


def covariance_check(left,right,*,independent=False,cross_channel=False,draws=8192,seed=52017):
    # Select bounded diagnostic pairs; draw only source cells they reference.
    select=torch.linspace(0,left.numel()-1,min(128,left.numel())).round().long()
    left=left.flatten()[select];right=right.flatten()[select]
    union,inverse=torch.unique(torch.cat((left,right)),return_inverse=True)
    n=left.numel();a,b=inverse[:n],inverse[n:]
    generator=torch.Generator(device='cpu').manual_seed(seed)
    source=torch.randn(draws,union.numel(),generator=generator)
    other=torch.randn(draws,union.numel(),generator=generator) if independent or cross_channel else source
    x=source[:,a];y=other[:,b]
    product=(x*y).mean(1);expected=0. if independent or cross_channel else float((left==right).float().mean())
    se=float(product.std(unbiased=True))/math.sqrt(draws)
    covariance=float(product.mean()-x.mean()*y.mean())
    tolerance=6*se+.002
    if abs(covariance-expected)>tolerance:raise AssertionError('Monte Carlo covariance disagrees with index-equality expectation')
    for values in (x,y):
        means=values.mean(1);squares=values.square().mean(1)
        if abs(float(means.mean()))>6*float(means.std())/math.sqrt(draws)+.002:raise AssertionError('Gaussian mean check')
        if abs(float(squares.mean())-1)>6*float(squares.std())/math.sqrt(draws)+.002:raise AssertionError('Gaussian variance check')
    return dict(pairs=n,draws=draws,distinct_source_cells=union.numel(),mean_left=float(x.mean()),mean_right=float(y.mean()),
                variance_left=float(x.var(unbiased=False)),variance_right=float(y.var(unbiased=False)),
                covariance=covariance,expected_covariance=expected,standard_error=se,tolerance=tolerance,
                independent_fields=independent,independent_channels=cross_channel,passed=True)


def run():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Full geometry preflight requires a compute allocation')
    require_validation();torch.set_num_threads(4);started=time.perf_counter();results={}
    for name in ('pixeldit','flux'):
        c,s=s_config(name);b,provenance=shape_backend(name);cameras=spherediff_camera_cover(c.view)
        assert camera_digest(cameras)==s['camera_sha256']
        cams=native_cameras(b,cameras);primary=primary_noise_size(cams);slots=representative_slots(cameras)
        pairs=[]
        for i in slots:
            j=max((j for j in range(len(cameras)) if j!=i),key=lambda j:float(cameras[i].forward()@cameras[j].forward()))
            pair=tuple(sorted((i,j)))
            if pair not in pairs:pairs.append(pair)
        selected=set(slots)|{i for pair in pairs for i in pair}
        records=[]
        for factor,h in [('half',math.ceil(primary[0]/2)),('primary',primary[0]),('double',primary[0]*2)]:
            projector=NearestERPIndexProjector(h,2*h);maps={};stats=[];digests=[]
            for i,cam in enumerate(cams):
                index=projector.index_map(cam);digests.append(hashlib.sha256(index.numpy().tobytes()).hexdigest())
                stats.append(dict(slot=i,**map_statistics(index)))
                if i in selected:maps[i]=index
            pair_records=[];lefts=[];rights=[]
            for i,j in pairs:
                ia,ib,angle=matched_ray_pairs(cams[i],cams[j]);a=maps[i].flatten()[ia];z=maps[j].flatten()[ib]
                lefts.append(a);rights.append(z)
                pair_records.append(dict(slots=[i,j],matched_rays=ia.numel(),angular_separation_degrees=dict(mean=float(angle.mean()),p95=float(torch.quantile(angle,.95)),max=float(angle.max())),
                                         same_source_fraction=float((a==z).float().mean())))
            a=torch.cat(lefts);z=torch.cat(rights)
            first=maps[slots[0]];hl=first[:,:-1].flatten();hr=first[:,1:].flatten()
            vl=first[:-1].flatten();vr=first[1:].flatten()
            mc=dict(horizontal=covariance_check(hl,hr),vertical=covariance_check(vl,vr),
                    cross_view_shared=covariance_check(a,z),cross_view_independent=covariance_check(a,z,independent=True),
                    cross_channel=covariance_check(a,a,cross_channel=True),same_cell=covariance_check(a,a))
            aggregate={k:dict(min=min(s[k] for s in stats),mean=sum(s[k] for s in stats)/len(stats),max=max(s[k] for s in stats)) for k in stats[0] if k!='slot'}
            records.append(dict(scale=factor,noise_size=[h,2*h],native_channels=b.native_channels,
                                native_local_shapes=sorted({(cam.height,cam.width) for cam in cams}),
                                source_field_bytes=c.generation.batch_size*b.native_channels*h*2*h*4,
                                coordinate_field_bytes=2*(h+2)*(2*h+2)*4,
                                maps_sha256=hashlib.sha256(''.join(digests).encode()).hexdigest(),
                                aggregate=aggregate,per_camera=stats,overlapping_pairs=pair_records,
                                matched_source_agreement=float((a==z).float().mean()),monte_carlo=mc))
            del projector,maps,index
            print('preflight',name,factor,'size',h,2*h,'duplicates',aggregate['duplicate_source_fraction'],'shared agreement',records[-1]['matched_source_agreement'],flush=True)
        results[name]=dict(primary_noise_size=list(primary),native_channels=b.native_channels,
                            native_scale=float(b.native_initial_noise_sigma),native_spatial_factor=b.native_spatial_factor,
                            camera_sha256=camera_digest(cameras),camera_poses=[asdict(x) for x in cameras],
                            representative_slots=slots,clean_rgb_erp_size=[c.erp.height,c.erp.width],provenance=provenance,records=records)
    result=dict(experiment='V',completed=True,passed=True,backends=results,
                formula='H=ceil(pi*max_camera(max(fx_native,fy_native)))); W=2H; half H=ceil(primary_H/2), double H=2*primary_H',
                rng='Independent dedicated CPU generators; diagnostics sample only referenced cells, seed 52017, 8192 draws',
                covariance='within-view A_i A_i^T identical for independent/shared; cross-view shared A_i A_j^T, independent zero',
                correspondence='World-ray projection followed by minimum angular separation among the 3x3 neighboring native pixel centers; report measured angular separation',
                model_calls=0,vae_calls=0,job=os.environ['SLURM_JOB_ID'],runtime_seconds=time.perf_counter()-started,
                host_max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    import json
    path=ROOT/'initialization-preflight.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':run()
