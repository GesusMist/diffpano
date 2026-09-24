"""Bounded, separately seeded exact-marginal noise statistics (no model calls)."""
import math
import torch
from diffpano.erp_noise_initialization import native_cameras
from scripts.noise_v_preflight import matched_ray_pairs

OFFSETS={'h':(0,1),'v':(1,0),'diag':(1,1),'h2':(0,2),'v2':(2,0),'h4':(0,4),'v4':(4,0)}
METHODS=('direct','nearest','gwtf')


def camera_pairs(cams):
    def ring(p):return [i for i,c in enumerate(cams) if abs(math.degrees(c.pitch)-p)<1e-4]
    def near(ids):return min(ids,key=lambda i:abs(cams[i].yaw))
    def adjacent(ids):
        ids=sorted(ids,key=lambda i:cams[i].yaw);i=ids.index(near(ids));return ids[i],ids[(i+1)%len(ids)]
    return {'equatorial_equatorial':adjacent(ring(0)),
            'equatorial_upper':(near(ring(0)),near(ring(22.5))),
            'upper_upper':adjacent(ring(45)), 'lower_lower':adjacent(ring(-45)),
            'north_polar_rotation':adjacent(ring(90)), 'south_polar_rotation':adjacent(ring(-90))}


def correspondence(cams):
    return {k:dict(slots=(i,j),indices=matched_ray_pairs(cams[i],cams[j],samples=12)) for k,(i,j) in camera_pairs(cams).items()}


def spatial_pairs(cam,slot,count=48):
    g=torch.Generator().manual_seed(650001+slot)
    result={}
    for label,(dy,dx) in OFFSETS.items():
        y=torch.randint(cam.height-dy,(count,),generator=g);x=torch.randint(cam.width-dx,(count,),generator=g)
        a=y*cam.width+x;b=(y+dy)*cam.width+x+dx;result[label]=(a,b)
    a=torch.randint(cam.height*cam.width,(count,),generator=g)
    delta=torch.randint(1,cam.height*cam.width,(count,),generator=g)
    result['random']=(a,(a+delta)%(cam.height*cam.width))
    extra=torch.randint(cam.height*cam.width,(256,),generator=g)
    targets=torch.unique(torch.cat([extra]+[x for pair in result.values() for x in pair]),sorted=True)
    return result,targets


class PairStats:
    def __init__(self,count):
        self.n=0;self.sums=torch.zeros(6,count,dtype=torch.float64);self.blocks=[]
    def add(self,x,y):
        # [realizations, actual native channels, matched pairs]
        a=x.double().reshape(-1,x.shape[-1]);b=y.double().reshape(-1,y.shape[-1])
        self.n+=a.shape[0]
        self.sums+=torch.stack((a.sum(0),b.sum(0),a.square().sum(0),b.square().sum(0),(a*b).sum(0),(a==b).sum(0)))
        self.blocks.append((x.double()*y.double()).mean((1,2)))
    def finish(self):
        x,y,x2,y2,xy,eq=self.sums/self.n
        cov=xy-x*y;vx=x2-x*x;vy=y2-y*y
        corr=cov/(vx*vy).sqrt().clamp_min(1e-20)
        blocks=torch.cat(self.blocks)
        return dict(covariance=float(cov.mean()),correlation=float(corr.mean()),mean_absolute_pair_covariance=float(cov.abs().mean()),
            exact_equality=float(eq.mean()),pairs=x.numel(),samples_per_pair=self.n,
            standard_error=float(blocks.std(unbiased=True)/math.sqrt(len(blocks))),
            covariance_per_pair=cov.tolist(),correlation_per_pair=corr.tolist())


class Moments:
    def __init__(self):self.n=0;self.sums=torch.zeros(4,dtype=torch.float64);self.blocks=[];self.sample=[]
    def add(self,x):
        z=x.double();self.n+=z.numel()
        self.sums+=torch.stack([z.pow(k).sum() for k in range(1,5)])
        self.blocks.append(torch.stack((z.mean((1,2)),z.square().mean((1,2))),1))
        if not self.sample:self.sample=[z.flatten()[:512].float().clone()]
    def finish(self):
        a,b,c,d=(self.sums/self.n).tolist();var=b-a*a;std=var**.5
        blocks=torch.cat(self.blocks);errors=blocks.std(0,unbiased=True)/(len(blocks)**.5)
        return dict(mean=a,variance=var,std=std,skewness=(c-3*a*b+2*a**3)/std**3,
            excess_kurtosis=(d-4*a*c+6*a*a*b-3*a**4)/var**2-3,
            mean_se=float(errors[0]),variance_se=float(errors[1]),count=self.n),self.sample[0]


def aggregate_camera_stats(rows):
    out={}
    for method in METHODS:
        items=[r['methods'][method] for r in rows];marg=[r['marginal'] for r in items]
        summary={k:sum(x[k] for x in marg)/len(marg) for k in ('mean','variance','std','skewness','excess_kurtosis')}
        # MC source draws are independent per camera; cross-view law tested separately.
        for key in ('mean_se','variance_se'):
            summary[key]=(sum(x[key]**2 for x in marg)**.5)/len(marg)
        summary['offsets']={}
        for key in (*OFFSETS,'random'):
            vals=[r['offsets'][key] for r in items]
            summary['offsets'][key]={k:sum(v[k] for v in vals)/len(vals) for k in ('covariance','correlation','mean_absolute_pair_covariance','exact_equality')}
            summary['offsets'][key]['standard_error']=(sum(v['standard_error']**2 for v in vals)**.5)/len(vals)
            summary['offsets'][key]['observations']=sum(v['pairs']*v['samples_per_pair'] for v in vals)
        summary['moran_equivalent']=(summary['offsets']['h']['covariance']+summary['offsets']['v']['covariance'])/(2*summary['variance'])
        samples=torch.tensor([v for r in rows for v in r['methods'][method]['normality_sample']]).sort().values.double()
        normal=.5*(1+torch.erf(samples/(2**.5)));rank=torch.arange(1,len(samples)+1,dtype=torch.float64)/len(samples)
        summary['normality_KS_effect']=float(torch.maximum((rank-normal).abs(),(normal-(rank-1/len(samples))).abs()).max())
        summary['quantiles']={str(p):float(torch.quantile(samples,p)) for p in (.001,.01,.05,.5,.95,.99,.999)}
        summary['diagnostic_sample_min']=float(samples.min());summary['diagnostic_sample_max']=float(samples.max())
        out[method]=summary
    return out


def state_statistics(states,backend,cameras):
    """Statistics of the exact scientific realization; no RNG or model calls."""
    rows=[];sigma=float(backend.native_initial_noise_sigma)
    for slot,s in enumerate(states):
        z=s.double()/sigma;mean=float(z.mean());var=float(z.var(unbiased=False));row=dict(slot=slot,mean=mean,std=var**.5,variance=var,offsets={})
        for key,(dy,dx) in OFFSETS.items():
            a=z[...,:z.shape[-2]-dy if dy else None,:z.shape[-1]-dx if dx else None]
            b=z[...,dy:,dx:]
            row['offsets'][key]=dict(covariance=float((a*b).mean()-a.mean()*b.mean()),exact_equality=float((a==b).double().mean()))
        row['moran_equivalent']=(row['offsets']['h']['covariance']+row['offsets']['v']['covariance'])/(2*var)
        rows.append(row)
    cams=native_cameras(backend,cameras);cross={}
    for label,record in correspondence(cams).items():
        i,j=record['slots'];ia,ib,angle=record['indices']
        a=states[i].flatten(2)[...,ia].double().flatten()/sigma;b=states[j].flatten(2)[...,ib].double().flatten()/sigma
        cov=float((a*b).mean()-a.mean()*b.mean())
        cross[label]=dict(slots=[i,j],covariance=cov,correlation=cov/float(a.std(unbiased=False)*b.std(unbiased=False)),
            angular_separation_mean_deg=float(angle.mean()),pairs=len(ia))
    result={key:sum(r[key] for r in rows)/len(rows) for key in ('mean','variance','std','moran_equivalent')}
    result['offsets']={key:{v:sum(r['offsets'][key][v] for r in rows)/len(rows) for v in ('covariance','exact_equality')} for key in OFFSETS}
    result.update(per_camera=rows,cross_view=cross,matched_ray_correlation=sum(v['correlation'] for v in cross.values())/len(cross),
        convention='Unscaled epsilon; population-centred adjacent covariance/variance is Moran-equivalent, not finite-sample centred Moran test statistic')
    return result
