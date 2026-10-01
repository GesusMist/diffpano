"""Actual prepared endpoint coefficients, never inferred from progress ratios."""
from dataclasses import dataclass,asdict
import math
import torch
from diffpano.pipelines.endpoints import ddim_endpoints,flow_bounds
from scripts.dense_erp_experiment import canonical_schedule,schedule_metadata
from studies.tt_cea.common import digest

@dataclass(frozen=True)
class Interval:
    k:int
    model_timestep:float
    alpha_high:float
    sigma_high:float
    alpha_low:float
    sigma_low:float
    eligible:bool
    schedule_identity:str

def eligible_indices(n,enabled=True):
    if not isinstance(n,int) or n<1:raise ValueError('Positive integer interval count required')
    return tuple(k for k in range(n) if enabled and n<=5*k<4*n)

def execution_plan(intervals,enabled):
    return tuple((v,kind) for v in intervals for kind in (('initial','replay') if enabled and v.eligible else ('initial',)))

def prepare_interval_table(backend):
    identity=digest(canonical_schedule(schedule_metadata(backend)))
    n=len(backend.timesteps);eligible=set(eligible_indices(n));out=[]
    scheduler=getattr(getattr(backend,'pipeline',None),'scheduler',None)
    dummy=torch.zeros(1,1,1,1,device=backend.device,dtype=torch.float32)
    for k,t in enumerate(backend.timesteps):
        if scheduler is None:
            s,sn=backend.solver.bounds_for(t);s=s.to(dummy);sn=sn.to(dummy);a,an=1-s,1-sn
        elif type(scheduler).__name__=='DDIMScheduler':
            pair=ddim_endpoints(scheduler,dummy,dummy,t)
            a,s,an,sn=pair.alpha,pair.sigma,pair.next_alpha,pair.next_sigma
        else:
            s,sn=flow_bounds(scheduler,t,dummy);a,an=1-s,1-sn
        values=tuple(float(x) for x in (a,s,an,sn))
        if not all(math.isfinite(x) for x in values) or not (0<=values[3]<=values[1] and 0<=values[0]<=values[2] and values[2]>0):
            raise ValueError('Invalid prepared interval coefficients: '+str(values))
        out.append(Interval(k,float(t),*values,k in eligible,identity))
    for left,right in zip(out,out[1:]):
        if abs(left.alpha_low-right.alpha_high)>2e-7 or abs(left.sigma_low-right.sigma_high)>2e-7:
            raise ValueError('Prepared schedule has discontinuous endpoints: '+str((left,right)))
    return tuple(out)

def verify_prediction(pair,interval):
    values=(pair.alpha,pair.sigma,pair.next_alpha,pair.next_sigma)
    expected=(interval.alpha_high,interval.sigma_high,interval.alpha_low,interval.sigma_low)
    for a,b in zip(values,expected):
        if not math.isfinite(float(a)) or abs(float(a)-b)>2e-7:raise AssertionError('Returned model coefficients differ from interval')

def records(table):return [asdict(v) for v in table]
