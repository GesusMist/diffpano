"""All-camera, full-native-geometry statistical gate, with streamed MC draws."""
import argparse
import gc
import math
import os
import resource
import time
import torch
from diffpano.erp_noise_initialization import (ERPNoiseConfig,initialize_erp_noise,NearestERPIndexProjector,native_cameras,states_digest)
from diffpano.gwtf_noise_initialization import GWTFlowNoiseConfig,initialize_gwtf_shared_noise,camera_graph,transport_values
from studies.gwtf_noise.common import *
from studies.gwtf_noise.statistics import *

DRAWS=512
BATCH=16


def mc_camera(graph,nearest,pairs,targets,channels,slot):
    plan,ids=graph.select(targets)
    nearest_ids=nearest.flatten()[ids]
    union=torch.unique(torch.cat((plan.source_ids,nearest_ids)),sorted=True)
    qpos=torch.searchsorted(union,plan.source_ids);npos=torch.searchsorted(union,nearest_ids)
    locations={k:(torch.searchsorted(ids,a),torch.searchsorted(ids,b)) for k,(a,b) in pairs.items()}
    moments={m:Moments() for m in METHODS}
    counters={m:{k:PairStats(len(a)) for k,(a,b) in pairs.items()} for m in METHODS}
    generators=[torch.Generator().manual_seed(880001+10007*slot+i*10000019) for i in range(3)]
    for first in range(0,DRAWS,BATCH):
        n=min(BATCH,DRAWS-first)
        q=torch.randn(n,channels,len(union),generator=generators[0])
        values=dict(direct=torch.randn(n,channels,len(ids),generator=generators[1]),nearest=q[...,npos],
            gwtf=transport_values(q[...,qpos],plan,generators[2],channel_chunk=128))
        for m,x in values.items():
            moments[m].add(x)
            for k,(a,b) in locations.items():counters[m][k].add(x[...,a],x[...,b])
    out={}
    for m in METHODS:
        marginal,sample=moments[m].finish()
        out[m]=dict(marginal=marginal,normality_sample=sample.tolist(),offsets={k:v.finish() for k,v in counters[m].items()})
    return out


def mc_cross(left,right,li,ri,ln,rn,channels,seed):
    a,idsa=left;b,idsb=right
    union=torch.unique(torch.cat((a.source_ids,b.source_ids,ln,rn)),sorted=True)
    posa=torch.searchsorted(union,a.source_ids);posb=torch.searchsorted(union,b.source_ids)
    npa=torch.searchsorted(union,ln);npb=torch.searchsorted(union,rn)
    ia=torch.searchsorted(idsa,li);ib=torch.searchsorted(idsb,ri)
    counters={m:PairStats(len(li)) for m in METHODS}
    gens=[torch.Generator().manual_seed(seed+i*10000019) for i in range(4)]
    for first in range(0,DRAWS,BATCH):
        n=min(BATCH,DRAWS-first);q=torch.randn(n,channels,len(union),generator=gens[0])
        va=transport_values(q[...,posa],a,gens[1],channel_chunk=128)[...,ia]
        vb=transport_values(q[...,posb],b,gens[2],channel_chunk=128)[...,ib]
        counters['gwtf'].add(va,vb);counters['nearest'].add(q[...,npa],q[...,npb])
        da=torch.randn(n,channels,len(idsa),generator=gens[3])[...,ia]
        db=torch.randn(n,channels,len(idsb),generator=gens[3])[...,ib]
        counters['direct'].add(da,db)
    return {m:v.finish() for m,v in counters.items()}


def statistical_gate(summary,cross):
    iid=summary['direct'];new=summary['gwtf'];old=summary['nearest'];checks={}
    for key,se,target in (('mean','mean_se',0.),('variance','variance_se',1.)):
        tol=abs(iid[key]-target)+6*max(iid[se],new[se])+1e-5
        checks[key]=dict(passed=abs(new[key]-target)<=tol,tolerance=tol,value=new[key],iid=iid[key])
    for key in OFFSETS:
        a=iid['offsets'][key];b=new['offsets'][key]
        tol=abs(a['covariance'])+6*max(a['standard_error'],b['standard_error'])+1e-5
        checks['cov_'+key]=dict(passed=abs(b['covariance'])<=tol,tolerance=tol,value=b['covariance'],iid=a['covariance'])
        # Exact FP32 equality: small binomial upper allowance calibrated to iid.
        n=b['observations'];tol=a['exact_equality']+6*math.sqrt(max(a['exact_equality'],1/n)/n)+1/n
        checks['duplicates_'+key]=dict(passed=b['exact_equality']<=tol,tolerance=tol,value=b['exact_equality'])
    old_strength=max(abs(old['offsets'][k]['covariance']) for k in ('h','v','diag'))
    new_strength=max(abs(new['offsets'][k]['covariance']) for k in ('h','v','diag'))
    checks['covariance_reduction']=dict(passed=new_strength<.2*old_strength,old=old_strength,new=new_strength,minimum_reduction_fraction=.8)
    old_dups=max(old['offsets'][k]['exact_equality'] for k in ('h','v','diag'))
    new_dups=max(new['offsets'][k]['exact_equality'] for k in ('h','v','diag'))
    checks['duplicate_reduction']=dict(passed=new_dups<.1*old_dups,old=old_dups,new=new_dups,minimum_reduction_fraction=.9)
    for label,row in cross.items():
        value=row['methods']['gwtf'];lower=value['covariance']-6*value['standard_error']
        checks['cross_'+label]=dict(passed=lower>0,covariance=value['covariance'],six_se_lower=lower)
    return dict(passed=all(v['passed'] for v in checks.values()),checks=checks,
        policy='Predeclared: iid-calibrated six-SE marginal/offset envelope + 1e-5 FP32 allowance; >=80% largest-neighbor covariance reduction; >=90% duplicate reduction; positive six-SE cross-covariance lower bounds. No image-based tuning.')


def run(name):
    require_validation();torch.set_num_threads(4);start=time.perf_counter()
    c,b,cams,old,evidence=backend_geometry(name);nc=native_cameras(b,cams);h,w=old['noise_grid']
    projector=NearestERPIndexProjector(h,w);matches=correspondence(nc);plans={};rows=[]
    for slot,cam in enumerate(nc):
        graph=camera_graph(cam,h,w,projector);nearest=projector.index_map(cam)
        pairs,targets=spatial_pairs(cam,slot)
        record=dict(slot=slot,geometry=graph.summary(),methods=mc_camera(graph,nearest,pairs,targets,b.native_channels,slot))
        rows.append(record)
        for label,item in matches.items():
            i,j=item['slots'];ia,ib,_=item['indices']
            if slot in (i,j):
                ids=ia if slot==i else ib
                plans[(label,slot)]=(graph.select(ids),nearest.flatten()[ids])
        del graph,nearest
        if slot%10==0:print(name,'MC camera',slot+1,'/89',flush=True)
    del projector
    cross={}
    for number,(label,item) in enumerate(matches.items()):
        i,j=item['slots'];ia,ib,angle=item['indices'];a,ln=plans[(label,i)];z,rn=plans[(label,j)]
        cross[label]=dict(slots=[i,j],angular_separation_degrees=angle.tolist(),
            methods=mc_cross(a,z,ia,ib,ln,rn,b.native_channels,910003+number))
    del plans
    summary=aggregate_camera_stats(rows)
    for method in METHODS:
        summary[method]['matched_ray_correlation']=sum(r['methods'][method]['correlation'] for r in cross.values())/len(cross)
        summary[method]['matched_ray_covariance']=sum(r['methods'][method]['covariance'] for r in cross.values())/len(cross)
    gate=statistical_gate(summary,cross)
    print(name,'MC gate',gate['passed'],flush=True)
    # Full exact seed-0 realizations, including all 89 cameras, for run identity/cost.
    exact={};rng_before=torch.random.get_rng_state().clone()
    for method in METHODS:
        t=time.perf_counter()
        if method=='gwtf':
            states,record=initialize_gwtf_shared_noise(b,cams,GWTFlowNoiseConfig(h,w,0),camera_slots=list(range(89)))
        else:
            states,record=initialize_erp_noise(b,cams,ERPNoiseConfig('S-direct-local' if method=='direct' else 'V-shared-erp',h,w,0))
        seconds=time.perf_counter()-t
        exact[method]=dict(initialization_seconds=seconds,record=record,statistics=state_statistics(states,b,cams),
            host_process_max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            persistent_state_bytes=sum(s.numel()*s.element_size() for s in states))
        del states;gc.collect()
    assert exact['nearest']['record']['initial_local_sha256']==old['initialization']['initial_local_sha256']
    replays=[]
    for order in (list(reversed(range(89))),torch.randperm(89,generator=torch.Generator().manual_seed(719)).tolist()):
        s,record=initialize_gwtf_shared_noise(b,cams,GWTFlowNoiseConfig(h,w,0),camera_slots=list(range(89)),execution_order=order)
        assert record['initial_local_sha256']==exact['gwtf']['record']['initial_local_sha256']
        replays.append(record['initial_local_sha256']);del s
    assert torch.equal(rng_before,torch.random.get_rng_state())
    result=dict(backend=name,passed=gate['passed'],gate=gate,summary=summary,per_camera=rows,cross_view=cross,exact_seed0=exact,
        draws=DRAWS,batch=BATCH,channels=b.native_channels,source_size=[h,w],native_shape=old['local_native_resolution'],
        camera_count=89,shape_provenance=evidence,order_replays=replays,rng_isolated=True,denoiser_calls=0,vae_calls=0,
        source_hashes=source_hashes(),job=os.environ['SLURM_JOB_ID'],seconds=time.perf_counter()-start,
        mc_policy='512 independent realizations per camera and category at actual channel count. Exact marginal subgraphs retain all outgoing sibling edges. MC RNG streams are separate from seed-0 scientific initialization.',
        memory_note='Process RSS peaks include Python, geometry, diagnostics and allocator retention; not exclusive allocation deltas per initializer.')
    write(ROOT/'statistical-preflight'/(name+'.json'),result)
    print('PREFLIGHT',name,'PASSED' if gate['passed'] else 'FAILED',summary,flush=True)
    if not gate['passed']:raise SystemExit(1)


def aggregate():
    require_validation();backends={name:read(ROOT/'statistical-preflight'/(name+'.json')) for name in BACKENDS}
    passed=all(v['passed'] and v['source_hashes']==source_hashes() for v in backends.values())
    write(ROOT/'preflight.json',dict(passed=passed,backends={name:dict(summary=v['summary'],gate=v['gate'],job=v['job'],seconds=v['seconds']) for name,v in backends.items()},source_hashes=source_hashes()))
    from studies.gwtf_noise.report import plot_noise
    plot_noise(backends)
    print('ALL-FIVE STATISTICAL GATE',passed,flush=True)
    if not passed:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=list(BACKENDS)+['aggregate']);a=p.parse_args()
    aggregate() if a.backend=='aggregate' else run(a.backend)
