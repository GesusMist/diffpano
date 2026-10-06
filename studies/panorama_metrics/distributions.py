"""Collection scores and strict matched-subset recomputation. No per-image FID."""
import functools
import numpy as np
from .common import *
from .features import gpu_identity
from .numerics import kid,inception_score

@functools.lru_cache(maxsize=600)
def load_feature(path,checksum):
    assert sha(path)==checksum
    with np.load(path,allow_pickle=False) as f:return dict(f)

@functools.lru_cache(maxsize=1)
def reference(identity,manifest_sha):
    m=read(ROOT/'reference_manifest.json');assert sha(ROOT/'reference_manifest.json')==manifest_sha
    records=[]
    for r in m['records']:
        k=digest(dict(image=r['sha256'],evaluator=identity));meta=CACHE/'reference_features'/(k+'.json')
        if not meta.exists():return None
        c=read(meta);assert c['protocol_sha256']==identity and c['image_sha256']==r['sha256']
        # Memory cache is reserved for generated features; real features are stacked once.
        assert sha(c['feature_path'])==c['feature_sha256']
        with np.load(c['feature_path'],allow_pickle=False) as f:records.append({n:f[n] for n in ('inception','omni_horizontal','omni_up','omni_down')})
    real={name:np.concatenate([r[name] for r in records],axis=0) if name=='inception' else np.stack([r[name] for r in records]) for name in records[0]}
    real['panorama_count']=len(records)
    stats_path=CACHE/'reference_stats'/(digest(dict(protocol=identity,manifest=manifest_sha))+'.npz')
    if stats_path.exists():
        with np.load(stats_path,allow_pickle=False) as f:real['stats']=dict(f)
    else:
        stats={}
        for name in ('inception','omni_horizontal','omni_up','omni_down'):
            a=real[name].astype(np.float64);stats[name+'_mean']=a.mean(0);stats[name+'_cov']=np.cov(a,rowvar=False)
        from .features import save_features
        save_features(stats_path,stats);real['stats']=stats
    return real

def fid_stats(x,mean,cov):
    x=np.asarray(x,dtype=np.float64)
    if len(x)<2:raise ValueError('FID needs two generated samples')
    xm=x.mean(0);a=(x-xm)/np.sqrt(len(x)-1)
    q=a@cov@a.T;q=(q+q.T)/2
    ev=np.linalg.eigvalsh(q);tol=1e-8*max(1.,float(np.max(np.abs(ev))))
    if ev.min() < -tol:raise ArithmeticError('Numerical non-PSD FID covariance product')
    result=float(np.sum((xm-mean)**2)+np.sum(a*a)+np.trace(cov)-2*np.sqrt(np.maximum(ev,0)).sum())
    if result < -1e-5:raise ArithmeticError('Negative FID beyond roundoff')
    return max(0.,result)

def enrich(group_table,paired,groups,gpu,rows,camera=None):
    identity=gpu_identity();real=None;manifest_sha=None
    done=ROOT/'reference_completion.json'
    if done.exists():
        d=read(done)
        if d['passed'] and d['protocol_sha256']==identity:
            manifest_sha=d['manifest_sha256'];real=reference(identity,manifest_sha)
    def eligible(rr):return [r for r in rr if r['completion_status']=='complete' and (r.get('image_sha256'),r['effective_prompt_sha256']) in gpu]
    def collection(rr):
        rr=eligible(rr)
        if not rr:return dict(IS_status='pending_features',FID_status='pending_features')
        f=[load_feature(gpu[(r['image_sha256'],r['effective_prompt_sha256'])]['feature_path'],gpu[(r['image_sha256'],r['effective_prompt_sha256'])]['feature_sha256']) for r in rr]
        count=len(rr);key=digest(dict(images=[r['image_sha256'] for r in rr],reference=manifest_sha,protocol=identity,numerics=sha(REPO/'studies/panorama_metrics/numerics.py'),aggregator=sha(REPO/'studies/panorama_metrics/distributions.py')))
        path=CACHE/'collections'/(key+'.json')
        if path.exists():return read(path)
        out=dict(panorama_count=count,rendered_view_count=count*8,feature_dimension=2048,
            small_sample_warning='21 or fewer panoramas; dependent views; rank-deficient generated covariance. Exploratory only.',
            generated_covariance_rank_upper_bound=min(count*8-1,2048),omni_generated_covariance_rank_upper_bound=min(count-1,2048))
        if 'logits' in f[0]:out['IS']=inception_score(np.concatenate([v['logits'] for v in f]));out['IS_status']='computed_collection'
        if real is None:
            for name in ('FID','KID','OmniFID'):out[name+'_status']='blocked_missing_reference_features'
        else:
            x=np.concatenate([v['inception'] for v in f]);st=real['stats']
            out['reference_panorama_count']=real['panorama_count'];out['reference_rendered_view_count']=len(real['inception'])
            out['FID']=fid_stats(x,st['inception_mean'],st['inception_cov']);out['FID_status']='computed_exploratory'
            k=kid(x,real['inception']);out['KID']=k['mean'];out.update({'KID_'+n:v for n,v in k.items() if n!='mean'});out['KID_status']='computed_exploratory'
            if count>=2:
                for n in ('horizontal','up','down'):
                    features=np.stack([v['omni_'+n] for v in f]);out['OmniFID_'+n]=fid_stats(features,st['omni_'+n+'_mean'],st['omni_'+n+'_cov'])
                out['OmniFID']=float(np.mean([out['OmniFID_'+n] for n in ('horizontal','up','down')]))
                out['OmniFID_status']='computed_exploratory'
            else:out['OmniFID_status']='blocked_sample_count_lt2'
        atomic(path,out);return out
    for g in group_table:g.update(collection(groups[g['group']]))
    # Each distribution comparison uses exactly the same panorama prompt IDs on both sides.
    pairs=sorted({(p['baseline'],p['comparison']) for p in paired})
    for a,b in pairs:
        aa={r['prompt_id']:r for r in eligible(groups.get(a,[]))};bb={r['prompt_id']:r for r in eligible(groups.get(b,[]))}
        common=sorted(set(aa)&set(bb))
        if not common:continue
        av=collection([aa[n] for n in common]);bv=collection([bb[n] for n in common])
        for metric in ('FID','KID','IS','OmniFID'):
            if metric in av and metric in bv:
                paired.append(dict(baseline=a,comparison=b,metric=metric,status='paired_collection',n=len(common),
                    prompts=';'.join(common),baseline_mean=av[metric],comparison_mean=bv[metric],mean=bv[metric]-av[metric],
                    delta_convention='difference between two collection scores on identical prompt sets; not per-image FID'))
    if camera is not None:
        cpairs=sorted({(r['baseline'],r['comparison'],r['common_prompts']) for r in camera if r['record_type']=='summary' and r.get('common_prompts')})
        for a,b,prompts in cpairs:
            common=prompts.split(';');aa={r['prompt_id']:r for r in eligible(groups.get(a,[]))};bb={r['prompt_id']:r for r in eligible(groups.get(b,[]))}
            if not all(n in aa and n in bb for n in common):continue
            av=collection([aa[n] for n in common]);bv=collection([bb[n] for n in common])
            for metric in ('FID','KID','IS','OmniFID'):
                if metric in av and metric in bv:
                    camera.append(dict(backend=a.split('/')[1],projection=a.split('/')[2],record_type='collection',
                        baseline=a,comparison=b,common_prompts=prompts,metric=metric,status='tiny_sample_exploratory',n=len(common),
                        baseline_mean=av[metric],comparison_mean=bv[metric],mean=bv[metric]-av[metric]))
