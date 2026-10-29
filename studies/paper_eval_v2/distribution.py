"""Group 2 collection metrics and matched prompt-cluster bootstrap estimands."""
import argparse
import hashlib
import io
import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
from studies.panorama_metrics.common import atomic, digest, read, sha
from studies.panorama_metrics.numerics import frechet, kid
from studies.panorama_metrics.distributions import fid_stats
from .rendering import protocol_definitions as rendering_protocols
from .statistics import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, PRIMARY_COMPARISONS


def protocol_definitions():
    from studies.panorama_metrics.protocols import protocols
    legacy = protocols()
    return dict(group2=dict(feature_model=legacy['FID']['checkpoint'],
        feature_extractor='pinned torch-fidelity inception-v3-compat;2048;float32 inference,float64 statistics',
        FID_real=dict(protocol='common_v2',views='eight horizontal 90-degree 512-square views',reference='explicit frozen real manifest'),
        FID_native=dict(protocol='native_five_v2',reference='bank A primary, bank B sensitivity; exact scene/direction multiplicities'),
        KID=dict(kernel='(x dot y/2048 + 1)^3',subsets=100,subset_size=1000,seed=2020,
            units='raw MMD squared; retain finite negatives',subset_sd='not comparison uncertainty',
            dependence='ordinary independent-sample unbiasedness does not directly apply to correlated views'),
        OmniFID=dict(preprocess='common_v2 ERP -> PIL bilinear1024x512 -> py360convert e2c256 bilinear',
            grouping='average four horizontal features per panorama; FID(horizontal),FID(up),FID(down), then average three distances'),
        bootstrap=dict(unit='matched prompt cluster',replicates=BOOTSTRAP_REPLICATES,seed=BOOTSTRAP_SEED,
            reference_estimands=['fixed large real reference','jointly sampled condition-linked native reference'],
            incomplete='report achieved replicates and pending interval when allocated time is insufficient'),
        dimensions='retain full2048; exact low-rank covariance algebra; no PCA',
        interpretation='exploratory compact collections; more crops do not create more independent scenes'))


def collection_scores(generated, reference, native=False):
    key = 'native_five' if native else 'inception'
    x = np.concatenate([row[key] for row in generated]).astype(np.float64)
    y = np.concatenate([row[key] for row in reference]).astype(np.float64)
    score = dict(FID=frechet(x,y),KID=kid(x,y),feature_dimension=x.shape[1],
        generated_panorama_count=len(generated),generated_view_count=len(x),
        reference_panorama_count=len(reference),reference_view_count=len(y))
    if not native:
        parts = {name:frechet(np.stack([row['omni_'+name] for row in generated]),
                              np.stack([row['omni_'+name] for row in reference])) for name in ('horizontal','up','down')}
        score.update(OmniFID=float(np.mean(list(parts.values()))),OmniFID_components=parts)
    return score


class FixedReferenceFID:
    """Exact full-dimensional FID after arbitrary resampling of a small collection."""
    def __init__(self, generated, reference):
        self.x = np.asarray(generated,dtype=np.float64)
        y = np.asarray(reference,dtype=np.float64)
        self.mean = y.mean(0)
        covariance = np.cov(y,rowvar=False)
        self.trace = float(np.trace(covariance))
        self.kernel = self.x @ covariance @ self.x.T
        self.norms = np.sum(self.x*self.x,axis=1)

    def __call__(self, indices):
        indices = np.asarray(indices,dtype=np.int64)
        n = len(indices)
        if n<2:
            raise ValueError('FID requires at least two samples')
        mean = self.x[indices].mean(0)
        trace = (self.norms[indices].sum()-n*np.dot(mean,mean))/(n-1)
        matrix = self.kernel[np.ix_(indices,indices)]
        matrix = (matrix-matrix.mean(0)[None,:]-matrix.mean(1)[:,None]+matrix.mean())/(n-1)
        eigenvalues = np.linalg.eigvalsh((matrix+matrix.T)/2)
        if eigenvalues.min() < -1e-8*max(1.,np.abs(eigenvalues).max()):
            raise ArithmeticError('Non-PSD FID cross covariance')
        value = float(np.sum((mean-self.mean)**2)+trace+self.trace-2*np.sqrt(np.maximum(eigenvalues,0)).sum())
        if value < -1e-5:
            raise ArithmeticError('Negative FID beyond roundoff')
        return max(0.,value)


class FixedReferenceKID:
    """Cache exact frozen-subset kernels; resampling preserves whole prompt clusters."""
    def __init__(self, generated, reference, subsets=100, subset_size=1000, seed=2020):
        self.x = np.asarray(generated,dtype=np.float64)
        y = np.asarray(reference,dtype=np.float64)
        self.n = len(self.x)
        self.reference = y
        self.options = dict(subsets=subsets,subset_size=subset_size,seed=seed)
        self.alternate = {}
        self.m = min(subset_size,self.n,len(y))
        if self.m<2:
            raise ValueError('KID requires at least two samples')
        d = self.x.shape[1]
        self.xx = (self.x@self.x.T/d+1)**3
        self.parts = []
        rng = np.random.RandomState(seed)
        for unused in range(subsets):
            ix = rng.choice(self.n,self.m,replace=False)
            iy = rng.choice(len(y),self.m,replace=False)
            b = y[iy]
            yy = (b@b.T/d+1)**3
            yy_value = (yy.sum()-np.trace(yy))/(self.m*(self.m-1))
            xy_means = ((self.x@b.T/d+1)**3).mean(1)
            self.parts.append((ix,yy_value,xy_means))

    def __call__(self, indices):
        indices = np.asarray(indices,dtype=np.int64)
        if len(indices)!=self.n:
            # Unequal seed multiplicities stay inside their prompt cluster.
            return kid(self.x[indices],self.reference,**self.options)['mean']
        scores = []
        for ix,yy,xy in self.parts:
            chosen = indices[ix]
            xx = self.xx[np.ix_(chosen,chosen)]
            scores.append((xx.sum()-np.trace(xx))/(self.m*(self.m-1))+yy-2*xy[chosen].mean())
        return float(np.mean(scores))


def paired_bootstrap(first, second, reference, metric='FID', *, linked_reference=False,
                     replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED, max_seconds=300.):
    """Mappings prompt->feature matrix; matched references use those same IDs."""
    ids = sorted(set(first)&set(second))
    if linked_reference:
        ids = sorted(set(ids)&set(reference))
    if len(ids)<2:
        return dict(state='insufficient_prompt_clusters',prompt_ids=ids,achieved_replicates=0,
                    requested_replicates=replicates,ci95=None)
    a = np.concatenate([first[p] for p in ids])
    b = np.concatenate([second[p] for p in ids])
    ai,bi = {},{}
    start_a=start_b=0
    for p in ids:
        ai[p]=np.arange(start_a,start_a+len(first[p]));start_a+=len(first[p])
        bi[p]=np.arange(start_b,start_b+len(second[p]));start_b+=len(second[p])
    started=time.monotonic()
    if linked_reference:
        def calculate(chosen):
            x=np.concatenate([first[p] for p in chosen]);z=np.concatenate([second[p] for p in chosen]);y=np.concatenate([reference[p] for p in chosen])
            score = frechet if metric=='FID' else lambda x,y:kid(x,y)['mean']
            return score(z,y)-score(x,y)
        observed=calculate(ids)
    else:
        model = FixedReferenceFID if metric=='FID' else FixedReferenceKID
        ca,cb=model(a,reference),model(b,reference)
        def calculate(chosen):
            return cb(np.concatenate([bi[p] for p in chosen]))-ca(np.concatenate([ai[p] for p in chosen]))
        observed=calculate(ids)
    rng=np.random.RandomState(seed)
    draws=[]
    for index in range(replicates):
        if max_seconds is not None and time.monotonic()-started>=max_seconds:
            break
        chosen=[ids[i] for i in rng.randint(len(ids),size=len(ids))]
        draws.append(float(calculate(chosen)))
    complete=len(draws)==replicates
    return dict(state='complete' if complete else 'pending_uncertainty_compute',mean_difference=float(observed),
        difference='comparison minus baseline',prompt_ids=ids,prompt_count=len(ids),
        achieved_replicates=len(draws),requested_replicates=replicates,bootstrap_seed=seed,
        ci95=np.percentile(draws,[2.5,97.5]).tolist() if complete else None,
        partial_percentiles=np.percentile(draws,[2.5,97.5]).tolist() if draws and not complete else None,
        replicates=draws,seconds=time.monotonic()-started,
        reference_estimand='jointly sampled condition-linked reference' if linked_reference else 'fixed large reference distribution')


def paired_omni_bootstrap(first, second, reference, **kwargs):
    started=time.monotonic()
    maximum=kwargs.pop('max_seconds',300.)
    components=[]
    for name in ('horizontal','up','down'):
        component=paired_bootstrap(
            {p:np.stack([row['omni_'+name] for row in values]) for p,values in first.items()},
            {p:np.stack([row['omni_'+name] for row in values]) for p,values in second.items()},
            np.stack([row['omni_'+name] for row in reference]),metric='FID',
            max_seconds=None if maximum is None else max(0.,maximum-(time.monotonic()-started)),**kwargs)
        components.append(component)
    count=min(item['achieved_replicates'] for item in components)
    requested=kwargs.get('replicates',BOOTSTRAP_REPLICATES)
    draws=np.mean([item.get('replicates',[])[:count] for item in components],axis=0).tolist() if count else []
    complete=count==requested
    return dict(state='complete' if complete else 'pending_uncertainty_compute',
        mean_difference=float(np.mean([item['mean_difference'] for item in components])) if all('mean_difference' in item for item in components) else None,
        achieved_replicates=count,requested_replicates=requested,
        ci95=np.percentile(draws,[2.5,97.5]).tolist() if complete else None,
        prompt_ids=components[0]['prompt_ids'],replicates=draws,
        reference_estimand='fixed large reference distribution',
        grouping='three collection FIDs after within-panorama horizontal feature average; identical prompt draws across three components')


def extract_native_reference(run, manifest_path):
    import torch
    from PIL import Image
    from studies.panorama_metrics.features import Extractor, gpu_identity
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    run=Path(run);manifest=read(manifest_path);bank=manifest['bank']
    assert bank in ('A','B')
    identity=digest(dict(protocol=protocol_definitions(),manifest=sha(manifest_path),reference_bank=bank,extractor=gpu_identity(),evaluator_source=sha(__file__)))
    directory=run/'group2';directory.mkdir(parents=True,exist_ok=True)
    database=directory/'native_features.sqlite3';extractor=Extractor(with_clip=False)
    output=[];failures=[]
    with sqlite3.connect(str(database),timeout=120) as connection:
        connection.execute('CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, sha256 TEXT NOT NULL, data BLOB NOT NULL)')
        for row in manifest['records']:
            if row.get('state')!='complete':
                failures.append(dict(prompt_id=row['prompt_id'],direction_index=row['direction_index'],state=row.get('state','missing')))
                continue
            key=digest(dict(image=row['image_sha256'],protocol=identity))
            cached=connection.execute('SELECT sha256 FROM features WHERE key=?',(key,)).fetchone()
            if cached is None:
                assert sha(row['image_path'])==row['image_sha256']
                with Image.open(row['image_path']) as image:rgb=np.asarray(image.convert('RGB'))
                tensor=torch.from_numpy(rgb.transpose(2,0,1).copy()).unsqueeze(0).to(extractor.device)
                with torch.inference_mode():features=extractor.inception(tensor)[0].cpu().numpy()
                assert features.shape==(1,2048)
                stream=io.BytesIO();np.savez_compressed(stream,native_five=features)
                blob=stream.getvalue();checksum=hashlib.sha256(blob).hexdigest()
                connection.execute('INSERT INTO features VALUES (?,?,?)',(key,checksum,blob));connection.commit()
            else:checksum=cached[0]
            output.append(dict(prompt_id=row['prompt_id'],direction_index=row['direction_index'],
                image_sha256=row['image_sha256'],feature_path='sqlite://'+str(database)+'#'+key,feature_sha256=checksum))
    result=dict(bank=bank,complete=len(output)==120 and not failures,records=output,failures=failures,
                reference_manifest_path=str(manifest_path),reference_manifest_sha256=sha(manifest_path),protocol_sha256=identity)
    atomic(directory/('native_'+bank+'_features.json'),result)
    return result


def extract_real_reference(run, manifest_path, limit=None):
    """New common_v2 features; never relabel historical native-resolution crops."""
    import zipfile
    from PIL import Image
    import torch
    from .rendering import resample_common
    from studies.panorama_metrics.features import Extractor, gpu_identity
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    run=Path(run);manifest=read(manifest_path);source_hash=sha(manifest_path)
    identity=digest(dict(protocol=protocol_definitions(),renderer=rendering_protocols()['common_v2'],manifest=source_hash,extractor=gpu_identity(),evaluator_source=sha(__file__)))
    directory=run/'group2';directory.mkdir(parents=True,exist_ok=True)
    database=directory/'reference_features.sqlite3'
    ex=Extractor(with_clip=False)
    records=[]
    archive=zipfile.ZipFile(manifest['archive']) if manifest.get('archive') else None
    try:
        with sqlite3.connect(str(database),timeout=120) as connection:
            connection.execute('CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, sha256 TEXT NOT NULL, data BLOB NOT NULL)')
            for index,row in enumerate(manifest['records']):
                if limit is not None and index>=limit:break
                image_hash=row.get('image_sha256',row.get('sha256'))
                key=digest(dict(image=image_hash,protocol=identity))
                cached=connection.execute('SELECT sha256 FROM features WHERE key=?',(key,)).fetchone()
                if cached is None:
                    raw=archive.read(row['member']) if archive is not None else Path(row.get('image_path',row.get('path'))).read_bytes()
                    assert hashlib.sha256(raw).hexdigest()==image_hash
                    with Image.open(io.BytesIO(raw)) as image:rgb=np.asarray(image.convert('RGB'))
                    features=ex.image_features(resample_common(rgb))
                    features={k:v for k,v in features.items() if k in ('inception','omni_horizontal','omni_up','omni_down')}
                    stream=io.BytesIO();np.savez_compressed(stream,**features);blob=stream.getvalue();checksum=hashlib.sha256(blob).hexdigest()
                    connection.execute('INSERT INTO features VALUES (?,?,?)',(key,checksum,blob));connection.commit()
                else:checksum=cached[0]
                records.append(dict(reference_id=row.get('reference_id',row.get('member',str(index))),
                    image_sha256=image_hash,feature_path='sqlite://'+str(database)+'#'+key,feature_sha256=checksum))
                if index%25==0:print('REFERENCE_FEATURES',index+1,len(manifest['records']),flush=True)
    finally:
        if archive is not None:archive.close()
    result=dict(namespace='common_v2',reference_manifest_path=str(manifest_path),reference_manifest_sha256=source_hash,
        protocol_sha256=identity,complete=len(records)==len(manifest['records']),records=records)
    atomic(directory/'reference_features.json',result)
    return result


def evaluate(run, records, registry=None, *, reference_manifest=None, bank_manifests=None,
             max_bootstrap_seconds=300., bootstrap_replicates=BOOTSTRAP_REPLICATES):
    from .metrics import load_features
    run=Path(run);directory=run/'group2';directory.mkdir(parents=True,exist_ok=True)
    atomic(directory/'protocol.json',dict(definitions=protocol_definitions(),registry=registry))
    source=run/'group1/per_panorama_metrics.json'
    if not source.exists():
        result=dict(state='pending_generated_features',groups=[],paired_comparisons=[])
        atomic(directory/'status.json',result);return result
    all_rows=read(source)
    cohorts=read(run/'inventory/cohorts.json')['cohorts']
    groups,pairs,blockers=[],[],[]
    reference_record=directory/'reference_features.json'
    reference=[]
    if reference_record.exists() and read(reference_record).get('complete'):
        reference=[load_features(row) for row in read(reference_record)['records']]
    else:
        blockers.append(dict(metric='FID-real/KID-real/OmniFID',namespace='common_v2',state='pending_common_reference_features',
            reference_manifest_path=str(reference_manifest) if reference_manifest else None))
    rows=[row for row in all_rows if row['namespace']=='common_v2' and 'feature_path'in row]
    features={row['case_id']:load_features(row) for row in rows}
    if reference:
        reference_x=np.concatenate([row['inception'] for row in reference])
        for cohort_id,cohort in cohorts.items():
            prompts=cohort['prompt_ids']
            if not prompts:continue
            for method in sorted({row['method_id'] for row in rows}):
                selected=[row for row in rows if row['method_id']==method and row['prompt_id'] in prompts]
                if {row['prompt_id'] for row in selected}!=set(prompts):continue
                score=collection_scores([features[row['case_id']] for row in selected],reference)
                groups.append(dict(namespace='common_v2',reference_kind='real',cohort_id=cohort_id,method_id=method,
                    prompt_ids=prompts,**score,interpretation='exploratory dependent-view distribution'))
            for baseline,comparison in PRIMARY_COMPARISONS:
                aa={p:np.concatenate([features[row['case_id']]['inception'] for row in rows if row['method_id']==baseline and row['prompt_id']==p]) for p in prompts if any(row['method_id']==baseline and row['prompt_id']==p for row in rows)}
                bb={p:np.concatenate([features[row['case_id']]['inception'] for row in rows if row['method_id']==comparison and row['prompt_id']==p]) for p in prompts if any(row['method_id']==comparison and row['prompt_id']==p for row in rows)}
                for metric in ('FID','KID'):
                    value=paired_bootstrap(aa,bb,reference_x,metric,replicates=bootstrap_replicates,max_seconds=max_bootstrap_seconds)
                    pairs.append(dict(namespace='common_v2',reference_kind='real',cohort_id=cohort_id,baseline=baseline,comparison=comparison,metric=metric,**value))
                aa_omni={p:[features[row['case_id']] for row in rows if row['method_id']==baseline and row['prompt_id']==p] for p in set(aa)&set(bb)}
                bb_omni={p:[features[row['case_id']] for row in rows if row['method_id']==comparison and row['prompt_id']==p] for p in set(aa)&set(bb)}
                value=paired_omni_bootstrap(aa_omni,bb_omni,reference,replicates=bootstrap_replicates,max_seconds=max_bootstrap_seconds)
                pairs.append(dict(namespace='common_v2',reference_kind='real',cohort_id=cohort_id,baseline=baseline,comparison=comparison,metric='OmniFID',**value))
    # New native banks are optional inputs, never inferred from historical seed0
    # aliases. Each bank record has prompt_id, direction_index and verified features.
    bank_manifests = bank_manifests or {bank:directory/('native_'+bank+'_features.json') for bank in ('A','B') if (directory/('native_'+bank+'_features.json')).exists()}
    bank_features = {}
    for bank,path in sorted((bank_manifests or {}).items()):
        bank_record=read(path)
        if not bank_record.get('complete'):
            blockers.append(dict(metric='FID-native/KID-native',bank=bank,state='pending_reference_generation'));continue
        refs=bank_record['records'];by_prompt={}
        for item in refs:
            by_prompt.setdefault(item['prompt_id'],[]).append(item)
        for cohort_id in ('development15','development24'):
            prompts=cohorts[cohort_id]['prompt_ids']
            if not all(p in by_prompt and sorted(x['direction_index'] for x in by_prompt[p])==list(range(5)) for p in prompts):
                blockers.append(dict(metric='FID-native/KID-native',bank=bank,cohort_id=cohort_id,state='missing_matched_directions'));continue
            reference_by_prompt={p:np.concatenate([load_features(x)['native_five'] for x in sorted(by_prompt[p],key=lambda x:x['direction_index'])]) for p in prompts}
            assert all(value.shape==(5,2048) for value in reference_by_prompt.values())
            bank_features[(bank,cohort_id)]=reference_by_prompt
            for method in sorted({row['method_id'] for row in rows}):
                selected=[row for row in rows if row['method_id']==method and row['prompt_id']in prompts]
                if {row['prompt_id'] for row in selected}!=set(prompts):continue
                x=np.concatenate([features[row['case_id']]['native_five'] for row in selected]);y=np.concatenate([reference_by_prompt[p] for p in prompts])
                groups.append(dict(namespace='native_five_v2',reference_kind='native',bank=bank,
                    role='primary' if bank=='A' else 'reference-sampling sensitivity',cohort_id=cohort_id,method_id=method,
                    FID=frechet(x,y),KID=kid(x,y),prompt_ids=prompts,reference_directional_slots=len(y),generated_view_count=len(x)))
            for baseline,comparison in PRIMARY_COMPARISONS:
                aa={p:np.concatenate([features[row['case_id']]['native_five'] for row in rows if row['method_id']==baseline and row['prompt_id']==p]) for p in prompts if any(row['method_id']==baseline and row['prompt_id']==p for row in rows)}
                bb={p:np.concatenate([features[row['case_id']]['native_five'] for row in rows if row['method_id']==comparison and row['prompt_id']==p]) for p in prompts if any(row['method_id']==comparison and row['prompt_id']==p for row in rows)}
                for metric in ('FID','KID'):
                    value=paired_bootstrap(aa,bb,reference_by_prompt,metric,linked_reference=True,replicates=bootstrap_replicates,max_seconds=max_bootstrap_seconds)
                    pairs.append(dict(namespace='native_five_v2',reference_kind='native',bank=bank,cohort_id=cohort_id,baseline=baseline,comparison=comparison,metric=metric,**value))
    for cohort_id in ('development15','development24'):
        if ('A',cohort_id) in bank_features and ('B',cohort_id) in bank_features:
            aa,bb=bank_features[('A',cohort_id)],bank_features[('B',cohort_id)]
            assert set(aa)==set(bb)
            x=np.concatenate([aa[p] for p in sorted(aa)]);y=np.concatenate([bb[p] for p in sorted(bb)])
            groups.append(dict(namespace='native_five_v2',reference_kind='native-to-native',cohort_id=cohort_id,
                banks=['A','B'],FID=frechet(x,y),KID=kid(x,y),prompt_ids=sorted(aa),
                interpretation='reference sampling diagnostic; not subtracted from method FID'))
    if not bank_manifests:
        blockers.append(dict(metric='FID-native/KID-native',state='pending_new_reference_banks',banks=['A','B'],legacy_references_not_substituted=True))
    assets=read(run/'group1/assets/status.json') if (run/'group1/assets/status.json').exists() else {}
    blockers.extend(dict(metric=name,**assets.get(key,{'status':'pending_asset_check'})) for name,key in [('FAED','faed'),('Distort-FID','distort_fid')])
    result=dict(groups=groups,paired_comparisons=pairs,blockers=blockers,
        state='partial' if blockers or any(p['state']!='complete' for p in pairs) else 'complete',
        exact_feature_dimension=2048,units='FID unscaled; KID raw MMD squared',generation_performed=False)
    atomic(directory/'group_metrics.json',groups);atomic(directory/'paired_comparisons.json',pairs);atomic(directory/'status.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference-manifest',type=Path);parser.add_argument('--stage',choices=('reference','aggregate'),default='aggregate')
    parser.add_argument('--limit',type=int);parser.add_argument('--bootstrap-seconds',type=float,default=300.)
    args=parser.parse_args()
    if args.stage=='reference':
        if args.reference_manifest is None:parser.error('--reference-manifest required')
        extract_real_reference(args.run,args.reference_manifest,args.limit)
    else:
        inventory=read(args.run/'inventory/records.json')
        evaluate(args.run,inventory['records'] if isinstance(inventory,dict) else inventory,read(args.run/'protocol_registry.json'),
            reference_manifest=args.reference_manifest,max_bootstrap_seconds=args.bootstrap_seconds)

if __name__=='__main__':main()
