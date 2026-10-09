"""Frozen panorama metrics with factor-aware keys and paired four-prompt summaries."""
import os,itertools,csv
import numpy as np
from PIL import Image
from studies.gradient_refinement.common import *

def main():
    import torch,cv2
    from studies.panorama_metrics.features import Extractor,save_features
    from studies.panorama_metrics.render import render,CUBE
    from studies.panorama_metrics.seams import ds,cubemap_seams
    from studies.panorama_metrics.numerics import inception_score
    from studies.panorama_metrics.protocols import cpu_identity
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(4);cv2.setNumThreads(1);preserved()
    manifest=read(OUT/'manifest.json');ex=Extractor();assert ex.inception is not None and ex.clip is not None
    results=[];features={};directory=OUT/'evaluation';directory.mkdir(exist_ok=True)
    for row in manifest['rows']:
        assert case_complete(row),'Missing '+row['key'];path=folder(row);meta=read(path/'metadata.json')
        with Image.open(path/'final.png') as im:rgb=np.asarray(im.convert('RGB'))
        cache=directory/'features'/(row['key']+'.npz');cache.parent.mkdir(parents=True,exist_ok=True)
        key=digest(dict(case=row['semantic_sha256'],image=sha(path/'final.png'),protocol=ex.identity,cpu=cpu_identity()))
        cache_meta=cache.with_suffix('.json')
        if cache_meta.exists() and read(cache_meta)['key']==key and sha(cache)==read(cache_meta)['sha256']:
            with np.load(cache,allow_pickle=False) as f:values=dict(f)
        else:
            values=ex.image_features(rgb);save_features(cache,values);write(cache_meta,dict(key=key,sha256=sha(cache)))
        scores,detail=ex.cs(values,meta['prompt']);scores.update(DS=ds(rgb));scores.update(cubemap_seams(render(rgb,CUBE)))
        result=dict(index=row['index'],key=row['key'],prompt=row['prompt'],reference_mode=row['reference_mode'],gradient_mode=row['gradient_mode'],last_fraction=row['last_fraction'],
            metrics=scores,cs_details=detail,semantic_sha256=row['semantic_sha256'],image_sha256=sha(path/'final.png'),
            generation_seconds=meta.get('generation_seconds'),solver_seconds=meta.get('solver_seconds'),peak_allocated_gib=meta.get('peak_allocated_gib'),counts=meta['counts'],reused=meta['reused'])
        results.append(result);features[row['key']]=values;write(directory/'cases'/(row['key']+'.json'),result)
        print('METRICS',row['key'],scores,flush=True)
    aggregates=[]
    for r,g,f in itertools.product(REFERENCES,GRADIENTS,FRACTIONS):
        group=[x for x in results if (x['reference_mode'],x['gradient_mode'],x['last_fraction'])==(r,g,f)]
        assert len(group)==4
        aggregates.append(dict(reference_mode=r,gradient_mode=g,last_fraction=f,panoramas=4,
            metrics={k:float(np.mean([x['metrics'][k] for x in group])) for k in ('DS','CS','Seam-SSIM','Seam-Sobel')},
            generation_seconds=float(np.mean([x['generation_seconds'] for x in group])),
            IS_descriptive=inception_score(np.concatenate([features[x['key']]['logits'] for x in group]))))
    pairs=[]
    for left in results:
        for right in results:
            if left['prompt']!=right['prompt']:continue
            changed=[key for key in ('reference_mode','gradient_mode','last_fraction') if left[key]!=right[key]]
            if len(changed)!=1:continue
            factor=changed[0]
            if factor=='reference_mode' and left[factor]!='screened':continue
            if factor=='gradient_mode' and left[factor]!='poisson_select':continue
            if factor=='last_fraction' and left[factor]!=0:continue
            pairs.append(dict(prompt=left['prompt'],factor=factor,baseline=left['key'],comparison=right['key'],
                deltas={k:right['metrics'][k]-left['metrics'][k] for k in ('DS','CS','Seam-SSIM','Seam-Sobel')},
                generation_seconds_delta=right['generation_seconds']-left['generation_seconds']))
    write(directory/'summary.json',dict(rows=results,aggregates=aggregates,paired=pairs,cpu_protocol=cpu_identity(),gpu_protocol=ex.identity,
        interpretation='Four matched prompts: descriptive ablation only. Rendered views are dependent. IS is descriptive; no distribution-level claim or small-sample FID.',
        frozen_metric_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'studies/panorama_metrics').glob('*.py'))},job=os.environ['SLURM_JOB_ID']))
    with (directory/'per-image.csv').open('w') as stream:
        keys=['key','prompt','reference_mode','gradient_mode','last_fraction','generation_seconds','solver_seconds','DS','CS','Seam-SSIM','Seam-Sobel']
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for r in results:writer.writerow({k:({**r,**r['metrics']})[k] for k in keys})
if __name__=='__main__':main()
