"""Read-only use of frozen metric functions; outputs/caches stay study-local."""
import os
import csv
import numpy as np
from PIL import Image
from studies.gradient_blending.common import *


def collection_summary(prompts,rows,features):
    from studies.panorama_metrics.numerics import inception_score
    result={}
    for mode in MODES:
        selected=[r for r in rows if r['mode']==mode and r['prompt'] in prompts]
        assert len(selected)==len(prompts) and {r['prompt'] for r in selected}==set(prompts)
        logits=np.concatenate([features[(prompt,mode)]['logits'] for prompt in prompts])
        result[mode]=dict(prompts=list(prompts),panorama_count=len(prompts),view_count=len(logits),
            IS=inception_score(logits),
            **{name:float(np.mean([r['metrics'][name] for r in selected])) for name in ['DS','CS','Seam-SSIM','Seam-Sobel']})
    return result

def main():
    import torch,cv2
    from studies.panorama_metrics.features import Extractor,save_features
    from studies.panorama_metrics.render import render,CUBE
    from studies.panorama_metrics.seams import ds,cubemap_seams
    from studies.panorama_metrics.numerics import inception_score
    from studies.panorama_metrics.protocols import cpu_identity
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    preserved();torch.set_num_threads(4);cv2.setNumThreads(1)
    with (ROOT/'outputs/metrics-evaluation/seed0-v1/per_image_metrics.csv').open() as stream:
        historical=list(csv.DictReader(stream))
    ex=Extractor();assert ex.inception is not None and ex.clip is not None
    directory=OUT/'evaluation';directory.mkdir(exist_ok=True);rows=[];features_by_case={}
    for prompt in PROMPTS:
        for mode in MODES:
            folder=OUT/'cases'/prompt/mode
            if not (folder/'status.json').exists() or read(folder/'status.json')['state']!='complete':
                raise RuntimeError('Matched pilot incomplete: '+prompt+'/'+mode)
            meta=read(folder/'metadata.json');image_path=folder/'final.png'
            assert sha(image_path)==meta['artifacts']['final.png']
            with Image.open(image_path) as im:rgb=np.asarray(im.convert('RGB'))
            cache=directory/(prompt+'-'+mode+'-features.npz')
            key=digest(dict(image=sha(image_path),protocol=ex.identity))
            cache_meta=cache.with_suffix('.json')
            if cache_meta.exists() and read(cache_meta).get('key')==key and sha(cache)==read(cache_meta)['sha256']:
                with np.load(cache,allow_pickle=False) as loaded:features=dict(loaded)
            else:
                features=ex.image_features(rgb);save_features(cache,features)
                write(cache_meta,dict(key=key,sha256=sha(cache)))
            prompt_record_value=prompt_record(ROOT/'prompts'/(prompt+'.txt'))[0]
            scores,detail=ex.cs(features,prompt_record_value)
            scores.update(DS=ds(rgb));scores.update(cubemap_seams(render(rgb,CUBE)))
            row=dict(backend=BACKEND,prompt=prompt,mode=mode,metrics=scores,cs_details=detail,image_sha256=sha(image_path),
                generation_seconds=meta.get('generation_seconds'),solver_seconds=meta.get('solver_seconds'),
                peak_allocated_gib=meta.get('peak_allocated_gib'),counts=meta.get('counts'),
                reused=meta.get('reused',False),cpu_protocol=cpu_identity(),gpu_protocol=ex.identity)
            if mode=='rgb':
                reference=next(r for r in historical if r.get('image_sha256')==row['image_sha256'])
                errors={k:abs(scores[k]-float(reference[k])) for k in ['DS','CS','Seam-SSIM','Seam-Sobel']}
                assert max(errors.values())<1e-4,('Frozen evaluator mismatch',BACKEND,prompt,errors)
                row['absolute_difference_from_frozen']=errors
            rows.append(row);features_by_case[(prompt,mode)]=features
            write(directory/(prompt+'-'+mode+'.json'),row)
            print('METRICS',prompt,mode,scores,flush=True)
    collections=collection_summary(PROMPTS,rows,features_by_case)
    collection_groups={}
    if SUITE=='flux-scenes20':
        collection_groups=dict(new_17=collection_summary(NEW_FLUX_PROMPTS,rows,features_by_case),
                               pilot_3=collection_summary(PILOT_PROMPTS,rows,features_by_case))
    paired=[]
    for prompt in PROMPTS:
        base=next(r for r in rows if r['prompt']==prompt and r['mode']=='rgb')
        for mode in MODES[1:]:
            value=next(r for r in rows if r['prompt']==prompt and r['mode']==mode)
            paired.append(dict(prompt=prompt,mode=mode,deltas={k:value['metrics'][k]-base['metrics'][k] for k in ['DS','CS','Seam-SSIM','Seam-Sobel']}))
    write(directory/'summary.json',dict(rows=rows,collections=collections,collection_groups=collection_groups,paired=paired,
        interpretation='n=%d descriptive only; IS is one full matched collection score over %d dependent views, not a per-image average'%(len(PROMPTS),collections['rgb']['view_count']),
        frozen_evaluator_hashes={k:v for k,v in read(BASE_OUT/'preservation.json')['hashes'].items() if k.startswith('studies/panorama_metrics/')},
        job=os.environ['SLURM_JOB_ID']))
    preserved()
if __name__=='__main__':main()
