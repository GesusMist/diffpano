"""Frozen DS/CS and matched four-panorama IS, with suite-local semantic caches."""
import argparse,csv,fcntl,traceback
import numpy as np
from PIL import Image
from studies.mean_max_old89.common import *
EVAL=OUT/'evaluation'


def csv_write(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    os.replace(str(tmp),str(path))


def summarize(rows,features,names=PROMPTS,subset='all15'):
    from studies.panorama_metrics.numerics import inception_score
    result=[]
    for ref in REFERENCES:
        for alpha in ALPHAS:
            group=[next(r for r in rows if r['reference_mode']==ref and r['gradient_mix_alpha']==alpha and r['prompt']==p) for p in names]
            scored=[r for r in group if r['scoring_status']=='complete'];done=len(scored)==len(names)
            record=dict(reference_mode=ref,gradient_mix_alpha=alpha,subset=subset,
                completed_panorama_count=sum(r['completion_status']=='complete' for r in group),
                scored_panorama_count=len(scored),expected_panorama_count=len(names),matched_prompt_ids=list(names),
                scored_prompt_ids=[r['prompt'] for r in scored],case_ids=[r['case_id'] for r in group],
                complete=done,provisional=not done,view_count=8*len(scored),expected_view_count=8*len(names),mean_DS=None,mean_CS=None,IS=None,mean_generation_seconds=None)
            if done:
                logits=np.concatenate([features[r['case_id']]['logits'] for r in group]);assert len(logits)==8*len(names)
                record.update(mean_DS=float(np.mean([r['DS'] for r in group])),mean_CS=float(np.mean([r['CS'] for r in group])),
                    IS=float(inception_score(logits)),mean_generation_seconds=float(np.mean([r['generation_seconds'] for r in group])))
            result.append(record)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    import torch,cv2
    from studies.panorama_metrics.features import Extractor,save_features
    from studies.panorama_metrics.seams import ds
    from studies.panorama_metrics.protocols import protocols,cpu_identity
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    m=freeze_check();preserved();EVAL.mkdir(parents=True,exist_ok=True)
    lock=(EVAL/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    current=protocols();old=read(ROOT/'outputs/metrics-evaluation/seed0-v1/metric_protocols.json')
    for metric in ('DS','IS','CS'):
        assert digest({k:v for k,v in current[metric].items() if k!='status'})==digest({k:v for k,v in old[metric].items() if k!='status'}),metric
    ex=Extractor();assert ex.clip is not None and ex.inception is not None
    # Read the historical evaluator certificate; never call its writer here.
    historical_gate=read(ROOT/'outputs/metrics-evaluation/seed0-v1/validation/gpu.json')
    assert historical_gate['passed'] and historical_gate['protocol_sha256']==ex.identity
    from studies.panorama_metrics.numerics import inception_score
    assert abs(inception_score(np.zeros((32,1008),dtype=np.float32))-1.)<1e-12
    import clip
    with torch.inference_mode():
        text='A coral reef underwater.';actual,detail=ex.text_embedding(text)
        expected=ex.clip.encode_text(clip.tokenize([text]).to(ex.device)).float();expected/=expected.norm(dim=-1,keepdim=True)
        np.testing.assert_allclose(actual,expected[0].cpu().numpy(),atol=2e-6)
    _,long=ex.text_embedding('coral reef fish colorful water '*100)
    assert len(long['chunks'])>1 and sum(long['chunks'])==long['payload_tokens'] and not long['truncated']
    write(EVAL/'validation.json',dict(passed=True,historical_gpu_gate_sha256=sha(ROOT/'outputs/metrics-evaluation/seed0-v1/validation/gpu.json'),
        identical_logits_IS=1.,CLIP_short_exact=True,CLIP_long_no_truncation=True,gpu_protocol=ex.identity))
    protocol=dict(metrics={k:current[k] for k in ('DS','IS','CS')},cpu_identity=cpu_identity(),gpu_identity=ex.identity,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'studies/panorama_metrics').glob('*.py'))},
        evaluator_sha256=sha(__file__),historical_protocol_sha256=sha(ROOT/'outputs/metrics-evaluation/seed0-v1/metric_protocols.json'),
        software=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__),
        input='final.png only; same uint8 display conversion protocol for every method',
        aggregation='Six matched 15-prompt collections, 120 horizontal views each; separate original11 (88 views) and new4 (32 views) subset IS; one full split, no per-image IS',
        prompt_ids=list(PROMPTS),cache_root=str(EVAL/'cache'))
    write(EVAL/'metric_protocol.json',protocol);protocol_hash=digest(protocol);rows=[];features={};errors=[]
    for row in m['rows']:
        info,_=prompt(row['prompt']);path=prompt_path(row['prompt']);directory=folder(row)
        status=read(directory/'status.json') if (directory/'status.json').exists() else {}
        record={k:row[k] for k in ('case_id','prompt','subset','layout','gradient_mode','reference_mode','gradient_mix_alpha','semantic_sha256')}
        record.update(prompt_path=str(path),prompt_sha256=info['original_sha256'],effective_prompt_sha256=info['effective_sha256'],
            anchor_source='fused_rgb_erp' if row['reference_mode']=='single_pixel' else None,
            anchor_semantics_version=2 if row['reference_mode']=='single_pixel' else None,
            completion_status=status.get('state','missing'),scoring_status='pending_generation',DS=None,CS=None,generation_seconds=None,image_sha256=None,error=None)
        try:
            if not complete(row):
                rows.append(record);continue
            image_path=directory/'final.png';h=sha(image_path);meta=read(directory/'metadata.json')
            assert meta['prompt']==info and h==meta['artifacts']['final.png']==status['image_sha256']
            key=digest(dict(image=h,prompt=info,configuration=row['semantic_sha256'],source=row['source_sha256'],protocol=protocol_hash))
            cache=EVAL/'cache'/(key+'.json');fp=EVAL/'cache'/(key+'.npz')
            scores=None;f=None
            if cache.exists() and fp.exists():
                cached=read(cache)
                if cached['key']==key and cached['feature_sha256']==sha(fp):
                    scores=cached
                    with np.load(fp,allow_pickle=False) as data:f=dict(data)
            if scores is None:
                with Image.open(image_path) as im:
                    assert im.size==(4096,2048);rgb=np.asarray(im.convert('RGB'))
                f=ex.image_features(rgb);save_features(fp,f)
                cs,details=ex.cs(f,info)
                scores=dict(key=key,DS=float(ds(rgb)),CS=float(cs['CS']),cs_details=details,feature_sha256=sha(fp))
                write(cache,scores)
            assert f['logits'].shape[0]==8 and f['clip_views'].shape[0]==10
            assert np.isfinite(scores['DS']) and np.isfinite(scores['CS'])
            features[row['case_id']]=f
            record.update(DS=scores['DS'],CS=scores['CS'],generation_seconds=meta['generation_seconds'],cs_details=scores['cs_details'],image_sha256=h,
                image_path=str(image_path),metadata_sha256=sha(directory/'metadata.json'),scoring_status='complete',
                feature_path=str(fp),feature_sha256=scores['feature_sha256'],cache_key=key,protocol_sha256=protocol_hash)
            print('SCORED',row['case_id'],scores['DS'],scores['CS'],flush=True)
        except Exception:
            record['scoring_status']='failed';record['error']=traceback.format_exc();errors.append(dict(case_id=row['case_id'],error=record['error']))
            print('SCORING_FAILED',row['case_id'],record['error'],flush=True)
        rows.append(record)
    groups=summarize(rows,features);subsets=summarize(rows,features,ORIGINAL,'original11')+summarize(rows,features,NEW,'new4')
    assert len(rows)==90 and len(groups)==6 and len(subsets)==12
    csv_write(EVAL/'per_image_scores.csv',rows,['case_id','prompt','prompt_path','prompt_sha256','effective_prompt_sha256','subset','layout','gradient_mode','reference_mode','gradient_mix_alpha','semantic_sha256','image_sha256','completion_status','scoring_status','DS','CS','generation_seconds','protocol_sha256','cache_key','error'])
    fields=['reference_mode','gradient_mix_alpha','subset','completed_panorama_count','scored_panorama_count','expected_panorama_count','matched_prompt_ids','scored_prompt_ids','view_count','expected_view_count','complete','provisional','mean_DS','mean_CS','IS','mean_generation_seconds']
    csv_write(EVAL/'summary_scores.csv',groups,fields);csv_write(EVAL/'subset_scores.csv',subsets,fields)
    counts=dict(valid_generations=sum(complete(r) for r in m['rows']),DS=sum(r['DS'] is not None for r in rows),CS=sum(r['CS'] is not None for r in rows),matched_IS=sum(g['IS'] is not None for g in groups))
    passed=counts==dict(valid_generations=90,DS=90,CS=90,matched_IS=6)
    freeze_check();preserved()
    write(EVAL/'metrics.json',dict(rows=rows,groups=groups,subsets=subsets,counts=counts,complete=passed,errors=errors,protocol_sha256=protocol_hash,job=os.environ['SLURM_JOB_ID']))
    write(EVAL/'status.json',dict(state='complete' if passed else 'partial',counts=counts,metric_failures=len(errors),job=os.environ['SLURM_JOB_ID']))
    if passed:write(OUT/'completion.json',dict(passed=True,counts=counts,manifest_sha256=sha(OUT/'manifest.json'),metrics_sha256=sha(EVAL/'metrics.json'),job=os.environ['SLURM_JOB_ID']))
    print('EVALUATION',counts,'complete',passed,flush=True)
    if not passed and not args.allow_partial:raise SystemExit(1)
if __name__=='__main__':
    try:main()
    except Exception:
        write(EVAL/'status.json',dict(state='failed',error=traceback.format_exc(),job=os.environ.get('SLURM_JOB_ID'),generation_results_modified=False))
        raise
