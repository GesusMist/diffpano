"""Comparable DS/CS per panorama and IS per matched FLUX collection."""
import argparse,csv,fcntl,itertools,json,os
from pathlib import Path
import numpy as np
from PIL import Image
from studies.gradient_refinement.common import ROOT,sha,read,write,digest

OUT=ROOT/'outputs/10.9gradient-metrics/seed0-v1'
REF=ROOT/'outputs/10.8gradient-refinement-flux/seed0-v1'
BLEND=ROOT/'outputs/10.7gradient-blending-flux/seed0-v1'
FACTOR_PROMPTS=('air_balloons','underwater','ruins','firework')
MODES=('rgb','poisson_mean','poisson_select')

def csv_write(path,rows,fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temp.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    os.replace(str(temp),str(path))

def inputs():
    from studies.gradient_refinement.common import case_complete
    ref=read(REF/'manifest.json');blend=read(BLEND/'manifest.json');rows=[]
    for row in ref['rows']:
        rows.append(dict(suite='refinement64',key=row['key'],prompt=row['prompt'],mode=row['gradient_mode'],reference_mode=row['reference_mode'],
            last_fraction=row['last_fraction'],directory=str(REF/'cases'/row['key']),semantic_sha256=row['semantic_sha256'],
            valid_semantics=case_complete(row,verify=False)))
    for prompt,mode in itertools.product(blend['prompts'],MODES):
        rows.append(dict(suite='blending20',key=prompt+'/'+mode,prompt=prompt,mode=mode,reference_mode='rgb' if mode=='rgb' else 'screened',
            last_fraction=0.,directory=str(BLEND/'cases'/prompt/mode),semantic_sha256=None,valid_semantics=True))
    assert len(rows)==124 and len(blend['prompts'])==20
    assert len({(r['suite'],r['key']) for r in rows})==124
    return rows,ref,blend

def legacy_features(row,image_hash,identity,cpu):
    if row['suite']=='refinement64':
        path=REF/'evaluation/features'/(row['key']+'.npz')
        key=digest(dict(case=row['semantic_sha256'],image=image_hash,protocol=identity,cpu=cpu))
    else:
        path=BLEND/'evaluation'/(row['prompt']+'-'+row['mode']+'-features.npz')
        key=digest(dict(image=image_hash,protocol=identity))
    meta=path.with_suffix('.json')
    if path.exists() and meta.exists():
        m=read(meta)
        if m.get('key')==key and m.get('sha256')==sha(path):
            with np.load(path,allow_pickle=False) as f:return dict(f),str(path)
    return None,None

def group_rows(suite,rows,features,prompts):
    from studies.panorama_metrics.numerics import inception_score
    groups=[]
    if suite=='refinement64':
        specs=[(r,g,f,'factorial4',list(FACTOR_PROMPTS)) for r,g,f in itertools.product(('screened','global_mean','single_pixel','coarse_color'),('poisson_select','poisson_max'),(0.,.1))]
    else:
        specs=[('rgb' if g=='rgb' else 'screened',g,0.,subset,selected) for g in MODES
            for subset,selected in [('all20',list(prompts)),('matched_factorial4',list(FACTOR_PROMPTS))]]
    for reference,mode,fraction,subset,expected in specs:
        selected=[r for r in rows if r['suite']==suite and r['reference_mode']==reference and r['mode']==mode and r['last_fraction']==fraction and r['prompt'] in expected]
        byprompt={r['prompt']:r for r in selected}
        assert len(byprompt)==len(selected), 'Duplicate prompt in collection'
        present=[p for p in expected if p in byprompt];missing=[p for p in expected if p not in byprompt]
        ordered=[byprompt[p] for p in present]
        record=dict(suite=suite,reference_mode=reference,mode=mode,last_fraction=fraction,subset=subset,
            complete=not missing,panorama_count=len(ordered),expected_panorama_count=len(expected),view_count=8*len(ordered),
            prompts=present,missing_prompts=missing,prompt_set_sha256=digest(expected),
            effective_prompts_sha256=digest([(r['prompt'],r['effective_prompt_sha256']) for r in ordered]),
            DS=None,IS=None,CS=None,relaxed_tolerance_cases=[r['key'] for r in ordered if r.get('relative_tolerance',0)>1e-5])
        # Never publish an incomplete collection as a matched experimental score.
        if not missing:
            logits=np.concatenate([features[(r['suite'],r['key'])]['logits'] for r in ordered])
            assert len(logits)==record['view_count']
            record.update(DS=float(np.mean([r['DS'] for r in ordered])),CS=float(np.mean([r['CS'] for r in ordered])),IS=float(inception_score(logits)))
        groups.append(record)
    return groups

def exports(rows,missing,groups,protocol):
    columns=['suite','key','prompt','mode','reference_mode','last_fraction','DS','CS','relative_tolerance','absolute_tolerance',
        'image_sha256','effective_prompt_sha256','metadata_sha256','image_path','feature_path','feature_sha256','cpu_protocol','gpu_protocol']
    fields=['suite','reference_mode','mode','last_fraction','subset','complete','panorama_count','expected_panorama_count','view_count','DS','IS','CS','prompt_set_sha256']
    csv_write(OUT/'per-image-metrics.csv',rows,columns);csv_write(OUT/'group-metrics.csv',groups,fields)
    for suite,expected in [('refinement64',64),('blending20',60)]:
        selected=[r for r in rows if r['suite']==suite];collection=[g for g in groups if g['suite']==suite]
        absent=[r for r in missing if r['suite']==suite];target=OUT/suite
        csv_write(target/'per-image-metrics.csv',selected,columns);csv_write(target/'group-metrics.csv',collection,fields)
        write(target/'summary.json',dict(expected=expected,scored=len(selected),complete=len(selected)==expected,rows=selected,groups=collection,missing=absent,protocol=protocol))
    write(OUT/'status.json',dict(expected=124,scored=len(rows),complete=len(rows)==124,missing=missing,job=os.environ['SLURM_JOB_ID']))
    lines=['FLUX DS / IS / CS evaluation',
        'Refinement: %d/64 images. Twenty-prompt blending: %d/60 images.'%(sum(r['suite']=='refinement64' for r in rows),sum(r['suite']=='blending20' for r in rows)),
        'The twenty-prompt study is stored in outputs/10.7gradient-blending-flux, not 10.8gradient-blending-flux.',
        'DS and CS are per-image scores; group DS/CS are means over matched prompts.',
        'IS is one full-collection score from all eight fixed horizontal views per panorama, never an average of per-image IS.',
        'Same DS implementation, camera renderer, CLIP/Inception checkpoints, prompt policy and numeric functions as earlier metrics.',
        'DS lower is better; IS and CS higher are better. Four-prompt collections are descriptive; views are dependent.',
        'Factorial variants are kept separate. No IS pools different experimental configurations.',
        'blending20/all20 uses 20 prompts. matched_factorial4 uses the same four prompts as refinement64 for later matched comparisons.',
        'Solver tolerance exceptions are retained per image and listed in the JSON groups.',
        '', 'suite | reference | mode | refinement | subset | images | DS | IS | CS']
    for g in groups:
        values='pending incomplete collection' if not g['complete'] else '%.8f | %.8f | %.8f'%(g['DS'],g['IS'],g['CS'])
        lines.append('%s | %s | %s | %.2f | %s | %d/%d | %s'%(g['suite'],g['reference_mode'],g['mode'],g['last_fraction'],g['subset'],g['panorama_count'],g['expected_panorama_count'],values))
    if missing:lines+=['','Missing cases:']+[r['suite']+'/'+r['key']+': '+r['reason'] for r in missing]
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    if len(rows)==124:
        write(OUT/'completion.json',dict(passed=True,images=124,refinement_images=64,blending_images=60,groups=len(groups),
            job=os.environ['SLURM_JOB_ID'],per_image_sha256=sha(OUT/'per-image-metrics.csv'),group_sha256=sha(OUT/'group-metrics.csv'),
            report_sha256=sha(OUT/'report.txt'),protocol_sha256=digest(protocol)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    import torch,cv2
    from studies.panorama_metrics.features import Extractor,save_features
    from studies.panorama_metrics.seams import ds
    from studies.panorama_metrics.protocols import protocols,cpu_identity
    from studies.panorama_metrics.numerics import inception_score
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available(), 'GPU Slurm allocation required'
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    expected,refmanifest,blendmanifest=inputs()
    source_files=sorted((ROOT/'studies/panorama_metrics').glob('*.py'))+[Path(__file__).resolve()]
    source_snapshot={str(p):sha(p) for p in source_files}
    ex=Extractor();assert ex.inception is not None and ex.clip is not None
    cpu=cpu_identity();prior_protocol=read(ROOT/'outputs/metrics-evaluation/seed0-v1/metric_protocols.json');current_protocol=protocols()
    for metric in ('DS','IS','CS'):
        # Historical status annotations may differ; all definitions and assets must match.
        old={k:v for k,v in prior_protocol[metric].items() if k!='status'}
        new={k:v for k,v in current_protocol[metric].items() if k!='status'}
        assert digest(old)==digest(new), ('Earlier metric protocol changed',metric) # normalize JSON list/tuple storage
    assert abs(inception_score(np.zeros((16,10),dtype=np.float32))-1.)<1e-12
    protocol=dict(metrics={k:current_protocol[k] for k in ('DS','IS','CS')},cpu_identity=cpu,gpu_identity=ex.identity,
        source_hashes=source_snapshot,historical_protocol_sha256=sha(ROOT/'outputs/metrics-evaluation/seed0-v1/metric_protocols.json'),
        inception_aggregation='one full collection split, 8 horizontal views per panorama; configurations and prompt subsets remain separate',
        software=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__))
    write(OUT/'protocol.json',protocol)
    with (ROOT/'outputs/metrics-evaluation/seed0-v1/per_image_metrics.csv').open() as stream:
        historical={r['image_sha256']:r for r in csv.DictReader(stream) if r.get('image_sha256')}
    rows=[];missing=[];features={};audited=[]
    for row in expected:
        path=Path(row['directory']);status=read(path/'status.json') if (path/'status.json').exists() else {}
        if status.get('state') not in ('complete','reused') or not row['valid_semantics']:
            missing.append(dict(suite=row['suite'],key=row['key'],reason=status.get('state','missing')));continue
        meta=read(path/'metadata.json');metadata_sha=sha(path/'metadata.json')
        assert metadata_sha==status['metadata_sha256'],('Metadata changed',path)
        image_path=path/'final.png';image_hash=sha(image_path)
        assert image_hash==meta['artifacts']['final.png']==status['image_sha256']
        prompt=meta['prompt'];assert prompt['name']==row['prompt']
        assert sha(ROOT/'prompts'/(row['prompt']+'.txt'))==prompt['original_sha256']
        feature_key=digest(dict(image=image_hash,protocol=ex.identity));feature_path=OUT/'cache/features'/(feature_key+'.npz');fm=feature_path.with_suffix('.json')
        feature_values=None;origin=None
        if fm.exists() and feature_path.exists():
            m=read(fm)
            if m['key']==feature_key and m['sha256']==sha(feature_path):
                with np.load(feature_path,allow_pickle=False) as data:feature_values=dict(data)
                origin=m['origin']
        rgb=None
        if feature_values is None:
            feature_values,origin=legacy_features(row,image_hash,ex.identity,cpu)
            if feature_values is None:
                with Image.open(image_path) as image:
                    assert image.size==(4096,2048),('Unexpected ERP size',image_path,image.size)
                    rgb=np.asarray(image.convert('RGB'))
                feature_values=ex.image_features(rgb);origin='fresh extraction'
            assert feature_values['logits'].shape[0]==8 and feature_values['clip_views'].shape[0]==10
            save_features(feature_path,feature_values);write(fm,dict(key=feature_key,sha256=sha(feature_path),origin=origin))
        feature_hash=sha(feature_path)
        score_key=digest(dict(image=image_hash,prompt=prompt['effective_sha256'],cpu=cpu,gpu=ex.identity))
        cache=OUT/'cache/scores'/(score_key+'.json')
        if cache.exists():
            scores=read(cache);assert scores['key']==score_key and scores['feature_sha256']==feature_hash
        else:
            if rgb is None:
                with Image.open(image_path) as image:
                    assert image.size==(4096,2048);rgb=np.asarray(image.convert('RGB'))
            cs,detail=ex.cs(feature_values,prompt)
            scores=dict(key=score_key,DS=float(ds(rgb)),CS=cs['CS'],cs_details=detail,feature_sha256=feature_hash)
            write(cache,scores)
        assert np.isfinite(scores['DS']) and np.isfinite(scores['CS'])
        if row['suite']=='blending20' and row['mode']=='rgb':
            h=historical[image_hash];errors={k:abs(scores[k]-float(h[k])) for k in ('DS','CS')}
            assert max(errors.values())<1e-4,('Historical evaluator mismatch',row['prompt'],errors)
            audited.append(dict(prompt=row['prompt'],image_sha256=image_hash,differences=errors))
        result={k:v for k,v in row.items() if k not in ('valid_semantics','directory')}
        result.update(DS=scores['DS'],CS=scores['CS'],cs_details=scores['cs_details'],image_sha256=image_hash,
            metadata_sha256=metadata_sha,effective_prompt_sha256=prompt['effective_sha256'],image_path=str(image_path),
            feature_path=str(feature_path),feature_sha256=feature_hash,feature_origin=origin,
            cpu_protocol=cpu,gpu_protocol=ex.identity,relative_tolerance=meta.get('fusion',{}).get('relative_tolerance',0.),
            absolute_tolerance=meta.get('fusion',{}).get('absolute_tolerance',0.),camera_geometry_sha256=meta.get('camera_geometry_sha256'),
            seed=0,steps=20,ERP_resolution=[2048,4096],generation_job=status.get('job'))
        rows.append(result);features[(row['suite'],row['key'])]=feature_values
        write(OUT/row['suite']/'cases'/(row['key']+'.json'),result)
        print('SCORED',row['suite'],row['key'],'DS',scores['DS'],'CS',scores['CS'],flush=True)
    groups=group_rows('refinement64',rows,features,FACTOR_PROMPTS)+group_rows('blending20',rows,features,blendmanifest['prompts'])
    assert all(sha(path)==value for path,value in source_snapshot.items()),'Evaluator changed during run'
    write(OUT/'baseline-validation.json',dict(passed=True,checked_rgb_controls=len(audited),rows=audited))
    write(OUT/'input-inventory.json',dict(expected=expected,refinement_manifest_sha256=sha(REF/'manifest.json'),blending_manifest_sha256=sha(BLEND/'manifest.json')))
    exports(rows,missing,groups,protocol)
    print('EVALUATION_FINISHED',len(rows),'of',len(expected),'missing',missing,flush=True)
    if missing and not args.allow_partial:raise RuntimeError('Requested matched evaluation incomplete; see status.json')

if __name__=='__main__':main()
