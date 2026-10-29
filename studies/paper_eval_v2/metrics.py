"""Group 1 evaluation with explicit legacy/common namespaces and content caches."""
import argparse
import copy
import csv
import io
import json
import os
import sqlite3
from pathlib import Path

from studies.panorama_metrics.common import atomic, digest, read, sha
from .rendering import HORIZONTAL, VIEWS14, protocol_definitions as render_protocols


def protocol_definitions():
    from studies.panorama_metrics.protocols import protocols
    from .statistics import protocol_definitions as statistical_protocols
    legacy = protocols()
    shared = render_protocols()['common_v2']
    common = copy.deepcopy({key: legacy[key] for key in ('DS','CS','IS','Seam-SSIM','Seam-Sobel')})
    common['DS']['resolution'] = 'common_v2 ERP2048x1024; fixed periodic pixel-center resampling'
    common['CS'].update(renderer=shared, directional='14 views; +90,+45,0,-45,-90 map to lines0,1,2,3,4; average each band then all five bands equally', text_policies={'common_v2_short_scoring_text':'frozen record.scoring_prompt descriptions; identical per scene/band across methods; token policy retained', 'common_v2_generation_text':'original full generation text with token-count-weighted chunks; separately labeled', 'legacy_generation_text':'historical full text and no truncation'})
    common['IS'].update(renderer=shared, representation='common_v2 eight horizontal views; one collection split, no shuffle')
    for name in ('Seam-SSIM','Seam-Sobel'):
        common[name]['source_erp'] = 'common_v2 ERP2048x1024'
    return dict(group1=dict(legacy={key:legacy[key] for key in common}, common_v2=common,
        MUSIQ=dict(model='MUSIQ KonIQ-10k', implementation='pinned PyIQA port verified tensor-for-tensor against official Google checkpoint',
            views=14, preprocessing='512-square RGB[0,1]; native/224/384 multiscale; patch32 stride32; no original-resolution truncation',
            direction='higher', units='original KonIQ MOS', aggregation='14-view mean; two-pole mean separately',
            unavailable='blank with explicit dependency status'),
        directions={'DS':'lower','CS-H':'higher','CS-north':'higher','CS-south':'higher','CS-polar':'higher',
            'CS-directional':'higher','IS':'higher','MUSIQ':'higher','MUSIQ-polar':'higher',
            'Seam-SSIM':'higher','Seam-Sobel':'texture diagnostic; no quality ranking'},
        interpretation='DS measures wrap discontinuity and may favor smoothness; Sobel measures edge-adjacent texture, not complete geometric consistency',
        **statistical_protocols()))


def _connect(run):
    path = Path(run)/'group1/cache.sqlite3'
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=120)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    connection.execute('CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, sha256 TEXT NOT NULL, data BLOB NOT NULL)')
    return connection


def _cached(connection, key):
    row = connection.execute('SELECT payload FROM scores WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else None


def _store(connection, key, value):
    connection.execute('INSERT OR REPLACE INTO scores VALUES (?,?)', (key,json.dumps(value,sort_keys=True,allow_nan=False)))
    connection.commit()


def save_features(connection, key, features, run):
    import hashlib
    import numpy as np
    stream = io.BytesIO()
    np.savez_compressed(stream, **features)
    raw = stream.getvalue()
    checksum = hashlib.sha256(raw).hexdigest()
    connection.execute('INSERT OR REPLACE INTO features VALUES (?,?,?)', (key, checksum, raw))
    connection.commit()
    return dict(feature_path='sqlite://'+str(Path(run)/'group1/cache.sqlite3')+'#'+key, feature_sha256=checksum)


def load_features(record):
    import hashlib
    import numpy as np
    path = record['feature_path']
    if path.startswith('sqlite://'):
        database, key = path[len('sqlite://'):].rsplit('#', 1)
        with sqlite3.connect('file:'+database+'?mode=ro', uri=True) as connection:
            row = connection.execute('SELECT data FROM features WHERE key=?', (key,)).fetchone()
        if row is None:
            raise FileNotFoundError(key)
        raw = bytes(row[0])
    else:
        raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record['feature_sha256']
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        return dict(archive)


def _base(row, namespace, identity):
    return dict(case_id=row['case_id'], panorama_id=row.get('panorama_id',row['case_id']),
        method_id=row['method_id'], prompt_id=row['prompt_id'], prompt_name=row.get('prompt_name'),
        seed=row.get('seed',row.get('parameters',{}).get('seed',0)), image_sha256=row['image_sha256'],
        effective_prompt_sha256=row['prompt']['effective_sha256'], namespace=namespace,
        protocol_sha256=identity, metrics={})


def _legacy_index():
    from .common import ROOT
    from studies.panorama_metrics.features import gpu_identity
    from studies.panorama_metrics.protocols import cpu_identity
    lookup = {}
    source = ROOT/'outputs/10.15layout24-flux216/seed0-v1/evaluation'
    if (source/'metrics.json').exists():
        protocol = read(source/'metric_protocol.json')
        if protocol['gpu_identity'] == gpu_identity() and protocol['cpu_identity'] == cpu_identity():
            table_sha = sha(source/'metrics.json')
            for row in read(source/'metrics.json')['rows']:
                if row.get('scoring_status') == 'complete':
                    lookup[(row['image_sha256'],row['effective_prompt_sha256'])] = dict(row,
                        legacy_source=str(source/'metrics.json'), legacy_source_sha256=table_sha)
    source = ROOT/'outputs/10.9gradient-metrics/seed0-v1'
    if (source/'protocol.json').exists():
        protocol = read(source/'protocol.json')
        if protocol['gpu_identity'] == gpu_identity() and protocol['cpu_identity'] == cpu_identity():
            table_sha = sha(source/'per-image-metrics.csv')
            with (source/'per-image-metrics.csv').open() as stream:
                for row in csv.DictReader(stream):
                    lookup.setdefault((row['image_sha256'],row['effective_prompt_sha256']),dict(row,
                        legacy_source=str(source/'per-image-metrics.csv'),legacy_source_sha256=table_sha))
    return lookup


def _verify_source(row):
    assert sha(row['image_path']) == row['image_sha256'], row['case_id']
    if row.get('metadata_path') and row.get('metadata_sha256'):
        assert sha(row['metadata_path']) == row['metadata_sha256'], row['case_id']
    path = Path(row['prompt_path'])
    if not path.is_absolute():
        from .common import ROOT
        path = ROOT/path
    assert sha(path) == row['prompt']['original_sha256'] == row['prompt']['effective_sha256']
    assert path.read_text().splitlines() == row['prompt']['effective_lines']


def _cpu(row, rgb, identity, legacy):
    from studies.panorama_metrics.seams import ds, cubemap_seams
    from studies.panorama_metrics.render import CUBE, render
    from .rendering import resample_common
    common = resample_common(rgb)
    value = _base(row,'common_v2',identity)
    value.update(source_resolution=list(rgb.shape[1::-1]), evaluation_resolution=[2048,1024],
        upsampled=rgb.shape[1]<2048 or rgb.shape[0]<1024)
    value['metrics'] = dict(DS=ds(common), **{k:v for k,v in cubemap_seams(render(common,CUBE)).items() if not k.startswith('edge_')})
    if rgb.shape[:2] != (2048,4096):
        return [value]
    old = _base(row,'legacy',identity)
    old['text_policy_id'] = 'legacy_generation_text'
    old['scoring_text_sha256'] = row['prompt']['effective_sha256']
    old['metrics']['DS'] = ds(rgb)
    old['source_resolution'] = list(rgb.shape[1::-1])
    if legacy:
        error = abs(old['metrics']['DS']-float(legacy['DS']))
        assert error <= 1e-4, ('Legacy DS regression', row['case_id'],error)
        old['legacy_regression'] = dict(DS_absolute_error=error,tolerance=1e-4,
            source_path=legacy['legacy_source'],source_sha256=legacy['legacy_source_sha256'])
        old['metrics']['CS-H'] = float(legacy['CS'])
        if legacy.get('feature_path') and Path(legacy['feature_path']).exists() and sha(legacy['feature_path']) == legacy['feature_sha256']:
            old.update(feature_path=legacy['feature_path'], feature_sha256=legacy['feature_sha256'],
                feature_origin='verified compatible historical cache')
    return [value,old]


def _gpu(row, rgb, identity, extractor, musiq, connection, run, common_protocol):
    import numpy as np
    from PIL import Image
    from .rendering import resample_common, render_common_views, render_native_five
    torch = extractor.torch
    value = _base(row,'common_v2',identity)
    common = resample_common(rgb)
    features = extractor.image_features(common)
    views = render_common_views(rgb)
    with torch.inference_mode():
        inputs = torch.stack([extractor.preprocess(Image.fromarray(view)) for view in views]).to(extractor.device)
        embeddings = extractor.clip.encode_image(inputs).float()
        embeddings /= embeddings.norm(dim=-1,keepdim=True)
        features['clip_directional'] = embeddings.cpu().numpy()
        native = render_native_five(rgb)
        # Native-FID uses its own full native ERP sampling, never common_v2 crops.
        native_features = []
        for begin in range(0,5,2):
            tensor = torch.from_numpy(native[begin:begin+2].transpose(0,3,1,2).copy()).to(extractor.device)
            native_features.append(extractor.inception(tensor)[0].cpu().numpy())
        features['native_five'] = np.concatenate(native_features)
        if musiq is not None:
            scores = []
            for begin in range(0,14,2):
                tensor = torch.from_numpy(views[begin:begin+2].transpose(0,3,1,2).copy()).to(extractor.device).float()/255.
                scores.extend(musiq(tensor).flatten().cpu().tolist())
            features['musiq'] = np.asarray(scores,dtype=np.float32)
    scoring = row.get('scoring_prompt',row['prompt'])
    assert len(scoring['effective_lines']) == 5
    value['text_policy_id'] = 'common_v2_short_scoring_text' if 'scoring_prompt' in row else 'common_v2_generation_text'
    value['scoring_text_sha256'] = scoring['effective_sha256']
    scores, detail = extractor.cs(features,scoring)
    value['metrics'] = {'CS-H':scores['CS'],'CS-north':scores['CS_north'],
        'CS-south':scores['CS_south'],'CS-polar':scores['CS_polar']}
    texts, tokenization = zip(*(extractor.text_embedding(scoring['effective_lines'][view['prompt_band']]) for view in VIEWS14))
    similarities = (features['clip_directional']*np.stack(texts)).sum(-1)*100.
    bands = [float(np.mean([score for score,view in zip(similarities,VIEWS14) if view['prompt_band']==band])) for band in range(5)]
    value['metrics']['CS-directional'] = float(np.mean(bands))
    value['tokenization'] = dict(horizontal_polar=detail['tokenization'],directional=list(tokenization),truncated=False)
    per_view = []
    for index,view in enumerate(VIEWS14):
        item = dict(view,CS=float(similarities[index]))
        if 'musiq' in features:
            item['MUSIQ'] = float(features['musiq'][index])
        per_view.append(item)
    if 'musiq' in features:
        value['metrics'].update(MUSIQ=float(features['musiq'].mean()),
            **{'MUSIQ-polar':float(features['musiq'][-2:].mean())})
    value['views14'] = per_view
    value['horizontal_cs'] = detail['views'][:8]
    key = digest(dict(image=row['image_sha256'],protocol=common_protocol,evaluator=identity,extractor=extractor.identity,
                      native_protocol=render_protocols()['native_five_v2']))
    value.update(save_features(connection,key,features,run))
    value['feature_identity'] = key
    return value


def _csv(path, records):
    path = Path(path)
    fields = sorted({key for row in records for key in row})
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key:json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value for key,value in row.items()})
    os.replace(temporary,path)


def _aggregate(run, records, cohorts=None):
    from .statistics import summarize
    from studies.panorama_metrics.numerics import inception_score
    import numpy as np
    run = Path(run)
    cohorts = cohorts or read(run/'inventory/cohorts.json')
    selected_ids = {row['case_id'] for row in cohorts['selected_cells']}
    records = [row for row in records if row['case_id'] in selected_ids]
    selected = {row['case_id']:row for row in records}
    with _connect(run) as connection:
        values = [json.loads(row[0]) for row in connection.execute('SELECT payload FROM scores')]
    active_identity = read(run/'group1/protocol.json')['identity']
    merged = {}
    for value in values:
        if value.get('marker') or value.get('case_id') not in selected or value.get('state')=='failed' or value.get('protocol_sha256') != active_identity:
            continue
        if value['image_sha256'] != selected[value['case_id']]['image_sha256']:
            continue
        key = (value['case_id'],value['namespace'])
        old = merged.setdefault(key,dict(value,metrics={}))
        old['metrics'].update(value['metrics'])
        for name in ('feature_path','feature_sha256','views14','horizontal_cs','tokenization','legacy_regression','text_policy_id','scoring_text_sha256','source_resolution','evaluation_resolution','upsampled'):
            if name in value:
                old[name] = value[name]
    per_panorama = list(merged.values())
    per_view = []
    for row in per_panorama:
        for view in row.get('views14',[]):
            per_view.append(dict(case_id=row['case_id'],method_id=row['method_id'],prompt_id=row['prompt_id'],
                                 namespace=row['namespace'],**view))
    cohorts = cohorts or read(run/'inventory/cohorts.json')
    groups, pairs, coverage = [], [], []
    for namespace in ('legacy','common_v2'):
        rows = [row for row in per_panorama if row['namespace']==namespace]
        for cohort_id,cohort in cohorts['cohorts'].items():
            prompts = cohort['prompt_ids']
            if not prompts:
                continue
            for metric in sorted({key for row in rows for key in row['metrics']}):
                policies = sorted({row.get('text_policy_id','unspecified') for row in rows if metric in row['metrics']}) if metric.startswith('CS') else ['not_applicable']
                for policy in policies:
                    metric_rows = [row for row in rows if row.get('text_policy_id','unspecified')==policy] if metric.startswith('CS') else rows
                    # Every method must score the same canonical text for a prompt.
                    if metric.startswith('CS'):
                        for prompt_id in prompts:
                            hashes={row.get('scoring_text_sha256',row['effective_prompt_sha256']) for row in metric_rows if row['prompt_id']==prompt_id}
                            if len(hashes)>1:raise ValueError('Scoring text differs across methods for '+prompt_id)
                    grouped,paired = summarize(metric_rows,metric,cohort_id,prompts)
                    groups.extend(dict(namespace=namespace,text_policy_id=policy,**item) for item in grouped)
                    pairs.extend(dict(namespace=namespace,text_policy_id=policy,**item) for item in paired)
            for method in sorted({row['method_id'] for row in records}):
                expected = [row for row in records if row['method_id']==method and row['prompt_id'] in prompts]
                available = [row for row in rows if row['method_id']==method and row['prompt_id'] in prompts]
                coverage.append(dict(namespace=namespace,cohort_id=cohort_id,method_id=method,
                    expected_prompt_count=len(prompts), scored_prompt_ids=sorted({row['prompt_id'] for row in available}),
                    source_states={state:sum(row['state']==state for row in expected) for state in ('complete','missing','invalid','ambiguous','configuration-incompatible')}))
                usable = [row for row in available if 'feature_path' in row]
                if len(usable)<2:
                    continue
                # IS remains a collection statistic. Only exact matched declared
                # prompt cohorts are published; partial extraction stays pending.
                if set(row['prompt_id'] for row in usable) != set(prompts):
                    continue
                logits = np.concatenate([load_features(row)['logits'] for row in sorted(usable,key=lambda r:(r['prompt_id'],str(r['seed'])))])
                groups.append(dict(namespace=namespace,cohort_id=cohort_id,method_id=method,metric='IS',
                    mean=float(inception_score(logits)),ci95=[None,None],prompt_count=len(prompts),
                    panorama_count=len(usable),view_count=len(logits),prompt_ids=prompts,
                    uncertainty_status='collection score; split variation is not comparison uncertainty',
                    interpretation='descriptive tiny collection' if len(prompts)<=4 else 'exploratory dependent crop collection'))
    result = dict(per_panorama=per_panorama,per_view=per_view,groups=groups,paired_comparisons=pairs,coverage=coverage)
    directory = run/'group1'
    for filename,key in (('per_panorama_metrics','per_panorama'),('per_view_metrics','per_view'),('group_metrics','groups'),('paired_comparisons','paired_comparisons'),('coverage','coverage')):
        atomic(directory/(filename+'.json'),result[key])
        _csv(directory/(filename+'.csv'),result[key])
    failures = [row for row in values if row.get('state')=='failed']
    atomic(directory/'failures.json',failures)
    lines = ['# Group 1 numerical evaluation', '',
        'Legacy native-4K and common_v2 ERP2048x1024 results have separate namespaces.',
        'DS measures wrap discontinuity and may favor smoothness. Seam-Sobel measures edge-adjacent texture gradients, not complete geometric consistency.',
        'CLIP uses 100 times raw cosine, five unchanged canonical prompt lines, and token-weighted 75-token chunks without truncation.',
        'Views are averaged within panoramas, seeds within prompts, and prompts equally; intervals use 2,000 paired prompt-cluster bootstrap replicates.',
        'IS is a full-collection score with one fixed split. Missing metrics remain blank; compact distribution summaries are exploratory development evidence.', '',
        '| Namespace | Cohort | Method | Metric | Mean | Prompts |', '|---|---|---|---|---:|---:|']
    tex = ['% Development evidence; intervals in machine-readable paired_comparisons.json',
           r'\begin{tabular}{llllrr}', r'Protocol & Cohort & Method & Metric & Mean & Prompts \\']
    for group in groups:
        number = '' if group.get('mean') is None else '%.6f' % group['mean']
        fields = [group['namespace'], group['cohort_id'], group['method_id'], group['metric'], number, str(group['prompt_count'])]
        lines.append('| '+' | '.join(fields)+' |')
        tex.append(' & '.join(value.replace('_', r'\_') for value in fields)+r' \\')
    tex.append(r'\end{tabular}')
    from .common import atomic_text
    atomic_text(directory/'report.md','\n'.join(lines)+'\n')
    atomic_text(directory/'results.tex','\n'.join(tex)+'\n')
    atomic(directory/'status.json',dict(scored_panorama_namespaces=len(per_panorama),source_case_count=len(records),
        common_v2_DS=sum(row['namespace']=='common_v2' and 'DS' in row['metrics'] for row in per_panorama),
        common_v2_CS=sum(row['namespace']=='common_v2' and 'CS-H' in row['metrics'] for row in per_panorama),
        MUSIQ=sum('MUSIQ' in row['metrics'] for row in per_panorama),failures=len(failures),
        missing_values='blank, never zero',raw_feature_caches='content and protocol addressed SQLite',generation_performed=False))
    return result


def aggregate(run, records, cohorts=None):
    from .common import lock
    with lock(Path(run)/'group1/aggregate.lock'):
        return _aggregate(run,records,cohorts)


def evaluate(run, records, registry=None, *, stage='cpu', cohorts=None, limit=None):
    import numpy as np
    import cv2
    from PIL import Image
    from .metric_assets import status, verify_musiq
    from studies.panorama_metrics.features import Extractor
    cv2.setNumThreads(1)
    run = Path(run)
    definition = protocol_definitions()
    asset = status(run)
    identity = digest(dict(protocol=definition,registry=registry,assets=asset,
                           evaluator_source=sha(__file__)))
    atomic(run/'group1/protocol.json',dict(definitions=definition,registry=registry,assets=asset,identity=identity))
    if stage=='aggregate':
        return aggregate(run,records,cohorts)
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Metric execution requires a compute allocation')
    extractor, musiq = None,None
    if stage=='gpu':
        import torch
        assert torch.cuda.is_available()
        torch.set_num_threads(4)
        extractor = Extractor()
        assert extractor.clip is not None and extractor.inception is not None
        if asset.get('musiq',{}).get('status') in ('weights_available_validation_pending','available'):
            try:
                musiq, evidence = verify_musiq(asset['musiq'])
                musiq = musiq.to('cuda').eval()
                atomic(run/'group1/assets/musiq-validation.json',dict(passed=True,**evidence))
            except Exception as error:
                atomic(run/'group1/assets/musiq-validation.json',dict(passed=False,error=str(error)))
    cohorts = cohorts or read(run/'inventory/cohorts.json')
    selected_ids = {row['case_id'] for row in cohorts['selected_cells']}
    records = [row for row in records if row['case_id'] in selected_ids]
    legacy = _legacy_index()
    failures = []
    with _connect(run) as connection:
        for index,row in enumerate(records):
            if row['state']!='complete' or not row.get('eligible',True):
                continue
            if limit is not None and index >= limit:
                break
            key = digest(dict(case=row['case_id'],image=row['image_sha256'],prompt=row['prompt']['effective_sha256'],scoring_prompt=row.get('scoring_prompt'),protocol=identity,stage=stage))
            if _cached(connection,key) is not None:
                continue
            try:
                _verify_source(row)
                with Image.open(row['image_path']) as image:
                    rgb = np.asarray(image.convert('RGB'))
                if stage=='cpu':
                    values = _cpu(row,rgb,identity,legacy.get((row['image_sha256'],row['prompt']['effective_sha256'])))
                    for value in values:
                        _store(connection,key+':'+value['namespace'],value)
                    _store(connection,key,dict(marker=True,case_id=row['case_id']))
                elif stage=='gpu':
                    value = _gpu(row,rgb,identity,extractor,musiq,connection,run,definition)
                    _store(connection,key,value)
                else:
                    raise ValueError('Unknown evaluation stage '+stage)
                print('METRIC_COMPLETE',stage,index,row['case_id'],flush=True)
            except Exception as error:
                failure = dict(case_id=row['case_id'],stage=stage,state='failed',error=type(error).__name__+': '+str(error))
                failures.append(failure)
                _store(connection,key+':failure',failure)
                print('METRIC_FAILED',json.dumps(failure),flush=True)
    result = aggregate(run,records,cohorts)
    return dict(**result,execution_failures=failures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--stage',choices=('cpu','gpu','aggregate','assets','render','reference','native-reference','distribution'),default='cpu')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--reference-manifest',type=Path)
    parser.add_argument('--bootstrap-seconds',type=float,default=300.)
    args = parser.parse_args()
    if args.stage=='assets':
        from .metric_assets import prepare
        print(json.dumps(prepare(args.run),sort_keys=True))
        return
    inventory = read(args.run/'inventory/records.json')
    records = inventory['records'] if isinstance(inventory,dict) else inventory
    if args.stage=='native-reference':
        from .distribution import extract_native_reference
        if args.reference_manifest is None: parser.error('--reference-manifest required')
        extract_native_reference(args.run,args.reference_manifest)
        return
    if args.stage=='reference':
        from .distribution import extract_real_reference
        if args.reference_manifest is None: parser.error('--reference-manifest required')
        extract_real_reference(args.run,args.reference_manifest,args.limit)
        return
    if args.stage=='distribution':
        from .distribution import evaluate as distribution_evaluate
        distribution_evaluate(args.run,records,read(args.run/'protocol_registry.json'),reference_manifest=args.reference_manifest,max_bootstrap_seconds=args.bootstrap_seconds)
        return
    if args.stage=='render':
        from .rendering import cache_views
        assert os.environ.get('SLURM_JOB_ID')
        selected = {row['case_id'] for row in read(args.run/'inventory/cohorts.json')['selected_cells']}
        views = cache_views(args.run,[row for row in records if row['case_id'] in selected and row.get('eligible',True)])
        atomic(args.run/'render_cache/views.json',dict(records=views))
        return
    registry = read(args.run/'protocol_registry.json')
    evaluate(args.run,records,registry,stage=args.stage,limit=args.limit)

if __name__=='__main__':
    main()
