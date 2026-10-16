"""Read-only legacy feature reuse and exact matched 24-scene DS/CS/IS groups."""
import argparse
import csv
import fcntl
import os
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
from studies.layout24.common import ROOT, OUT, VARIANTS, read, write, sha, digest, folder, complete, freeze_check
from studies.followup24.common import PROMPTS24, PROMPTS20, NEW4, prompt

EVAL = OUT / 'evaluation'


def csv_write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    os.replace(str(temp), str(path))


def summarize(rows, features, names=PROMPTS24, subset='all24'):
    from studies.panorama_metrics.numerics import inception_score
    output = []
    for variant in VARIANTS:
        group = [next(r for r in rows if r['variant'] == variant and r['prompt'] == name) for name in names]
        scored = [r for r in group if r['scoring_status'] == 'complete']
        done = len(scored) == len(names)
        times = [r['generation_seconds'] for r in group if r.get('generation_seconds') is not None]
        record = dict(variant=variant, subset=subset, camera_count=group[0]['camera_count'],
                      completed_panorama_count=sum(r['completion_status'] == 'complete' for r in group),
                      scored_panorama_count=len(scored), expected_panorama_count=len(names),
                      matched_prompt_ids=list(names), scored_prompt_ids=[r['prompt'] for r in scored],
                      complete=done, provisional=not done, view_count=8*len(scored), expected_view_count=8*len(names),
                      mean_DS=None, mean_CS=None, IS=None, mean_generation_seconds=None,
                      timed_panorama_count=len(times), missing_timing_count=len(names)-len(times))
        if done:
            logits = np.concatenate([features[r['case_id']]['logits'] for r in group])
            assert logits.shape[0] == 8*len(names)
            record.update(mean_DS=float(np.mean([r['DS'] for r in group])),
                          mean_CS=float(np.mean([r['CS'] for r in group])), IS=float(inception_score(logits)),
                          mean_generation_seconds=float(np.mean(times)) if times else None)
        output.append(record)
    return output


def feature_candidates(gpu_identity):
    """Index immutable historical records, never invoke their current-source gates."""
    candidates = {}
    def add(record, source):
        image = record.get('image_sha256')
        if image and record.get('feature_path') and record.get('feature_sha256'):
            candidates.setdefault(image, []).append(dict(record, record_source=str(source)))
    global_root = ROOT / 'outputs/metrics-evaluation/seed0-v1'
    for path in sorted((global_root/'cache/gpu').glob('*.json')):
        record = read(path)
        if record.get('validated') and record.get('protocol_sha256') == gpu_identity:
            add(record, path)
    refreshed = ROOT/'outputs/10.9gradient-metrics/seed0-v1/per-image-metrics.csv'
    if refreshed.exists():
        with refreshed.open() as stream:
            for record in csv.DictReader(stream):
                if record.get('gpu_protocol') == gpu_identity:
                    add(record, refreshed)
    # Only known evaluator summaries carrying a matching immutable GPU protocol.
    for relative in ('10.9ournew-flux28/seed0-v1', '10.14mean-max-old89-controls/seed0-v1'):
        root = ROOT/'outputs'/relative/'evaluation'
        protocol_path, metrics_path = root/'metric_protocol.json', root/'metrics.json'
        if protocol_path.exists() and metrics_path.exists():
            protocol, metrics = read(protocol_path), read(metrics_path)
            if protocol.get('gpu_identity') != gpu_identity or metrics.get('protocol_sha256') != digest(protocol):
                continue
            for record in metrics['rows']:
                if record.get('scoring_status') == 'complete':add(record, metrics_path)
    return candidates


def compatible_features(candidates, image_hash, prompt_hash):
    for candidate in candidates.get(image_hash, []):
        if candidate.get('effective_prompt_sha256') != prompt_hash:continue
        path = Path(candidate['feature_path'])
        if not path.is_file() or sha(path) != candidate['feature_sha256']:continue
        with np.load(path, allow_pickle=False) as archive:features = dict(archive)
        if features.get('logits', np.empty((0,))).shape[0] != 8:continue
        if features.get('clip_views', np.empty((0,))).shape[0] != 10:continue
        if not all(np.isfinite(features[k]).all() for k in ('logits', 'clip_views')):continue
        return features, dict(feature_path=str(path), feature_sha256=sha(path),
                              record_path=candidate['record_source'], record_sha256=sha(candidate['record_source']))
    return None, None


def measured_time(meta):
    for key in ('generation_seconds', 'generation_region_seconds'):
        value = meta.get(key)
        if isinstance(value, (int, float)) and value > 0:return float(value)
    return None


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--allow-partial', action='store_true');args=parser.parse_args()
    import torch, cv2
    from studies.panorama_metrics.features import Extractor, save_features
    from studies.panorama_metrics.protocols import protocols, cpu_identity
    from studies.panorama_metrics.seams import ds
    from studies.panorama_metrics.numerics import inception_score
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available(), 'GPU Slurm allocation required'
    EVAL.mkdir(parents=True, exist_ok=True)
    lock = (EVAL/'lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = freeze_check()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    scientific = {str(p):sha(p) for p in sorted((ROOT/'studies/panorama_metrics').glob('*.py'))}
    scientific[str(Path(__file__).resolve())] = sha(__file__)
    historical = ROOT/'outputs/metrics-evaluation/seed0-v1'
    current, previous = protocols(), read(historical/'metric_protocols.json')
    for metric in ('DS','CS','IS'):
        assert digest({k:v for k,v in current[metric].items() if k!='status'}) == digest({k:v for k,v in previous[metric].items() if k!='status'}), metric
    extractor = Extractor();assert extractor.clip is not None and extractor.inception is not None
    gate = read(historical/'validation/gpu.json')
    assert gate['passed'] and gate['protocol_sha256'] == extractor.identity
    assert abs(inception_score(np.zeros((192,1008), dtype=np.float32))-1.) < 1e-12
    cpu = cpu_identity()
    protocol = dict(metrics={k:current[k] for k in ('DS','CS','IS')}, cpu_identity=cpu,
                    gpu_identity=extractor.identity, source_hashes=scientific,
                    historical_protocol_sha256=sha(historical/'metric_protocols.json'),
                    historical_gpu_gate_sha256=sha(historical/'validation/gpu.json'),
                    input='Original final.png uint8 display image, no resizing before established view extraction',
                    aggregation='Nine separate 24-panorama/192-horizontal-view collections; original20=160 views and new4=32 views independently; one full split, never average per-image IS',
                    prompt_ids=list(PROMPTS24), software=dict(torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__))
    protocol_hash=digest(protocol);write(EVAL/'metric_protocol.json', protocol)
    # Snapshot completion once. New results appearing during this pass are handled by the next pass.
    snapshot=[]
    for row in manifest['rows']:
        valid=complete(row)
        snapshot.append(dict(row, snapshot_complete=valid,
                             snapshot_image_sha256=sha(folder(row)/'final.png') if valid else None,
                             snapshot_metadata_sha256=sha(folder(row)/'metadata.json') if valid else None))
    write(EVAL/'input-snapshot.json', dict(manifest_sha256=sha(OUT/'manifest.json'),rows=snapshot,job=os.environ['SLURM_JOB_ID']))
    candidates=feature_candidates(extractor.identity);records=[];features={};errors=[]
    for row in snapshot:
        info,_=prompt(row['prompt']);directory=folder(row)
        record={k:row[k] for k in ('case_id','prompt','prompt_path','variant','camera_count','layout','subset')}
        record.update(effective_prompt_sha256=info['effective_sha256'], completion_status='complete' if row['snapshot_complete'] else 'missing',
                      scoring_status='pending_generation', DS=None, CS=None, image_sha256=None,generation_seconds=None,error=None)
        try:
            if not row['snapshot_complete']:records.append(record);continue
            image_hash=row['snapshot_image_sha256'];image_path=directory/'final.png'
            meta=read(directory/'metadata.json');status=read(directory/'status.json')
            # Reuse metadata already carries verified normalized timing. Its original
            # provenance can be a Slurm log rather than JSON; never reinterpret it.
            key=digest(dict(image=image_hash,prompt=info['effective_sha256'],cpu=cpu,gpu=extractor.identity))
            cache=EVAL/'cache'/(key+'.json');fp=EVAL/'cache'/(key+'.npz');scores=None;values=None;origin=None
            if cache.exists() and fp.exists():
                old=read(cache)
                if old.get('key')==key and old.get('feature_sha256')==sha(fp):
                    with np.load(fp,allow_pickle=False) as archive:values=dict(archive)
                    scores=old;origin=old['feature_origin']
            if scores is None:
                values,origin=compatible_features(candidates,image_hash,info['effective_sha256'])
                with Image.open(image_path) as image:
                    assert image.size==(4096,2048);rgb=np.asarray(image.convert('RGB'))
                if values is None:values=extractor.image_features(rgb);origin=dict(kind='fresh extraction')
                save_features(fp,values)
                cs,detail=extractor.cs(values,info)
                scores=dict(key=key,DS=float(ds(rgb)),CS=float(cs['CS']),cs_details=detail,
                            feature_sha256=sha(fp),feature_origin=origin)
                write(cache,scores)
            assert values['logits'].shape[0]==8 and values['clip_views'].shape[0]==10
            assert np.isfinite(scores['DS']) and np.isfinite(scores['CS'])
            assert sha(image_path)==image_hash and sha(directory/'metadata.json')==row['snapshot_metadata_sha256']
            features[row['case_id']]=values
            record.update(DS=scores['DS'],CS=scores['CS'],generation_seconds=measured_time(meta),
                          scoring_status='complete',image_sha256=image_hash,image_path=str(image_path),
                          metadata_sha256=row['snapshot_metadata_sha256'],reused=bool(status.get('reused')),
                          feature_path=str(fp),feature_sha256=scores['feature_sha256'],feature_origin=origin,
                          protocol_sha256=protocol_hash,cache_key=key)
            print('SCORED',row['case_id'],record['DS'],record['CS'],'features',origin,flush=True)
        except Exception:
            record.update(scoring_status='failed',error=traceback.format_exc());errors.append(dict(case_id=row['case_id'],error=record['error']))
            print('SCORING_FAILED',row['case_id'],record['error'],flush=True)
        records.append(record)
    groups=summarize(records,features);subsets=summarize(records,features,PROMPTS20,'original20')+summarize(records,features,NEW4,'new4')
    assert len(records)==216 and len(groups)==9 and len(subsets)==18
    csv_write(EVAL/'per_image_scores.csv',records,['case_id','prompt','prompt_path','effective_prompt_sha256','variant','camera_count','layout','subset','completion_status','scoring_status','reused','DS','CS','generation_seconds','image_sha256','image_path','metadata_sha256','feature_path','feature_sha256','feature_origin','protocol_sha256','cache_key','error'])
    fields=['variant','camera_count','subset','completed_panorama_count','scored_panorama_count','expected_panorama_count','matched_prompt_ids','scored_prompt_ids','view_count','expected_view_count','complete','provisional','mean_DS','mean_CS','IS','mean_generation_seconds','timed_panorama_count','missing_timing_count']
    csv_write(EVAL/'summary_scores.csv',groups,fields);csv_write(EVAL/'subset_scores.csv',subsets,fields)
    counts=dict(valid_generations=sum(r['completion_status']=='complete' for r in records),DS=sum(r['DS'] is not None for r in records),CS=sum(r['CS'] is not None for r in records),matched_IS=sum(g['IS'] is not None for g in groups))
    passed=counts==dict(valid_generations=216,DS=216,CS=216,matched_IS=9) and not errors
    assert all(sha(p)==h for p,h in scientific.items()),'Evaluator changed during execution'
    freeze_check()
    write(EVAL/'metrics.json',dict(rows=records,groups=groups,subsets=subsets,counts=counts,complete=passed,errors=errors,protocol_sha256=protocol_hash,job=os.environ['SLURM_JOB_ID']))
    write(EVAL/'status.json',dict(state='complete' if passed else 'partial',counts=counts,metric_failures=len(errors),job=os.environ['SLURM_JOB_ID']))
    if passed:write(OUT/'completion.json',dict(passed=True,counts=counts,manifest_sha256=sha(OUT/'manifest.json'),metrics_sha256=sha(EVAL/'metrics.json')))
    print('EVALUATION',counts,'complete',passed,flush=True)
    if not passed and not args.allow_partial:raise SystemExit(1)

if __name__=='__main__':main()
