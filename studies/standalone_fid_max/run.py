"""Extend the frozen 20-view FID evaluation to ten matched 15-scene groups."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import fcntl
import io
import os
from pathlib import Path
import time
import traceback

from studies.standalone_fid import common as original
from studies.standalone_fid.common import ROOT, sha, digest, read, write

OUT = ROOT / 'outputs/10.23fid-standalone85/max-blending-20views'
SOURCE = ROOT / 'studies/standalone_fid_max'
PLAN = OUT / 'plan.json'


def sources():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(SOURCE.iterdir())
            if p.is_file() and p.suffix in ('.py', '.slurm')}


def load_plan():
    original.load_plan()
    plan = read(PLAN)
    assert sha(PLAN) == (OUT / 'plan.sha256').read_text().strip()
    assert plan['source_hashes'] == sources()
    for path, expected in plan['input_hashes'].items():
        assert sha(path) == expected, path
    return plan


def prepare():
    from studies.standalone_fid_max.inputs import build_inputs
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if PLAN.exists():
            print('VALID_PLAN', load_plan()['expected'], flush=True)
            return
        original.load_plan()
        done = read(original.OUT / 'completion.json')
        assert done['passed'] and done['results_sha256'] == sha(original.OUT / 'results.json')
        inputs = build_inputs()
        assert len(inputs['rows']) == 150 and len(inputs['prompts']) == 15
        variants = list(dict.fromkeys(row['variant'] for row in inputs['rows']))
        assert len(variants) == 10
        for variant in variants:
            group = [row for row in inputs['rows'] if row['variant'] == variant]
            assert len(group) == 15 and {row['prompt'] for row in group} == set(inputs['prompts'])
        old_protocol = read(original.OUT / 'protocol.json')
        assert old_protocol['generated']['views_per_panorama'] == 20
        protocol = copy.deepcopy(old_protocol)
        protocol['name'] = 'Twenty-view FID against 85 standalone references, matched 15-scene max-blending comparison'
        protocol['generated'].update(variants=variants, prompts=inputs['prompts'],
                                     panoramas_per_variant=15, views_per_variant=300)
        protocol['metric']['aggregation'] = 'One collection-level FID per method across the same 15 scenes'
        protocol['limitations'] = [value.replace('24 scenes', '15 scenes') for value in protocol['limitations']]
        protocol['limitations'].append('Compare these 15-scene values with the matched 15-scene baselines; the earlier 24-scene scores use a different collection.')
        protocol['tolerance_override'] = inputs.get('tolerance_override')
        protocol['feature_protocol_sha256'] = digest(old_protocol)
        protocol['feature_extraction_identity'] = read(original.OUT / 'results.json')['runtime']['extraction_identity']
        write(OUT / 'input-manifest.json', inputs)
        write(OUT / 'protocol.json', protocol)
        write(OUT / 'reference-manifest.json', read(original.OUT / 'reference-manifest.json'))
        for name in ('logs', 'cache'):
            (OUT / name).mkdir(parents=True, exist_ok=True)
        paths = [OUT / name for name in ('input-manifest.json', 'protocol.json', 'reference-manifest.json')]
        paths += [original.OUT / name for name in ('results.json', 'completion.json', 'protocol.json')]
        plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_hashes=sources(),
                    input_hashes={str(path): sha(path) for path in paths},
                    protocol_sha256=digest(protocol), expected=dict(reference=85, variants=10,
                    panoramas=150, views_per_panorama=20, views_per_variant=300, generated_views=3000),
                    generation_required=False)
        write(PLAN, plan)
        (OUT / 'plan.sha256').write_text(sha(PLAN) + '\n')
        load_plan()
        print('PREPARED_MAX_FID', plan['expected'], flush=True)


def atomic_text(path, text):
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(text)
    os.replace(str(temporary), str(path))


def report(payload):
    rows = payload['rows']
    output = io.StringIO()
    fields = ('variant', 'label', 'alpha', 'FID', 'reference_images', 'panoramas', 'generated_views')
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(OUT / 'results.csv', output.getvalue())
    caption = ('FLUX max-blending comparison on the same 15 scenes. Each method contributes '
               '300 perspective views (20 per ERP, 80-degree FOV) against 85 unique standalone '
               'FLUX references. Lower FID is better. The firefly coarse-color max-blending '
               'alpha = 0.25 case uses the previously authorized solver-tolerance retry. '
               'Scores are exploratory.')
    lines = [r'\begin{table}[t]', r'\centering', r'\caption{' + caption + '}',
             r'\label{tab:flux_max_blending_fid85}', r'\resizebox{\columnwidth}{!}{',
             r'\begin{tabular}{lcc}', r'\toprule', r'Method & $\alpha$ & FID $\downarrow$ \\', r'\midrule']
    ranking = sorted(rows, key=lambda row: row['FID'])
    for row in rows:
        value = '%.3f' % row['FID']
        if row['variant'] == ranking[0]['variant']:
            value = r'\textbf{' + value + '}'
        elif row['variant'] == ranking[1]['variant']:
            value = r'\underline{' + value + '}'
        alpha = '--' if row['alpha'] is None else '%.2f' % row['alpha']
        lines.append(row['label'].replace('_', r'\_') + ' & ' + alpha + ' & ' + value + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '}', r'\end{table}', '']
    atomic_text(OUT / 'results.tex', '\n'.join(lines))
    text = [caption, '', '| Method | alpha | FID ↓ |', '|---|---:|---:|']
    for row in rows:
        text.append('| %s | %s | %.3f |' % (row['label'],
                    '--' if row['alpha'] is None else '%.2f' % row['alpha'], row['FID']))
    text += ['', 'All groups use the same 15 prompts, listed in protocol.json. The original '
             '24-scene, 20-view evaluation is preserved separately at ../seed0-v1/results.json. '
             'Reference images and exact view geometry/preprocessing are unchanged. '
             'Existing compatible panorama features are reused by content hash.',
             '', 'The original code calls max_blending poisson_mean_max, with guidance '
             '(1-alpha)*g_mean + alpha*g_max. lambda_blending denotes screened reference.',
             '', 'Authorized exception: firefly / coarse_color / alpha = 0.25 uses relative '
             'tolerance 2e-4 instead of 1e-4, as in the completed report figures. No new '
             'generation or tolerance changes were performed.', '', 'Limitations:']
    text += payload['limitations'] + ['']
    atomic_text(OUT / 'report.txt', '\n'.join(text))


def evaluate():
    assert os.environ.get('SLURM_JOB_ID'), 'Use a Slurm GPU allocation'
    plan = load_plan()
    with (OUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        runtime = dict(job=os.environ['SLURM_JOB_ID'], cluster=os.environ.get('SLURM_CLUSTER_NAME'),
                       plan_sha256=sha(PLAN), started_utc=datetime.now(timezone.utc).isoformat())
        write(OUT / 'status.json', dict(runtime, state='running', stage='imports'))
        try:
            import numpy as np
            import torch
            import cv2
            import inspect
            import importlib.metadata
            from PIL import Image
            from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
            from studies.panorama_metrics.numerics import frechet
            from studies.standalone_fid.render import render
            from studies.standalone_fid import run as reusable
            torch.set_num_threads(4)
            cv2.setNumThreads(1)
            torch.backends.cudnn.benchmark = False
            assert torch.cuda.is_available()
            old_results = read(original.OUT / 'results.json')
            protocol = read(OUT / 'protocol.json')
            inputs = read(OUT / 'input-manifest.json')
            checkpoint = protocol['features']['checkpoint']
            assert sha(checkpoint['path']) == checkpoint['sha256']
            runtime['software'] = dict(torch=torch.__version__, numpy=np.__version__,
                opencv=cv2.__version__, torch_fidelity=importlib.metadata.version('torch-fidelity'))
            runtime['extractor_source_sha256'] = sha(inspect.getfile(FeatureExtractorInceptionV3))
            runtime['gpu'] = torch.cuda.get_device_name(0)
            extraction_identity = digest(dict(protocol=protocol['feature_protocol_sha256'],
                software=runtime['software'], extractor_source=runtime['extractor_source_sha256'],
                renderer_source=original.load_plan()['source_hashes']['studies/standalone_fid/render.py']))
            assert extraction_identity == protocol['feature_extraction_identity']
            runtime['extraction_identity'] = extraction_identity
            extractor = FeatureExtractorInceptionV3('inception-v3-compat', ['2048'],
                feature_extractor_weights_path=checkpoint['path']).cuda().eval()

            def cached(cache, count):
                assert sha(cache['path']) == cache['sha256']
                with np.load(cache['path'], allow_pickle=False) as archive:
                    values = archive['features']
                assert values.shape == (count, 2048) and np.isfinite(values).all()
                return values

            reference_cache = old_results['reference_feature_cache']
            reference = cached(reference_cache, 85)
            assert abs(frechet(reference, reference)) < 1e-5
            old_caches = {row['image_sha256']: row['feature_cache']
                          for row in old_results['panorama_feature_records']}
            reusable.OUT = OUT  # Reuse the validated cache writer only; never edit historical caches.
            views = protocol['generated']['views']
            assert len(views) == 20
            groups = {variant: [] for variant in protocol['generated']['variants']}
            records, reused, extracted = [], 0, 0
            for index, row in enumerate(inputs['rows']):
                assert sha(row['image_path']) == row['image_sha256']
                assert sha(row['metadata_path']) == row['metadata_sha256']
                if row['image_sha256'] in old_caches:
                    cache = dict(old_caches[row['image_sha256']], reused=True)
                    value = cached(cache, 20)
                    reused += 1
                else:
                    def produce():
                        with Image.open(row['image_path']) as image:
                            assert image.size == (4096, 2048)
                            rgb = np.asarray(image.convert('RGB'))
                        images = [render(rgb, view['yaw'], view['pitch'], size=1024, fov=80.) for view in views]
                        batches = []
                        for start in range(0, 20, 8):
                            batch = np.stack(images[start:start + 8])
                            tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2).copy()).cuda()
                            with torch.inference_mode():
                                output, = extractor(tensor)
                            batches.append(output.cpu().numpy())
                        return np.concatenate(batches)
                    key = digest(dict(kind='panorama_views', extractor=extraction_identity, image=row['image_sha256']))
                    value, cache = reusable.feature_cache(key, 20, produce)
                    extracted += not cache['reused']
                    reused += cache['reused']
                groups[row['variant']].append(value)
                records.append(dict(row, feature_cache=cache))
                write(OUT / 'status.json', dict(runtime, state='running', stage='features',
                      completed_panoramas=index + 1, expected_panoramas=150,
                      reused_feature_sets=reused, newly_extracted_feature_sets=extracted,
                      elapsed_seconds=time.monotonic() - started))
                print('MAX_FID_FEATURES', index + 1, '/', 150, row['variant'], row['prompt'], flush=True)
            results = []
            for variant, values in groups.items():
                assert len(values) == 15
                features = np.concatenate(values)
                assert features.shape == (300, 2048)
                row = next(row for row in inputs['rows'] if row['variant'] == variant)
                value = frechet(reference, features)
                results.append(dict(variant=variant, label=row['label'], alpha=row['alpha'],
                    reference_mode=row['reference_mode'], FID=value, reference_images=85,
                    panoramas=15, generated_views=300))
                print('MAX_FID_RESULT', variant, value, flush=True)
            load_plan()
            payload = dict(complete=True, rows=results, metric=protocol['name'],
                           protocol_sha256=digest(protocol), reference_feature_cache=reference_cache,
                           panorama_feature_records=records, tolerance_override=inputs.get('tolerance_override'),
                           limitations=protocol['limitations'], runtime=dict(runtime,
                           elapsed_seconds=time.monotonic() - started, reused_feature_sets=reused,
                           newly_extracted_feature_sets=extracted))
            write(OUT / 'results.json', payload)
            report(payload)
            write(OUT / 'completion.json', dict(passed=True, plan_sha256=sha(PLAN),
                  results_sha256=sha(OUT / 'results.json'), variants=10, panoramas=150,
                  reference_images=85, views_per_variant=300, job=os.environ['SLURM_JOB_ID']))
            write(OUT / 'status.json', dict(runtime, state='complete',
                  elapsed_seconds=time.monotonic() - started, variants=10))
        except BaseException:
            write(OUT / 'status.json', dict(runtime, state='failed', error=traceback.format_exc()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else evaluate()
