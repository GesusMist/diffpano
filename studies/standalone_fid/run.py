"""Extract compact Inception features and report seven standalone-reference FIDs."""
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import inspect
import os
import socket
import time
import traceback

from studies.standalone_fid.common import *


def atomic_text(path, value):
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(value)
    os.replace(str(temporary), str(path))


def feature_cache(key, expected, produce):
    import numpy as np
    path = OUT / 'cache' / (key + '.npz')
    meta_path = path.with_suffix('.json')
    if path.exists() and meta_path.exists():
        metadata = read(meta_path)
        if metadata.get('key') == key and metadata.get('sha256') == sha(path):
            with np.load(path, allow_pickle=False) as archive:
                values = archive['features']
            if values.shape == (expected, 2048) and np.isfinite(values).all():
                return values, dict(path=str(path), sha256=metadata['sha256'], reused=True)
    values = np.asarray(produce(), dtype=np.float32)
    assert values.shape == (expected, 2048) and np.isfinite(values).all()
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, features=values)
    os.replace(str(temporary), str(path))
    metadata = dict(key=key, sha256=sha(path), shape=list(values.shape),
                    job=os.environ['SLURM_JOB_ID'])
    write(meta_path, metadata)
    return values, dict(path=str(path), sha256=metadata['sha256'], reused=False)


def report(results):
    import csv
    import io
    rows = results['rows']
    stream = io.StringIO()
    fields = ['variant', 'label', 'FID', 'reference_images', 'panoramas', 'generated_views']
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(OUT / 'results.csv', stream.getvalue())
    view_count = rows[0]['generated_views']
    caption = ('FLUX panorama views compared with 85 unique standalone FLUX reference images. '
               'Each method contributes %d fixed 80-degree perspective views from the same '
               '24 scenes. Lower FID is better. The synthetic reference set is small; '
               'scores are exploratory.' % view_count)
    lines = [r'\begin{table}[t]', r'\centering', r'\caption{' + caption + '}',
             r'\label{tab:flux_standalone_fid85}', r'\resizebox{\columnwidth}{!}{',
             r'\begin{tabular}{lc}', r'\toprule',
             r'Method & FID $\downarrow$ \\', r'\midrule']
    ranking = sorted(rows, key=lambda row: row['FID'])
    for row in rows:
        value = '%.3f' % row['FID']
        if row['variant'] == ranking[0]['variant']:
            value = r'\textbf{' + value + '}'
        elif row['variant'] == ranking[1]['variant']:
            value = r'\underline{' + value + '}'
        lines.append(row['label'] + ' & ' + value + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '}', r'\end{table}', '']
    atomic_text(OUT / 'results.tex', '\n'.join(lines))
    text = [caption, '', 'Reference images are deduplicated by exact PNG SHA256. '
            'All aliases remain recorded in reference-manifest.json. Each unique reference '
            'counts once; no reference duplication, augmentation, or generation was used.',
            '', '| Method | FID ↓ |', '|---|---:|']
    text += ['| %s | %.3f |' % (row['label'], row['FID']) for row in rows]
    text += ['', 'Evaluation settings: Inception-v3-compatible 2048-dimensional features; '
             'float32 inference and float64 statistics; both collections enter identical '
             'preprocessing as 1024x1024 RGB uint8. Panorama pitches are +90,+10,0,-10,-90 '
             'degrees, using the yaws recorded in protocol.json. All seven methods use '
             'the same evaluation cameras, independent of their generation camera layout.',
             '', 'Limitations: deduplication changes the reference scene/direction weights; '
             'the generated collection remains balanced over the 24 scenes and five directions. '
             '85 unique references and 24 panoramas per method are small samples. Multiple '
             'views are correlated, and pole yaws rotate the image plane. Standalone '
             'images have no enforced optical FOV. Scores measure similarity to standalone '
             'FLUX generation, not real-image fidelity or global panorama consistency.',
             '', 'Sources: https://arxiv.org/abs/1706.08500 ; '
             'https://arxiv.org/abs/1911.07023 ; https://github.com/toshas/torch-fidelity', '']
    atomic_text(OUT / 'report.txt', '\n'.join(text))


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run feature extraction in a Slurm GPU job'
    plan = load_plan()
    with (OUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completion = OUT / 'completion.json'
        if completion.exists():
            done = read(completion)
            if done.get('passed') and done.get('plan_sha256') == sha(PLAN):
                assert done['results_sha256'] == sha(OUT / 'results.json')
                print('VALID_FID_CACHE', flush=True)
                return
        started = time.monotonic()
        provenance = dict(job=os.environ['SLURM_JOB_ID'], hostname=socket.gethostname(),
                          cluster=os.environ.get('SLURM_CLUSTER_NAME'),
                          plan_sha256=sha(PLAN), started_utc=datetime.now(timezone.utc).isoformat())
        write(OUT / 'status.json', dict(provenance, state='running', stage='imports'))
        try:
            import numpy as np
            import torch
            import cv2
            from PIL import Image
            from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
            from studies.standalone_fid.render import render
            from studies.panorama_metrics.numerics import frechet
            assert torch.cuda.is_available()
            torch.set_num_threads(4)
            cv2.setNumThreads(1)
            torch.backends.cudnn.benchmark = False
            protocol = read(OUT / 'protocol.json')
            references = read(OUT / 'reference-manifest.json')['rows']
            panoramas = read(OUT / 'panorama-manifest.json')['rows']
            checkpoint = protocol['features']['checkpoint']
            assert sha(checkpoint['path']) == checkpoint['sha256']
            extractor = FeatureExtractorInceptionV3(
                'inception-v3-compat', ['2048'],
                feature_extractor_weights_path=checkpoint['path']).cuda().eval()
            provenance['software'] = dict(torch=torch.__version__, numpy=np.__version__,
                opencv=cv2.__version__, torch_fidelity=importlib.metadata.version('torch-fidelity'))
            provenance['extractor_source_sha256'] = sha(inspect.getfile(FeatureExtractorInceptionV3))
            provenance['gpu'] = torch.cuda.get_device_name(0)
            extraction_identity = digest(dict(protocol=plan['protocol_sha256'],
                software=provenance['software'], extractor_source=provenance['extractor_source_sha256'],
                renderer_source=plan['source_hashes']['studies/standalone_fid/render.py']))
            provenance['extraction_identity'] = extraction_identity
            batch_size = 8

            def extract(images):
                features = []
                for start in range(0, len(images), batch_size):
                    batch = np.stack(images[start:start + batch_size])
                    assert batch.dtype == np.uint8 and batch.shape[1:] == (1024, 1024, 3)
                    tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2).copy()).cuda()
                    with torch.inference_mode():
                        output, = extractor(tensor)
                    features.append(output.cpu().numpy())
                return np.concatenate(features)

            for row in references:
                assert sha(row['image_path']) == row['image_sha256']
            assert len(references) == len({row['image_sha256'] for row in references}) == 85

            def reference_features():
                output = []
                for start in range(0, len(references), batch_size):
                    images = []
                    for row in references[start:start + batch_size]:
                        with Image.open(row['image_path']) as image:
                            assert image.size == (1024, 1024) and image.mode == 'RGB'
                            images.append(np.asarray(image))
                    output.append(extract(images))
                return np.concatenate(output)

            reference_key = digest(dict(kind='reference85', extractor=extraction_identity,
                                        images=[row['image_sha256'] for row in references]))
            reference, reference_cache = feature_cache(reference_key, 85, reference_features)
            assert abs(frechet(reference, reference)) < 1e-5, 'FID self-distance validation failed'
            print('FID_REFERENCE_READY', reference.shape, flush=True)
            by_variant = {variant: [] for variant in VARIANTS}
            records = []
            views = protocol['generated']['views']
            for index, row in enumerate(panoramas):
                assert sha(row['image_path']) == row['image_sha256'], row['case_id']
                assert sha(row['metadata_path']) == row['metadata_sha256']
                key = digest(dict(kind='panorama_views', extractor=extraction_identity,
                                  image=row['image_sha256']))

                def produce():
                    with Image.open(row['image_path']) as image:
                        assert image.size == (4096, 2048)
                        rgb = np.asarray(image.convert('RGB'))
                    return extract([render(rgb, view['yaw'], view['pitch'], size=1024, fov=80.)
                                    for view in views])

                values, cache = feature_cache(key, len(views), produce)
                by_variant[row['variant']].append(values)
                records.append(dict(row, feature_cache=cache))
                write(OUT / 'status.json', dict(provenance, state='running', stage='panorama_features',
                      references=85, completed_panoramas=index + 1, expected_panoramas=168,
                      elapsed_seconds=time.monotonic() - started))
                print('FID_FEATURES', index + 1, '/', len(panoramas), row['case_id'], flush=True)
            results = []
            for variant in VARIANTS:
                assert len(by_variant[variant]) == 24
                generated = np.concatenate(by_variant[variant])
                assert generated.shape == (plan['expected']['views_per_variant'], 2048)
                value = frechet(reference, generated)
                results.append(dict(variant=variant, label=LABELS[variant], FID=value,
                                    reference_images=85, panoramas=24, generated_views=len(generated)))
                print('FID_RESULT', variant, value, flush=True)
            load_plan()
            payload = dict(complete=True, metric=protocol['name'], protocol_sha256=plan['protocol_sha256'],
                           reference_feature_cache=reference_cache, panorama_feature_records=records,
                           rows=results, runtime=dict(provenance, elapsed_seconds=time.monotonic() - started),
                           limitations=protocol['limitations'])
            write(OUT / 'results.json', payload)
            report(payload)
            write(completion, dict(passed=True, plan_sha256=sha(PLAN), results_sha256=sha(OUT / 'results.json'),
                                   variants=7, reference_images=85, panoramas=168,
                                   generated_views=plan['expected']['generated_views'],
                                   job=os.environ['SLURM_JOB_ID']))
            write(OUT / 'status.json', dict(provenance, state='complete',
                  elapsed_seconds=time.monotonic() - started, variants=7, reference_images=85))
        except BaseException:
            write(OUT / 'status.json', dict(provenance, state='failed', error=traceback.format_exc(),
                                          elapsed_seconds=time.monotonic() - started))
            raise


if __name__ == '__main__':
    main()
