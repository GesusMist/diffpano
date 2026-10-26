"""Select five cached yaw-zero views and recompute full collection FID on CPU."""
import copy
import csv
from datetime import datetime, timezone
import fcntl
import io
import os
from pathlib import Path
import time

from studies.standalone_fid import common as original
from studies.standalone_fid.common import ROOT, VARIANTS, LABELS, sha, digest, read, write

OUT = ROOT / 'outputs/10.23fid-standalone85/five-views-yaw0'


def atomic_text(path, value):
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(value)
    os.replace(str(temporary), str(path))


def main():
    import numpy as np
    from studies.panorama_metrics.numerics import frechet
    from studies.followup24.common import PROMPTS24
    started = time.monotonic()
    original_plan = original.load_plan()
    old_results = read(original.OUT / 'results.json')
    old_completion = read(original.OUT / 'completion.json')
    old_protocol = read(original.OUT / 'protocol.json')
    assert old_completion['passed'] and old_results['complete']
    assert old_completion['results_sha256'] == sha(original.OUT / 'results.json')
    assert old_completion['plan_sha256'] == sha(original.PLAN)
    assert old_results['protocol_sha256'] == digest(old_protocol) == original_plan['protocol_sha256']
    original_views = old_protocol['generated']['views']
    indices = [index for index, view in enumerate(original_views) if view['yaw'] == 0.]
    views = [original_views[index] for index in indices]
    assert indices == [0, 4, 8, 12, 16]
    assert [view['pitch'] for view in views] == [90., 10., 0., -10., -90.]
    assert old_protocol['generated']['fov_x'] == old_protocol['generated']['fov_y'] == 80.
    reference_manifest = read(original.OUT / 'reference-manifest.json')
    assert reference_manifest['unique_count'] == 85
    assert len({row['image_sha256'] for row in reference_manifest['rows']}) == 85
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = copy.deepcopy(old_protocol)
        protocol['name'] = 'FID against 85 unique standalone FLUX images, five views at yaw0'
        protocol['generated'].update(views=views, views_per_panorama=5, views_per_variant=120,
                                     fixed_yaw=0., poles='One view per pole, using the fixed yaw0 frame')
        protocol['limitations'] = [text for text in protocol['limitations']
                                   if 'pole yaws' not in text]
        protocol['limitations'].append('The five views from each panorama are correlated; yaw0 is fixed for every scene and method.')
        protocol['feature_reuse'] = dict(
            source_protocol_sha256=digest(old_protocol),
            source_results_path=str(original.OUT / 'results.json'),
            source_results_sha256=sha(original.OUT / 'results.json'),
            selected_view_indices=indices,
            reason='Reuse the exact already-extracted yaw0 feature vectors; recompute FID from the selected collections',
            new_model_inference=False)

        def features(cache, count):
            path = Path(cache['path'])
            assert sha(path) == cache['sha256'], path
            metadata = read(path.with_suffix('.json'))
            assert metadata['sha256'] == cache['sha256'] and metadata['key'] == path.stem
            with np.load(path, allow_pickle=False) as archive:
                value = archive['features']
            assert value.shape == (count, 2048) and value.dtype == np.float32
            assert np.isfinite(value).all()
            return value

        reference_cache = old_results['reference_feature_cache']
        reference = features(reference_cache, 85)
        assert abs(frechet(reference, reference)) < 1e-5
        records = old_results['panorama_feature_records']
        assert len(records) == 168 and len({row['case_id'] for row in records}) == 168
        results, provenance = [], []
        for variant in VARIANTS:
            group = [row for row in records if row['variant'] == variant]
            assert len(group) == 24 and {row['prompt'] for row in group} == set(PROMPTS24)
            selected = []
            for row in group:
                value = features(row['feature_cache'], 20)
                selected.append(value[indices])
                provenance.append(dict(row, selected_view_indices=indices,
                                       selected_view_count=5))
            matrix = np.concatenate(selected)
            assert matrix.shape == (120, 2048)
            score = frechet(reference, matrix)
            results.append(dict(variant=variant, label=LABELS[variant], FID=score,
                                reference_images=85, panoramas=24, generated_views=120))
            print('FIVE_VIEW_FID', variant, score, flush=True)
        original.load_plan()
        assert sha(original.OUT / 'results.json') == old_completion['results_sha256']
        write(OUT / 'protocol.json', protocol)
        write(OUT / 'reference-manifest.json', reference_manifest)
        payload = dict(complete=True, metric=protocol['name'], protocol_sha256=digest(protocol),
                       rows=results, reference_feature_cache=reference_cache,
                       panorama_feature_records=provenance,
                       source_results_sha256=old_completion['results_sha256'],
                       feature_extraction_runtime=old_results['runtime'],
                       computation=dict(device='CPU', new_model_inference=False,
                           created_utc=datetime.now(timezone.utc).isoformat(),
                           elapsed_seconds=time.monotonic() - started,
                           source_hashes={str(Path(__file__).relative_to(ROOT)): sha(__file__),
                             'studies/panorama_metrics/numerics.py': sha(ROOT / 'studies/panorama_metrics/numerics.py')}),
                       limitations=protocol['limitations'])
        write(OUT / 'results.json', payload)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=('variant', 'label', 'FID', 'reference_images', 'panoramas', 'generated_views'))
        writer.writeheader()
        writer.writerows(results)
        atomic_text(OUT / 'results.csv', stream.getvalue())
        caption = ('FLUX panorama views compared with 85 unique standalone FLUX reference images. '
                   'Each method contributes 120 views from24 scenes: five views per panorama '
                   'at fixed yaw0 degrees and pitches +90,+10,0,-10,-90 degrees, with80-degree '
                   'horizontal and vertical FOV. Lower FID is better. Scores are exploratory.')
        caption = caption.replace('from24', 'from 24').replace('yaw0', 'yaw 0').replace('with80', 'with 80')
        latex = [r'\begin{table}[t]', r'\centering', r'\caption{' + caption + '}',
                 r'\label{tab:flux_standalone_fid85_five_views}', r'\resizebox{\columnwidth}{!}{',
                 r'\begin{tabular}{lc}', r'\toprule', r'Method & FID $\downarrow$ \\', r'\midrule']
        ranking = sorted(results, key=lambda row: row['FID'])
        for row in results:
            value = '%.3f' % row['FID']
            if row['variant'] == ranking[0]['variant']:
                value = r'\textbf{' + value + '}'
            elif row['variant'] == ranking[1]['variant']:
                value = r'\underline{' + value + '}'
            latex.append(row['label'] + ' & ' + value + r' \\')
        latex += [r'\bottomrule', r'\end{tabular}', '}', r'\end{table}', '']
        atomic_text(OUT / 'results.tex', '\n'.join(latex))
        text = [caption, '', '| Method | FID ↓ |', '|---|---:|']
        text += ['| %s | %.3f |' % (row['label'], row['FID']) for row in results]
        text += ['', 'Selected cached indices:0,4,8,12,16. All169 feature-file hashes were verified '
                 '(85-reference feature matrix plus168 panorama feature matrices). FID was '
                 'recomputed from full2048-dimensional vectors using float64 statistics. '
                 'The existing20-view scores and all generation files remain unchanged.',
                 '', 'Limitations:', *protocol['limitations'], '']
        atomic_text(OUT / 'report.txt', '\n'.join(text))
        write(OUT / 'completion.json', dict(passed=True, reference_images=85, panoramas=168,
              variants=7, views_per_panorama=5, views_per_variant=120, generated_views=840,
              fixed_yaw=0., protocol_sha256=digest(protocol), results_sha256=sha(OUT / 'results.json')))
        print('FIVE_VIEW_COMPLETE', OUT, flush=True)


if __name__ == '__main__':
    main()
