"""Separate, cleanup-aware reporting for the four resumed Uniform collections."""
import collections
from pathlib import Path

from studies.faster_uniform_resume.common import ROOT, OUT, PLAN, load_plan
from studies.faster_uniform_resume.completion import load_verifier
from studies.faster_uniform_resume.evaluate import UNIFORM_VARIANTS, resume_metrics
from studies.layout24 import common as original
from studies.followup24.common import PROMPTS24

LABELS = {
    'diffpano_uniform89_rgb': 'Uniform89 RGB (ours)',
    'diffpano_uniform70_rgb': 'Uniform70 RGB (ours)',
    'diffpano_uniform50_rgb': 'Uniform50 RGB (ours)',
    'diffpano_uniform89_screened_select': 'Uniform89 lambda_blending + select (ours)',
}


def generation_summary(manifest, verifier):
    uniform = [row for row in manifest['rows'] if row['variant'] in UNIFORM_VARIANTS]
    states = collections.Counter()
    variants = {name: collections.Counter() for name in UNIFORM_VARIANTS}
    for row in uniform:
        if verifier.protected_complete(row):
            state = 'protected_complete'
        elif verifier.checked_complete(row):
            state = 'resumed_complete'
        else:
            status_path = original.folder(row) / 'status.json'
            status = original.read(status_path) if status_path.exists() else {}
            state = status.get('state', 'missing')
            if state == 'complete':
                state = 'invalid'
        states[state] += 1
        variants[row['variant']][state] += 1
    return dict(counts=dict(states), per_variant={k: dict(v) for k, v in variants.items()},
                completed=sum(states[k] for k in ('protected_complete', 'resumed_complete')),
                expected=96, expected_resumed=75, expected_protected=21)


def figures(manifest, verifier):
    from PIL import Image, ImageDraw, ImageFont
    import os
    target = OUT / 'figures'
    target.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 24)
    previous_path = target / 'provenance.json'
    previous = original.read(previous_path) if previous_path.exists() else {}
    old = {row['prompt']: row for row in previous.get('figures', [])}
    saved, pending, provenance = [], [], []
    for name in PROMPTS24:
        group = [next(row for row in manifest['rows']
                      if row['prompt'] == name and row['variant'] == variant)
                 for variant in UNIFORM_VARIANTS]
        if not all(verifier.checked_complete(row) for row in group):
            pending.append(name)
            continue
        sources = [dict(variant=row['variant'], path=str(original.folder(row) / 'final.png'),
                        sha256=original.sha(original.folder(row) / 'final.png'),
                        metadata_sha256=original.sha(original.folder(row) / 'metadata.json'))
                   for row in group]
        path = target / (name + '.png')
        prior = old.get(name, {})
        if (prior.get('sources') == sources and path.exists()
                and prior.get('sha256') == original.sha(path)):
            saved.append(name)
            provenance.append(prior)
            continue
        canvas = Image.new('RGB', (2080, 1144), 'white')
        draw = ImageDraw.Draw(canvas)
        for index, (row, source) in enumerate(zip(group, sources)):
            x, y = (index % 2) * 1040 + 8, (index // 2) * 572 + 8
            draw.text((x, y), LABELS[row['variant']], fill='black', font=font)
            with Image.open(source['path']) as image:
                assert image.size == (4096, 2048)
                canvas.paste(image.convert('RGB').resize(
                    (1024, 512), Image.Resampling.LANCZOS), (x, y + 44))
        temporary = path.with_suffix('.tmp.png')
        canvas.save(temporary)
        os.replace(str(temporary), str(path))
        saved.append(name)
        provenance.append(dict(prompt=name, path=str(path), sha256=original.sha(path),
                               sources=sources))
    original.write(previous_path, dict(
        figures=provenance, complete_prompts=saved, pending_prompts=pending,
        order=list(UNIFORM_VARIANTS),
        display='Original final PNG; identical Lanczos display resize only.'))
    return saved, pending


def main():
    from studies.layout24.report import atomic_text, fmt, latex
    load_plan()
    verifier = load_verifier(strict_complete=original.complete)
    manifest = original.freeze_check()
    generation = generation_summary(manifest, verifier)
    metric_path = OUT / 'evaluation/metrics.json'
    metrics = original.read(metric_path) if metric_path.exists() else None
    groups = metrics['groups'] if metrics else []
    metric_summary = resume_metrics(metrics) if metrics else None
    if metrics:
        atomic_text(OUT / 'results.tex', latex(groups))
    saved, pending = figures(manifest, verifier)
    lines = [
        'FASTER Uniform-camera continuation', '',
        'Scope: 75 missing FLUX panoramas, completing four Uniform collections over '
        'the same 24 prompts. The 21 existing Uniform panoramas and 80 other completed '
        'panoramas remain protected. Forty missing Old70/Old50 cases are outside this resume.',
        '',
        'Uniform generation: %d / 96 complete; %d / 75 newly completed in this continuation.'
        % (generation['completed'], generation['counts'].get('resumed_complete', 0)),
        'Generation states: ' + str(generation['counts']) + '.',
        'Uniform metrics: ' + str(metric_summary['uniform_counts'] if metrics else 'pending') + '.',
        'All-matrix metrics: ' + str(metrics['counts'] if metrics else 'pending') + '.',
        'Uniform four-panel figures: %d / 24 complete.' % len(saved), '',
        'All cases retain seed 0, 20 denoising intervals, 80-degree perspective FOV, '
        'and 4096 × 2048 ERP. The existing metric protocol is unchanged. IS uses 192 '
        'horizontal views per complete 24-prompt variant; original20 and new4 subsets '
        'are computed independently. Incomplete collections have no aggregate score.', '',
        '| Variant | N | Scored / 24 | DS ↓ | CS ↑ | IS ↑ | Generation minutes | Timed cases |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    from studies.layout24.report import LABELS as ALL_LABELS
    for group in groups:
        seconds = group['mean_generation_seconds']
        lines.append('| %s | %d | %d | %s | %s | %s | %s | %d |' % (
            ALL_LABELS[group['variant']], group['camera_count'], group['scored_panorama_count'],
            fmt(group['mean_DS'], 4), fmt(group['mean_CS']), fmt(group['IS']),
            fmt(None if seconds is None else seconds / 60, 2), group['timed_panorama_count']))
    lines += ['', 'Generation times may mix original Grace measurements and new FASTER '
              'measurements. Hardware and cluster provenance remain in each result; '
              'do not interpret these mixed times as a controlled cluster comparison.',
              '', 'Historical images, metadata, metric caches, and evaluation reports were '
              'read without modification. Cleanup-aware completion verifies retained '
              'image and metadata hashes instead of requiring deliberately deleted tensors.',
              '', 'Full 216-case completion is separate from this 96-case Uniform scope.']
    atomic_text(OUT / 'report.md', '\n'.join(lines) + '\n')
    original.write(OUT / 'status-summary.json', dict(
        generation=generation, metrics=metric_summary, figures=len(saved),
        pending_figures=pending, plan_sha256=original.sha(PLAN),
        uniform_complete=bool(metric_summary and metric_summary['evaluation_complete']
                              and generation['completed'] == 96 and len(saved) == 24)))
    print('UNIFORM_REPORT', generation, 'figures', len(saved),
          'metrics', metric_summary, flush=True)


if __name__ == '__main__':
    main()
