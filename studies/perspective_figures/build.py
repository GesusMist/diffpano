"""Compose twenty existing ERP/perspective comparisons with fixed report views."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from studies.followup24.common import PROMPTS20
from studies.panorama_metrics import render as projection
from studies.perspective24.common import LABELS

ROOT = Path('/home/shig/diffpano')
OUT = ROOT / 'outputs/10.16report_images/perspective'
REFERENCES = ROOT / 'outputs/10.15perspective24-flux/seed0-v1'
OURS = ROOT / 'outputs/10.7gradient-blending-flux/seed0-v1/cases'
SPHERE = ROOT / 'outputs/10.1flux-step-controls-seed0/spherediff/flux/steps20'
PLAN = ROOT / 'outputs/10.15layout24-flux216/seed0-v1/generation_plan.json'
VIEWS = ((0., 90.), (0., 10.), (0., 0.), (0., -10.), (0., -90.))
ROW_LABELS = ('Up', 'Slightly up', 'Eye level', 'Slightly down', 'Down')
ERP_SIZE = (4096, 2048)
PATCH_SIZE, RENDER_SIZE, ROW_GAP = 400, 1024, 48
MARGIN, COLUMN_GAP, BODY_TOP = 32, 40, 184
WIDTHS = (ERP_SIZE[0], PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, ERP_SIZE[0])
STACK_HEIGHT = 5 * PATCH_SIZE + 4 * ROW_GAP
SIZE = (2 * MARGIN + sum(WIDTHS) + 4 * COLUMN_GAP, BODY_TOP + STACK_HEIGHT + 32)
X = tuple(MARGIN + sum(WIDTHS[:i]) + i * COLUMN_GAP for i in range(5))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def checked_source(path, expected_hash, size, metadata_path, expected_metadata_hash=None):
    assert sha(path) == expected_hash, str(path)
    if expected_metadata_hash is not None:
        assert sha(metadata_path) == expected_metadata_hash, str(metadata_path)
    with Image.open(path) as image:
        assert image.format == 'PNG' and image.size == size and image.mode == 'RGB', str(path)
        image.verify()
    return dict(path=str(path), resolved_path=str(path.resolve()), sha256=expected_hash,
                size=list(size), metadata_path=str(metadata_path), metadata_sha256=sha(metadata_path))


def inventory():
    """Verify all 140 source images before exporting any figure."""
    plan = {row['case_id']: row for row in read(PLAN)['rows']}
    references = {row['case_id']: row for row in read(REFERENCES / 'manifest.json')['rows']}
    reference_index = {(row['scene'], row['direction_label']): row for row in read(REFERENCES / 'index.json')['rows']}
    result = []
    for name in PROMPTS20:
        prompt_path = ROOT / 'prompts' / (name + '.txt')
        prompt_hash = sha(prompt_path)
        source = plan['prompts/' + name + '/spherediff_official_20']['source']
        assert source['backend'] == 'flux' and source['layout'] == 'official'
        assert source['steps'] == 20 and source['seed'] == 0
        sphere_path = SPHERE / name / 'final.png'
        assert sphere_path.resolve() == Path(source['image_path']).resolve()
        sphere = checked_source(sphere_path, source['image_sha256'], ERP_SIZE,
                                Path(source['original_metadata_path']), source['original_metadata_sha256'])
        # The existing audited reuse record ties the official image to the prompt.
        assert source['prompt_record']['original_sha256'] == prompt_hash
        sphere.update(backend='flux', steps=20, seed=0, layout='official')

        ours_path = OURS / name / 'poisson_select' / 'final.png'
        ours_metadata_path = OURS / name / 'poisson_select' / 'metadata.json'
        metadata = read(ours_metadata_path)
        assert metadata['backend'] == 'flux'
        assert metadata['configuration']['generation']['num_inference_steps'] == 20
        assert metadata['configuration']['experiment']['seed'] == 0
        assert metadata['prompt']['original_sha256'] == prompt_hash
        assert metadata['fusion']['mode'] == metadata['terminal_fusion_mode'] == 'poisson_select'
        ours = checked_source(ours_path, metadata['artifacts']['final.png'], ERP_SIZE, ours_metadata_path)
        ours.update(backend='flux', steps=20, seed=0, fusion=metadata['fusion'],
                    camera_geometry_sha256=metadata['camera_geometry_sha256'])
        refs = []
        for label in LABELS:
            case_id = 'prompts/' + name + '/' + label
            row = references[case_id]
            assert row['original_prompt_sha256'] == prompt_hash
            folder = REFERENCES / 'cases' / case_id
            meta = read(folder / 'metadata.json')
            # Per-case status files were deleted by the October 16 cleanup;
            # the preserved final index records their verified image/metadata hashes.
            index = reference_index[name, label]
            assert index['valid'] and Path(index['image']).resolve() == (folder / 'final.png').resolve()
            for key in ('semantic_sha256', 'source_sha256', 'line_sha256'):
                assert meta[key] == row[key], (case_id, key)
            assert meta['counts'] == dict(denoiser=20, encode=0, decode=1, initialize=1)
            assert meta['guided_steps'] == meta['native_intervals'] == 20
            assert meta['initialization']['seed'] == 0 and meta['initialization']['fresh_generator_per_image']
            assert meta['artifacts']['final.png'] == index['image_sha256']
            assert sha(folder / 'config.json') == meta['artifacts']['config.json']
            reference = checked_source(folder / 'final.png', meta['artifacts']['final.png'],
                                       (1024, 1024), folder / 'metadata.json', index['metadata_sha256'])
            reference.update(case_id=case_id, direction_label=label, line_index=row['line_index'],
                             line_text=row['line_text'], line_sha256=row['line_sha256'])
            refs.append(reference)
        result.append(dict(prompt=name, prompt_path=str(prompt_path), prompt_sha256=prompt_hash,
                           spherediff=sphere, ours=ours, references=refs))
    assert len(result) == 20
    return result


def geometry_check():
    """Check the five report cameras against an analytic world-direction ERP."""
    h, w = 512, 1024
    lon = ((np.arange(w) + .5) / w - .5) * 2 * np.pi
    lat = (.5 - (np.arange(h) + .5) / h) * np.pi
    xx, yy = np.meshgrid(lon, lat)
    world = np.stack((np.sin(xx) * np.cos(yy), np.sin(yy), np.cos(xx) * np.cos(yy)), -1).astype(np.float32)
    checks = []
    for yaw, pitch in VIEWS:
        actual = projection.render(world, ((yaw, pitch),), size=65)[0]
        error = float(np.max(np.abs(actual - projection.rays(yaw, pitch, size=65))))
        assert error < .007, (yaw, pitch, error)
        checks.append(dict(yaw=yaw, pitch=pitch, max_world_direction_error=error))
    return dict(passed=True, checks=checks)


def font(size, bold=False):
    return ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans' + ('-Bold' if bold else '') + '.ttf', size)


def centered(draw, text, x, y, width, size=28, bold=False):
    face = font(size, bold)
    bounds = draw.textbbox((0, 0), text, font=face)
    assert bounds[2] - bounds[0] <= width, text
    draw.text((x + (width - bounds[2] + bounds[0]) // 2, y - bounds[1]), text, font=face, fill='black')


def load(path):
    with Image.open(path) as image:
        return image.convert('RGB')


def compose(record):
    canvas = Image.new('RGB', SIZE, 'white')
    draw = ImageDraw.Draw(canvas)
    centered(draw, record['prompt'].replace('_', ' ').capitalize(), 0, 20, SIZE[0], 44, True)
    titles = ('SphereDiff (original)', 'SphereDiff patches', 'Perspective references', 'Ours patches', 'Ours (poisson_select)')
    subtitles = ('Full ERP', 'Perspective ERP views', '10.15perspective24-flux', 'Perspective ERP views', 'Full ERP · 10.7gradient-blending-flux')
    for i, (title, subtitle) in enumerate(zip(titles, subtitles)):
        centered(draw, title, X[i], 88, WIDTHS[i], 28 if i in (1, 2, 3) else 36, True)
        centered(draw, subtitle, X[i], 128, WIDTHS[i], 21 if i in (1, 2, 3) else 26)
    sphere, ours = load(record['spherediff']['path']), load(record['ours']['path'])
    patches = (projection.render(np.asarray(sphere), VIEWS, size=RENDER_SIZE),
               projection.render(np.asarray(ours), VIEWS, size=RENDER_SIZE))
    panels = []
    expected = []

    def paste(image, x, y, kind, **extra):
        canvas.paste(image, (x, y))
        box = (x, y, x + image.width, y + image.height)
        panels.append(dict(kind=kind, box=list(box), **extra))
        expected.append((box, hashlib.sha256(image.tobytes()).hexdigest()))

    erp_y = BODY_TOP + (STACK_HEIGHT - ERP_SIZE[1]) // 2
    paste(sphere, X[0], erp_y, 'spherediff_erp')
    paste(ours, X[4], erp_y, 'ours_erp')
    for i, (label, view, reference) in enumerate(zip(ROW_LABELS, VIEWS, record['references'])):
        y = BODY_TOP + i * (PATCH_SIZE + ROW_GAP)
        for column in (1, 3):
            centered(draw, '%+g° pitch' % view[1], X[column], y - 28, PATCH_SIZE, 21)
        centered(draw, '%d. %s' % (i + 1, label), X[2], y - 28, PATCH_SIZE, 21)
        images = (Image.fromarray(patches[0][i]), load(reference['path']), Image.fromarray(patches[1][i]))
        for column, image, kind in zip((1, 2, 3), images, ('spherediff_patch', 'reference', 'ours_patch')):
            paste(image.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS),
                  X[column], y, kind, row=i + 1, direction_label=LABELS[i])
    path = OUT / (record['prompt'] + '.png')
    temporary = path.with_suffix('.tmp.png')
    canvas.save(temporary)
    # Verify all 17 image panels in the saved figure, including unchanged full ERPs.
    with Image.open(temporary) as exported:
        exported.load()
        assert exported.size == SIZE and exported.mode == 'RGB'
        for box, expected_hash in expected:
            assert hashlib.sha256(exported.crop(box).tobytes()).hexdigest() == expected_hash, (path, box)
    os.replace(temporary, path)
    return dict(record, path=str(path), sha256=sha(path), panels=panels, verified_panels=len(expected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', nargs='+', choices=PROMPTS20, help='Optional preview subset')
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Render inside a CPU allocation'
    OUT.mkdir(parents=True, exist_ok=True)
    records = inventory()
    geometry = geometry_check()
    print('SOURCE_AUDIT_PASSED 20 prompts / 140 images', flush=True)
    chosen = [row for row in records if not args.prompts or row['prompt'] in args.prompts]
    figures = []
    for record in chosen:
        figures.append(compose(record))
        print('FIGURE_COMPLETE', record['prompt'], flush=True)
    manifest = dict(complete=len(figures) == 20, expected_figure_count=20, figure_count=len(figures),
                    source_image_count=140, figure_size=list(SIZE), figures=figures,
                    column_order=['SphereDiff original ERP', 'SphereDiff ERP patches',
                                  'Standalone perspective references', 'Ours ERP patches', 'Ours poisson_select ERP'],
                    geometry_validation=geometry, views=[dict(yaw=yaw, pitch=pitch, roll=0) for yaw, pitch in VIEWS],
                    fov_x=90, fov_y=90, patch_render_size=RENDER_SIZE, patch_display_size=PATCH_SIZE,
                    erp_display='Full 4096x2048 original pixels, uncropped, no resizing or color changes',
                    reference_note='Independent prompt-line images have no defined camera or FOV. ERP extraction uses fixed report cameras at the five directional-conditioning pitch centers.',
                    patch_display='Original full-resolution ERPs; periodic bilinear perspective rendering, then Lanczos downsampling. Shared cameras for all prompts and both methods.',
                    yaw_selection='Fixed yaw 0 for every scene and both methods; no content-based selection',
                    builder_sha256=sha(__file__), renderer_path=str(Path(projection.__file__)),
                    renderer_sha256=sha(projection.__file__), job=os.environ['SLURM_JOB_ID'])
    write(OUT / 'provenance' / ('preview.json' if args.prompts else 'figures.json'), manifest)
    print('FIGURES_VERIFIED', len(figures), '17 panels each', flush=True)


if __name__ == '__main__':
    main()
