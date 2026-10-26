"""Compose existing ERP and pole views; no generation or image correction."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/10.23report'
PAD, GAP, HEADER, LABEL, ROW_GAP, FOOT = 32, 32, 140, 62, 34, 42


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def write(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + '\n')


def face(size, bold=False):
    name = 'DejaVuSans' + ('-Bold' if bold else '') + '.ttf'
    for directory in ('/usr/share/fonts/dejavu', '/usr/share/fonts/truetype/dejavu'):
        path = Path(directory) / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise FileNotFoundError('DejaVuSans font is required for report labels')


def center(draw, text, x, y, width, size=34, bold=False):
    font = face(size, bold)
    box = draw.textbbox((0, 0), text, font=font)
    if box[2] - box[0] > width:
        raise ValueError('Label is wider than its panel: ' + text)
    draw.text((x + (width - box[2] + box[0]) // 2, y - box[1]),
              text, fill='#171717', font=font)


def row_groups(figure):
    panels = {panel['key']: panel for panel in figure['panels']}
    assert len(panels) == len(figure['panels']), 'Duplicate panel identity'
    if figure['kind'] == 'uniform':
        keys = [
            ['spherediff_official_20'],
            ['diffpano_old89_rgb', 'diffpano_old70_rgb', 'diffpano_old50_rgb'],
            ['diffpano_uniform89_rgb', 'diffpano_uniform70_rgb', 'diffpano_uniform50_rgb'],
            ['diffpano_old89_screened_select', 'diffpano_uniform89_screened_select'],
        ]
        groups = [[panels[key] for key in row if key in panels] for row in keys]
        assert [len(row) for row in groups] in ([1, 3, 3, 2], [1, 1, 3, 2])
        assert sum(map(len, groups)) == len(panels)
        return groups, 3
    if figure['kind'] == 'refinement_percent':
        assert len(panels) == 3
        assert [panel['display_percent'] for panel in figure['panels']] == [0, 20, 30]
        return [figure['panels']], 3
    assert figure['kind'] in ('refinement28', 'refinement20')
    assert len(panels) == 2
    return [figure['panels']], 2


def checked_image(source):
    data = Path(source['image_path']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == source['image_sha256'], 'Source image changed'
    assert sha(source['metadata_path']) == source['metadata_sha256'], 'Source metadata changed'
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == 'PNG' and image.mode == 'RGB' and image.size == (4096, 2048)
        image.load()
        return image.copy()


def compose(figure, destination):
    from studies.mean_max_figures.pole_panels import PANEL_SIZE, make_panel
    groups, columns = row_groups(figure)
    pw, ph = PANEL_SIZE
    width = 2 * PAD + columns * pw + (columns - 1) * GAP
    height = HEADER + len(groups) * (LABEL + ph) + (len(groups) - 1) * ROW_GAP + FOOT
    name = figure['prompt']
    assert Path(name).name == name and name not in ('.', '..')
    output = destination / (name + '.png')
    assert not output.exists(), 'Preserve existing report figures: ' + str(output)
    if figure.get('prompt_path'):
        assert sha(figure['prompt_path']) == figure['prompt_sha256']
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    center(draw, figure['title'], 0, 25, width, 48, True)
    center(draw, figure['subtitle'], 0, 88, width, 30)
    panels = []
    for row_index, group in enumerate(groups):
        group_width = len(group) * pw + (len(group) - 1) * GAP
        first_x = (width - group_width) // 2
        y = HEADER + row_index * (LABEL + ph + ROW_GAP)
        for column, source in enumerate(group):
            x = first_x + column * (pw + GAP)
            center(draw, source['label'], x, y + 8, pw, 34, True)
            image = checked_image(source)
            tile = make_panel(image)
            image.close()
            canvas.paste(tile, (x, y + LABEL))
            panels.append(dict(source, row=row_index + 1, column_in_row=column + 1,
                               box=[x, y + LABEL, x + pw, y + LABEL + ph],
                               rendered_pixels_sha256=hashlib.sha256(tile.tobytes()).hexdigest()))
            tile.close()
    temporary = output.with_suffix('.tmp.png')
    canvas.save(temporary, format='PNG', compress_level=4, dpi=(300, 300))
    canvas.close()
    with Image.open(temporary) as exported:
        exported.load()
        assert exported.size == (width, height) and exported.mode == 'RGB'
        for panel in panels:
            crop = exported.crop(panel['box'])
            assert hashlib.sha256(crop.tobytes()).hexdigest() == panel['rendered_pixels_sha256']
            crop.close()
            assert sha(panel['image_path']) == panel['image_sha256']
            assert sha(panel['metadata_path']) == panel['metadata_sha256']
    temporary.replace(output)
    result = dict(figure, path=str(output), sha256=sha(output), figure_size=[width, height],
                  panel_count=len(panels), panels=panels, all_exported_panels_pixel_verified=True,
                  all_source_images_and_metadata_unchanged=True)
    write(destination / 'provenance' / (name + '.json'), result)
    print('FIGURE_VERIFIED', figure['kind'], name, len(panels), flush=True)
    return result


def group_report(kind, figures, destination, evidence, geometry, workers):
    from studies.mean_max_figures.pole_panels import PROTOCOL, RENDERER
    destination.mkdir(parents=True, exist_ok=True)
    provenance = destination / 'provenance'
    provenance.mkdir(exist_ok=True)
    if list(destination.glob('*.png')) or (destination / 'manifest.json').exists():
        raise FileExistsError('Output already contains figures: ' + str(destination))
    assert len({figure['prompt'] for figure in figures}) == len(figures)
    inventory = dict(kind=kind, figures=figures, evidence=evidence)
    write(provenance / 'sources.json', inventory)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rendered = list(pool.map(lambda figure: compose(figure, destination), figures))
    missing = {figure['prompt']: figure['missing'] for figure in figures if figure.get('missing')}
    manifest = dict(complete=True, kind=kind, figure_count=len(rendered),
                    source_panel_count=sum(f['panel_count'] for f in rendered),
                    figures=rendered, omitted_controls=missing,
                    pole_views=PROTOCOL, geometry_validation=geometry,
                    source_inventory_sha256=sha(provenance / 'sources.json'),
                    builder_sha256=sha(__file__), pole_renderer_sha256=sha(RENDERER),
                    pole_panel_builder_sha256=sha(ROOT / 'studies/mean_max_figures/pole_panels.py'),
                    all_exported_panels_pixel_verified=True, all_source_hashes_preserved=True,
                    no_generation_performed=True, job=os.environ.get('SLURM_JOB_ID'))
    write(destination / 'manifest.json', manifest)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['prompt', 'figure', 'panels', 'omitted_controls'])
    writer.writeheader()
    for figure in rendered:
        writer.writerow(dict(prompt=figure['prompt'], figure=Path(figure['path']).name,
                             panels=figure['panel_count'],
                             omitted_controls='; '.join(p['key'] for p in figure.get('missing', []))))
    atomic_text(destination / 'index.csv', stream.getvalue())
    notes = [f'{len(rendered)} verified {kind} comparison figures.', '',
             'Each panel: original ERP displayed at 1536x768, plus fixed north/south pole views.',
             'Pole projection: yaw 0, pitch +/-90 degrees, FOV 90; projected from the original 4096x2048 PNG.',
             'Identical Lanczos display downsampling; no exposure, contrast, or color correction.',
             'Source images and metadata are unchanged. Saved panel pixels are verified after export.',
             'Exact source paths, hashes, settings, and omitted controls are in manifest.json and provenance/.']
    if kind == 'uniform':
        notes += ['', 'Rows: SphereDiff; available Old89/70/50 RGB; Uniform89/70/50 RGB;',
                  'Old89 and Uniform89 screened poisson_select (lambda_blending + select_closest).',
                  'Missing Old70/Old50 controls are omitted, as requested; remaining panels are centered.',
                  'All results use FLUX, 20 steps, seed 0. SphereDiff retains its original camera/noise protocol.']
    else:
        notes += ['', 'Screened poisson_select, lambda_color = 0.1, Old89 cameras, seed 0.',
                  'The refinement fraction counts independent steps within the stated total step count.']
        if kind == 'refinement_percent':
            notes += ['Left to right: 0% (no refinement), 20%, 30%; step counts are omitted from figure labels as requested.',
                      '20% and 30% are the requested display labels for 5 and 8 refinement steps out of 28.',
                      'The 0% panel is the existing 20-step baseline; total steps differ from the refined results.',
                      'Exact generation parameters are retained in provenance; no new inference was performed.']
        elif kind == 'refinement28':
            notes += ['Left: 5/28, 23 coupled + 5 independent. Right: 8/28, 20 coupled + 8 independent.',
                      'Four historical 20-step off/2-step comparisons are separately saved under history20/.']
        else:
            notes += ['Historical 20-step experiment only: off versus 2/20 independent refinement.',
                      'Retained image/metadata provenance is verified; old status files were intentionally absent.',
                      'These are not a 28-step no-refinement baseline.']
    atomic_text(destination / 'README.txt', '\n'.join(notes) + '\n')
    for figure in rendered:
        if figure['prompt'] not in ('air_balloons', 'aurora'):
            continue
        with Image.open(figure['path']) as image:
            image.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
            image.save(provenance / (figure['prompt'] + '-preview.png'))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Render report figures in a CPU Slurm allocation'
    assert 1 <= args.workers <= 4
    import cv2
    cv2.setNumThreads(1)
    from studies.mean_max_figures.pole_panels import validate_geometry
    from .uniform import collect as collect_uniform
    from .refinement import collect as collect_refinement
    geometry = validate_geometry()
    print('COLLECT_UNIFORM', flush=True)
    uniform = collect_uniform()
    print('COLLECT_REFINEMENT', flush=True)
    refinement = collect_refinement()
    groups = [('uniform', uniform['figures'], OUT / 'uniform', uniform['evidence']),
              ('refinement28', [f for f in refinement['figures'] if f['kind'] == 'refinement28'],
               OUT / 'refinement', refinement['evidence']),
              ('refinement20', [f for f in refinement['figures'] if f['kind'] == 'refinement20'],
               OUT / 'refinement/history20', refinement['evidence'])]
    assert [len(group[1]) for group in groups] == [24, 24, 4]
    results = [group_report(*group, geometry, args.workers) for group in groups]
    print('REPORT_COMPLETE', [(r['kind'], r['figure_count'], r['source_panel_count']) for r in results], flush=True)


if __name__ == '__main__':
    main()
