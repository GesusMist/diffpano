"""Compose audited FLUX 20/28-step pairs without resampling source pixels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ERP_SIZE = (4096, 2048)
MARGIN, GAP, TOP, BOTTOM = 48, 48, 224, 80
SIZE = (2 * MARGIN + 2 * ERP_SIZE[0] + GAP, TOP + ERP_SIZE[1] + BOTTOM)
BOXES = ((MARGIN, TOP, MARGIN + ERP_SIZE[0], TOP + ERP_SIZE[1]),
         (MARGIN + ERP_SIZE[0] + GAP, TOP, MARGIN + 2 * ERP_SIZE[0] + GAP, TOP + ERP_SIZE[1]))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def font(size, bold=False):
    name = 'DejaVuSans' + ('-Bold' if bold else '') + '.ttf'
    for directory in ('/usr/share/fonts/dejavu', '/usr/share/fonts/truetype/dejavu'):
        path = Path(directory) / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def centered(draw, text, x, y, width, size, bold=False):
    face = font(size, bold)
    bounds = draw.textbbox((0, 0), text, font=face)
    if bounds[2] - bounds[0] > width:
        raise ValueError('Figure label exceeds panel width: ' + text)
    draw.text((x + (width - bounds[2] + bounds[0]) // 2, y - bounds[1]),
              text, fill='#161616', font=face)


def checked_image(source):
    path = Path(source['image_path'])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != source['image_sha256']:
        raise ValueError('Source image hash changed: ' + str(path))
    if sha(source['metadata_path']) != source['metadata_sha256']:
        raise ValueError('Source metadata hash changed: ' + source['metadata_path'])
    with Image.open(io.BytesIO(data)) as image:
        if image.format != 'PNG' or image.mode != 'RGB' or image.size != ERP_SIZE:
            raise ValueError('Expected native RGB 4096x2048 PNG: ' + str(path))
        image.load()
        return image.copy()


def compose(pair, output):
    filename = pair['method_id'] + '__' + pair['prompt_name'] + '.png'
    if Path(filename).name != filename:
        raise ValueError('Unsafe output filename')
    path = output / filename
    if path.exists():
        raise FileExistsError('Refusing to overwrite an existing comparison: ' + str(path))
    canvas = Image.new('RGB', SIZE, 'white')
    draw = ImageDraw.Draw(canvas)
    title = pair['prompt_name'].replace('_', ' ').capitalize()
    if pair['prompt_name'] == 'native_control':
        title += ' (adapted prompt)'
    centered(draw, title + ' | ' + pair['method_label'] + ' | FLUX', 0, 30, SIZE[0], 72, True)
    panels = []
    for steps, box in zip((20, 28), BOXES):
        source = pair['steps' + str(steps)]
        image = checked_image(source)
        centered(draw, str(steps) + ' steps', box[0], 128, ERP_SIZE[0], 64, True)
        canvas.paste(image, box[:2])
        panels.append(dict(steps=steps, box=list(box),
                           pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(), **source))
        image.close()
    centered(draw, 'Seed 0  |  ERP 4096 x 2048  |  same method and prompt',
             0, TOP + ERP_SIZE[1] + 24, SIZE[0], 38)
    temporary = path.with_suffix('.tmp.png')
    canvas.save(temporary, format='PNG', compress_level=4)
    canvas.close()
    with Image.open(temporary) as saved:
        saved.load()
        assert saved.size == SIZE and saved.mode == 'RGB'
        for panel in panels:
            assert hashlib.sha256(saved.crop(panel['box']).tobytes()).hexdigest() == panel['pixels_sha256']
            assert sha(panel['image_path']) == panel['image_sha256']
            assert sha(panel['metadata_path']) == panel['metadata_sha256']
    temporary.replace(path)
    print('FIGURE_VERIFIED', filename, flush=True)
    return dict(pair, figure_path=str(path), figure_sha256=sha(path), size=list(SIZE),
                panels=panels, native_pixels_preserved=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/10.23report/2820')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be between 1 and 8')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(args.sources.read_text())
    pairs = inventory['pairs']
    identities = [(p['method_id'], p['prompt_path'], p['effective_prompt_sha256']) for p in pairs]
    if len(set(identities)) != len(pairs):
        raise ValueError('Duplicate method/prompt identity')
    for pair in pairs:
        if not pair['validation'].get('verified'):
            raise ValueError('Source pair lacks a completed provenance audit')
        if sha(pair['prompt_path']) != pair['prompt_sha256']:
            raise ValueError('Prompt changed after source audit: ' + pair['prompt_path'])
    if list(output.glob('*.png')) or (output / 'manifest.json').exists():
        raise FileExistsError('Output already has comparison figures; use a new destination')
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        figures = list(executor.map(lambda pair: compose(pair, output), pairs))
    manifest = dict(complete=True, figure_count=len(figures), source_image_count=2 * len(figures),
                    column_order=['20 steps', '28 steps'], backend='flux', seed=0,
                    native_erp_size=list(ERP_SIZE), figure_size=list(SIZE),
                    source_inventory_sha256=sha(args.sources), builder_sha256=sha(__file__),
                    figures=figures, missing=inventory.get('missing', []),
                    all_source_images_and_metadata_unchanged=True,
                    all_saved_panels_pixel_identical_to_sources=True)
    atomic_text(output / 'manifest.json', json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    buffer = io.StringIO(newline='')
    fields = ['method', 'prompt', 'prompt_path', 'figure', 'steps20_image', 'steps28_image']
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(dict(method=p['method_label'], prompt=p['prompt_name'], prompt_path=p['prompt_path'],
                         figure=Path(p['figure_path']).name, steps20_image=p['steps20']['image_path'],
                         steps28_image=p['steps28']['image_path']) for p in figures)
    atomic_text(output / 'index.csv', buffer.getvalue())
    print('COMPLETE', len(figures), 'figures;', 2 * len(figures), 'native panels verified', flush=True)


if __name__ == '__main__':
    main()
