"""Assemble verified existing images; never synthesize a missing result."""
import argparse
import copy
import os
from pathlib import Path
from studies.mean_max_old89 import common as mixed
from studies.mean_max_controls import common as controls

ROOT = mixed.ROOT
OUT = ROOT / 'outputs/10.16report_images'
INVENTORY = OUT / 'provenance/control-inventory.json'
ERP = (1536, 768)
PANEL = (1900, 768)
PAD, TITLE, LABEL = 16, 64, 48
SIZE = (3 * PANEL[0] + 4 * PAD,
        TITLE + 4 * (LABEL + PANEL[1] + PAD) + PAD)


def position(index):
    return (0, 1) if index == 0 else (1 + (index - 1) // 3, (index - 1) % 3)


def historical(record):
    if record is None:
        return None
    image = Path(record['path'])
    metadata = Path(record['metadata_path'])
    assert mixed.sha(image) == record['image_sha256'], str(image)
    assert mixed.sha(metadata) == record['metadata_sha256'], str(metadata)
    return dict(record)


def generated(module, row):
    if not module.complete(row, verify=False):
        return None
    folder = module.folder(row)
    metadata = mixed.read(folder / 'metadata.json')
    status = mixed.read(folder / 'status.json')
    image = folder / 'final.png'
    assert mixed.sha(image) == metadata['artifacts']['final.png'] == status['image_sha256']
    assert mixed.sha(folder / 'metadata.json') == status['metadata_sha256']
    assert metadata['camera_geometry_sha256'] == 'b55b4eb5051cdfb8c579e81d4f51ece1f15aeb0e8d7e1fde6b7b74a49154435c'
    assert metadata['refinement']['last_fraction'] == 0
    return dict(path=str(image), image_sha256=mixed.sha(image),
                metadata_path=str(folder / 'metadata.json'), metadata_sha256=status['metadata_sha256'],
                case_id=row['case_id'], gradient_mode=row['gradient_mode'],
                reference_mode=row['reference_mode'], gradient_mix_alpha=row.get('gradient_mix_alpha'),
                source_sha256=row['source_sha256'], semantic_sha256=row['semantic_sha256'],
                tolerance_retry=metadata.get('tolerance_retry'))


def retry_override(manifest):
    """Use the explicitly authorized firefly retry without relabeling the freeze."""
    path = ROOT / 'outputs/10.15mean-max-firefly-retry/seed0-v1/manifest.json'
    if not path.exists():
        return None
    from studies.mean_max_firefly_retry import common as retry
    retry_manifest = retry.freeze_check()
    assert len(retry_manifest['rows']) == 1
    row = retry_manifest['rows'][0]
    original = manifest['rows'][27]
    assert (row['prompt'], row['reference_mode'], row['gradient_mix_alpha']) == ('firefly', 'coarse_color', .25)
    assert row['case_id'] == original['case_id']
    semantic = copy.deepcopy(row['semantic'])
    new_rtol = semantic['solver']['relative_tolerance']
    old_rtol = original['semantic']['solver']['relative_tolerance']
    assert new_rtol > old_rtol
    semantic['solver']['relative_tolerance'] = old_rtol
    assert semantic == original['semantic'], 'Retry changes more than the authorized solver tolerance'
    return retry, row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-all', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Render inside a CPU allocation'
    from PIL import Image, ImageDraw, ImageFont
    from studies.mean_max_figures.pole_panels import make_panel, validate_geometry, PANEL_SIZE, PROTOCOL, RENDERER
    assert PANEL == PANEL_SIZE
    geometry_validation = validate_geometry()
    manifest = mixed.freeze_check()
    override = retry_override(manifest)
    mixed.preserved()
    cm = controls.freeze_check() if (controls.OUT / 'manifest.json').exists() else {'rows': []}
    inventory = mixed.read(INVENTORY)
    by_prompt = {p['prompt']: p['sources'] for p in inventory['prompts']}
    assert set(by_prompt) == set(mixed.PROMPTS)
    title_font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', 40)
    font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', 32)
    figures, pending = [], []
    OUT.mkdir(parents=True, exist_ok=True)
    for name in mixed.PROMPTS:
        old = by_prompt[name]
        panels = [('SphereDiff (original)', historical(old['spherediff'])),
                  ('RGB (ours)', historical(old['rgb']))]
        for reference in ('coarse_color', 'screened'):
            key = reference + '_select'
            record = historical(old[key])
            if record is None:
                rows = [r for r in cm['rows'] if r['prompt'] == name and r['reference_mode'] == reference]
                assert len(rows) == 1, (name, reference, 'Missing control manifest row')
                record = generated(controls, rows[0])
            label = 'lambda_blending' if reference == 'screened' else 'coarse_color'
            panels.append((label + ' + select_closest (ours)', record))
        for reference in ('coarse_color', 'screened'):
            for alpha in mixed.ALPHAS:
                rows = [r for r in manifest['rows'] if r['prompt'] == name and
                        r['reference_mode'] == reference and r['gradient_mix_alpha'] == alpha]
                assert len(rows) == 1
                label = 'lambda_blending' if reference == 'screened' else 'coarse_color'
                module, source_row = mixed, rows[0]
                if override is not None and rows[0]['case_id'] == override[1]['case_id']:
                    module, source_row = override
                panels.append((label + ' + max_blending %03d (ours)' % round(alpha * 100), generated(module, source_row)))
        assert len(panels) == 10
        missing = [label for label, source in panels if source is None]
        if missing:
            pending.append(dict(prompt=name, missing_panels=missing))
            print('PENDING_FIGURE', name, missing, flush=True)
            continue
        canvas = Image.new('RGB', SIZE, 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((PAD, PAD), name.replace('_', ' ').capitalize(), font=title_font, fill='black')
        records = []
        for index, (label, source) in enumerate(panels):
            row, column = position(index)
            x = PAD + column * (PANEL[0] + PAD)
            y = TITLE + PAD + row * (LABEL + PANEL[1] + PAD)
            bounds = draw.textbbox((0, 0), label, font=font)
            assert bounds[2] - bounds[0] <= PANEL[0], label
            draw.text((x + (PANEL[0] - bounds[2] + bounds[0]) // 2, y), label, font=font, fill='black')
            with Image.open(source['path']) as image:
                assert image.size == (4096, 2048), (source['path'], image.size)
                tile = make_panel(image.convert('RGB'))
            canvas.paste(tile, (x, y + LABEL))
            records.append(dict(source, title=label, row=row + 1, column=column + 1))
        path = OUT / (name + '.png')
        temporary = path.with_suffix('.tmp.png')
        canvas.save(temporary)
        os.replace(str(temporary), str(path))
        # Verify every exported panel against its actual resized source.
        with Image.open(path) as exported:
            exported.load()
            assert exported.size == SIZE and exported.mode == 'RGB'
            for index, (_, source) in enumerate(panels):
                row, column = position(index)
                x = PAD + column * (PANEL[0] + PAD)
                y = TITLE + PAD + row * (LABEL + PANEL[1] + PAD) + LABEL
                with Image.open(source['path']) as image:
                    expected = make_panel(image.convert('RGB'))
                assert exported.crop((x, y, x + PANEL[0], y + PANEL[1])).tobytes() == expected.tobytes()
        figures.append(dict(prompt=name, path=str(path), sha256=mixed.sha(path), panels=records))
        print('FIGURE_COMPLETE', name, path, flush=True)
    report = dict(figure_count=len(figures), expected_figure_count=15, complete=len(figures) == 15,
                  figures=figures, pending=pending, backend='flux', seed=0, steps=20,
                  our_camera_layout='old89', original_spherediff='Original implementation and original camera protocol',
                  rows=4, columns=3, figure_size=SIZE, erp_panel_size=ERP, panel_size=PANEL, camera_views=True,
                  layout='SphereDiff at first-row center; RGB/coarse select/screened select on second row; coarse alpha025/050/075 on third; screened alpha025/050/075 on fourth',
                  display='Full existing ERP PNGs and north/south pole views on their right; common display size, unchanged exposure and color',
                  pole_views=PROTOCOL, geometry_validation=geometry_validation,
                  pole_panel_builder_sha256=mixed.sha(Path(__file__).with_name('pole_panels.py')),
                  pole_renderer_sha256=mixed.sha(RENDERER),
                  explanatory_footer=False,
                  label_mapping={'max_blending': 'poisson_mean_max; (1-alpha)*g_mean + alpha*g_max',
                                 'select_closest': 'poisson_select; maximum geometric endpoint confidence, stable camera-ID ties',
                                 'lambda_blending': 'screened reference, lambda_color=0.1'},
                  historical_control_policy_notes=inventory.get('notes', []),
                  authorized_tolerance_override=None if override is None else dict(
                      prompt='firefly', reference_mode='coarse_color', gradient_mix_alpha=.25,
                      previous_relative_tolerance=manifest['rows'][27]['semantic']['solver']['relative_tolerance'],
                      relative_tolerance=override[1]['semantic']['solver']['relative_tolerance'],
                      manifest=str(override[0].OUT / 'manifest.json'),
                      manifest_sha256=mixed.sha(override[0].OUT / 'manifest.json'),
                      original_failed_case_preserved=True),
                  source_inventory_sha256=mixed.sha(INVENTORY), builder_sha256=mixed.sha(__file__),
                  job=os.environ['SLURM_JOB_ID'])
    mixed.write(OUT / 'provenance/figures.json', report)
    mixed.preserved()
    print('REPORT_FIGURES', len(figures), '/ 15', flush=True)
    if args.require_all:
        assert len(figures) == 15, 'Incomplete results remain: ' + ', '.join(p['prompt'] for p in pending)


if __name__ == '__main__':
    main()
