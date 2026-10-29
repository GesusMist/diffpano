"""Per-prompt comparisons for the exact eleven scenes in the reported tables."""
import hashlib
import html
import io
import json
import os
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from studies.paper_eval_v2.common import RUN, ROOT, read, write, freeze, sha, now, atomic_text
from studies.paper_eval_v2.cli import verify_protocol
from studies.paper_eval_v2.rendering import VIEWS14, resample_common, render_views, read_view_bytes

OUT = RUN / 'figures/confirmation11'
COHORT = Path('/home/shig/professor-report-20261027-all-variants/confirmation_matched_scores.json')
METHODS = ('sphere20', 'spherope', 'select')
LABELS = {'real': 'Real reference', 'sphere20': 'SphereDiff', 'spherope': 'SpheRoPE', 'select': 'Old89 Select'}
VIEW_IDS = ('p+00_y180', 'north', 'south')
VIEWS = [next(v for v in VIEWS14 if v['view_id'] == name) for name in VIEW_IDS]
FONT_DIR = Path('/usr/share/fonts/dejavu')
PAD, LABEL_WIDTH, GAP, ERP_WIDTH, SIZE = 40, 280, 24, 1024, 512
FIRST_X = PAD + LABEL_WIDTH + GAP
POSITIONS = [FIRST_X, FIRST_X + ERP_WIDTH + GAP,
             FIRST_X + ERP_WIDTH + SIZE + 2 * GAP,
             FIRST_X + ERP_WIDTH + 2 * SIZE + 3 * GAP]
WIDTH = POSITIONS[-1] + SIZE + PAD


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')), size)


def text(draw, value, xy, size=28, bold=False, color='#172333'):
    draw.text(xy, value, font=font(size, bold), fill=color, anchor='lt')


def wrap(value, width, size):
    lines, line = [], ''
    f = font(size)
    for word in value.split():
        candidate = (line + ' ' + word).strip()
        if line and f.getlength(candidate) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    assert all(f.getlength(line) <= width for line in lines)
    return lines


def collect():
    scores = read(COHORT)
    prompt_ids = set(scores[0]['prompt_ids'])
    assert len(prompt_ids) == 11
    assert all(set(row['prompt_ids']) == prompt_ids for row in scores)
    captions = [c for c in read(RUN / 'captions.json')['rows'] if c['prompt_id'] in prompt_ids]
    assert len(captions) == 11
    records = [r for r in read(RUN / 'generation/inventory_records.json')
               if r['prompt_id'] in prompt_ids and r['method_id'] in METHODS and r['seed'] == 0]
    assert len(records) == 33 and all(r['state'] == 'complete' and r['eligible'] for r in records)
    by_cell = {(r['prompt_id'], r['method_id']): r for r in records}
    assert len(by_cell) == 33
    figures = []
    for caption in sorted(captions, key=lambda c: c['scene_id']):
        prompt_path = Path(caption['generation_prompt']['path'])
        assert sha(prompt_path) == caption['generation_prompt']['effective_sha256']
        assert prompt_path.read_text().splitlines() == caption['directional_descriptions']
        panels = [dict(method_id='real', image_path=caption['source_image_path'],
                       image_sha256=caption['source_image_sha256'])]
        for method in METHODS:
            row = by_cell[(caption['prompt_id'], method)]
            assert row['effective_prompt_sha256'] == caption['generation_prompt']['effective_sha256']
            assert sha(row['metadata_path']) == row['metadata_sha256']
            metadata = read(row['metadata_path'])
            assert metadata['image_sha256'] == row['image_sha256']
            panels.append({k: row[k] for k in ('method_id','case_id','image_path','image_sha256',
                           'metadata_path','metadata_sha256','seed')})
            panels[-1]['steps'] = row['parameters']['steps']
        for panel in panels:
            assert sha(panel['image_path']) == panel['image_sha256']
        figures.append(dict(scene_id=caption['scene_id'], prompt_id=caption['prompt_id'],
            category=caption['category'], prompt_path=str(prompt_path),
            prompt_sha256=caption['generation_prompt']['effective_sha256'],
            global_description=caption['global_generation_description'],
            directional_descriptions=caption['directional_descriptions'], panels=panels,
            missing_reference_polar_capture=caption['scene_id'] == 'odisr-test-031'))
    return figures


def compose(figure, cached):
    description = wrap(figure['global_description'], WIDTH - 2*PAD, 29)
    header_y = 128 + 38*len(description) + 34
    first_y = header_y + 72
    footer_y = first_y + 4*(SIZE + 32)
    height = footer_y + 142
    canvas = Image.new('RGB', (WIDTH, height), '#ffffff')
    draw = ImageDraw.Draw(canvas)
    text(draw, figure['scene_id'].replace('odisr-test-', 'ODI-SR scene ') + '  |  ' + figure['category'].capitalize(),
         (PAD, 28), 48, True)
    text(draw, 'Matched confirmation set  •  real reference and three generated panoramas', (PAD, 91), 27, color='#526176')
    for i, line in enumerate(description):
        text(draw, line, (PAD, 136 + 38*i), 29)
    for x, width, heading, sub in zip(POSITIONS, [ERP_WIDTH, SIZE, SIZE, SIZE],
            ['Full panorama', 'Wrap-seam view', 'North pole', 'South pole'],
            ['Equirectangular projection', 'Yaw 180° / pitch 0°', 'Yaw 0° / pitch +90°', 'Yaw 0° / pitch −90°']):
        f = font(30, True)
        text(draw, heading, (x + (width-f.getlength(heading))/2, header_y), 30, True)
        f = font(23)
        text(draw, sub, (x + (width-f.getlength(sub))/2, header_y+37), 23, color='#526176')
    proof = []
    for index, source in enumerate(figure['panels']):
        y = first_y + index*(SIZE+32)
        method = source['method_id']
        draw.rounded_rectangle((PAD-10,y-10,WIDTH-PAD+10,y+SIZE+10),radius=10,
            fill='#edf4fb' if method=='select' else '#f3f5f7')
        text(draw, LABELS[method], (PAD+8,y+175), 34, True,
             color='#135891' if method=='select' else '#172333')
        with Image.open(source['image_path']) as im:
            im.load()
            native_size = im.size
            common = resample_common(np.asarray(im.convert('RGB')))
        text(draw, 'Prompt source' if method=='real' else ('Ours' if method=='select' else 'Baseline'),
             (PAD+8,y+226), 26, color='#526176')
        if method!='real':
            text(draw, str(source['steps'])+' steps · seed 0', (PAD+8,y+268), 25)
        text(draw, str(native_size[0])+' × '+str(native_size[1]), (PAD+8,y+308), 25, color='#526176')
        erp = Image.fromarray(common).resize((ERP_WIDTH,SIZE),Image.Resampling.LANCZOS)
        perspectives = render_views(common,VIEWS,size=SIZE,fov=90.)
        tiles = [erp]+[Image.fromarray(a) for a in perspectives]
        if method!='real':
            # Generated crops must equal the view pixels used for saved scoring.
            for view_id, tile in zip(VIEW_IDS, tiles[1:]):
                record = cached[(source['case_id'],view_id)]
                assert record['source_image_sha256'] == source['image_sha256']
                with Image.open(io.BytesIO(read_view_bytes(record))) as im:
                    assert np.array_equal(np.asarray(tile),np.asarray(im)), (source['case_id'],view_id)
        for j, (x,tile) in enumerate(zip(POSITIONS,tiles)):
            box = (x,y,x+tile.width,y+tile.height)
            canvas.paste(tile,(x,y))
            proof.append(dict(method=method,view='ERP' if j==0 else VIEW_IDS[j-1],
                              box=list(box),pixels_sha256=hashlib.sha256(tile.tobytes()).hexdigest()))
            tile.close()
    footer = [
        'All perspective views: 90° FOV, 512 × 512; rendered from common 2048 × 1024 panoramas.',
        'The real image provides prompt context. Generated scenes are not spatially registered reconstructions.',
        'Reference scene 031 contains missing polar capture regions, retained as recorded.'
        if figure['missing_reference_polar_capture'] else
        'Source images are preserved. All methods use the same display and projection settings.']
    for index,line in enumerate(footer):
        text(draw,line,(PAD,footer_y+12+35*index),25,
             color='#9b4d0b' if figure['missing_reference_polar_capture'] and index==2 else '#526176')
    path = OUT/'png'/(figure['scene_id']+'.png')
    tmp = path.with_suffix('.tmp.png')
    canvas.save(tmp,format='PNG',compress_level=4,dpi=(300,300))
    with Image.open(tmp) as check:
        check.load()
        assert check.size==(WIDTH,height)
        for item in proof:
            assert hashlib.sha256(check.crop(item['box']).tobytes()).hexdigest()==item['pixels_sha256']
    tmp.replace(path)
    canvas.save(OUT/'pdf'/(figure['scene_id']+'.pdf'),format='PDF',resolution=200.,quality=95,
                title=figure['scene_id']+' comparison',author='DiffPano evaluation')
    preview = canvas.copy()
    preview.thumbnail((1700,1700),Image.Resampling.LANCZOS)
    preview.save(OUT/'previews'/(figure['scene_id']+'.jpg'),quality=92,subsampling=0)
    preview.close()
    canvas.close()
    for source in figure['panels']:
        assert sha(source['image_path'])==source['image_sha256']
        if source.get('metadata_path'): assert sha(source['metadata_path'])==source['metadata_sha256']
    record = dict(figure,png=str(path),pdf=str(OUT/'pdf'/(figure['scene_id']+'.pdf')),
                  png_sha256=sha(path),size=[WIDTH,height],panel_proof=proof,
                  generated_views_equal_scoring_views=True,source_hashes_preserved=True)
    write(OUT/'provenance'/(figure['scene_id']+'.json'),record)
    atomic_text(OUT/'prompts'/(figure['scene_id']+'.txt'),Path(figure['prompt_path']).read_text())
    print(json.dumps(dict(event='figure_verified',scene=figure['scene_id'],size=[WIDTH,height],panels=len(proof))),flush=True)
    return record


def package(figures):
    images = []
    try:
        for figure in figures:
            with Image.open(figure['png']) as im: images.append(im.copy())
        images[0].save(OUT/'confirmation11-comparisons.pdf',format='PDF',save_all=True,
            append_images=images[1:],resolution=200.,quality=95,title='DiffPano: 11 matched confirmation scenes')
    finally:
        for im in images:im.close()
    escape = html.escape
    navigation = ' '.join('<a href="#'+f['scene_id']+'">'+f['scene_id'].split('-')[-1]+'</a>' for f in figures)
    blocks = []
    latex = ['% Required package: \\usepackage{graphicx}', '% Paths are relative to this extracted bundle.']
    for figure in figures:
        scene = figure['scene_id']
        links = ' · '.join('<a href="'+path+'">'+label+'</a>' for path,label in
            [('png/'+scene+'.png','Full-resolution PNG'),('pdf/'+scene+'.pdf','PDF'),('prompts/'+scene+'.txt','Five-line prompt')])
        descriptions = ''.join('<li><b>'+label+':</b> '+escape(value)+'</li>' for label,value in zip(
            ['North','Upper','Horizon','Lower','South'],figure['directional_descriptions']))
        blocks.append('<section id="'+scene+'"><h2>'+scene+'</h2><p>'+links+'</p>'
            '<a href="png/'+scene+'.png"><img loading="lazy" src="previews/'+scene+'.jpg" alt="'+scene+' comparison"></a>'
            '<details><summary>Generation prompts</summary><p><b>Global description:</b> '+escape(figure['global_description'])+
            '</p><ol>'+descriptions+'</ol></details></section>')
        latex.extend([r'\begin{figure*}[t]',r'\centering',r'\includegraphics[width=\textwidth]{png/'+scene+'.png}',
            r'\caption{Comparison for ODI-SR scene '+scene.split('-')[-1]+r'. Rows: real reference, SphereDiff (20 steps), SpheRoPE (50 steps), and Old89 Select (ours; 20 steps). Columns: full panorama, wrap-seam view, north pole, and south pole. Perspective views use identical 90-degree FOV and common-resolution projection. The reference supplies prompt context; generated scenes are not registered reconstructions.'+
            (' The real reference contains missing polar capture regions.' if figure['missing_reference_polar_capture'] else '')+'}',
            r'\label{fig:confirmation-'+scene+r'}',r'\end{figure*}',''])
    html_text = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DiffPano — 11 matched confirmation figures</title><style>
body{font:17px/1.55 system-ui,sans-serif;max-width:1500px;margin:36px auto;padding:0 24px;color:#172333;background:#f4f6f9}
h1{line-height:1.2}a{color:#135891}nav{display:flex;gap:18px;flex-wrap:wrap;margin:22px 0}
section{background:white;padding:24px;margin:28px 0;border-radius:12px;box-shadow:0 1px 7px #17233312}
img{display:block;width:100%;height:auto}summary{cursor:pointer;font-weight:600}details{padding:16px;background:#f3f5f7}li{margin:10px 0}
</style><h1>11 matched confirmation scenes</h1><p>One figure per prompt, using the exact scene subset in the reported tables. Each figure compares the real reference, SphereDiff, SpheRoPE, and Old89 Select, with panorama, wrap-seam, and polar views.</p>
<p><a href="confirmation11-comparisons.pdf">Download the 11-page PDF</a> · <a href="figures.tex">LaTeX figure snippets</a></p><nav>'''+navigation+'</nav>'+''.join(blocks)+'</html>'
    atomic_text(OUT/'index.html',html_text)
    atomic_text(OUT/'figures.tex','\n'.join(latex)+'\n')
    manifest = dict(complete=True,created_utc=now(),figure_count=len(figures),source_image_count=44,
        generated_image_count=33,selected_cohort_path=str(COHORT),selected_cohort_sha256=sha(COHORT),
        figures=figures,renderer_source_sha256=sha(ROOT/'studies/paper_eval_v2/rendering.py'),
        builder_sha256=sha(__file__),generated_views_equal_scoring_views=True,
        exported_png_panel_pixels_verified=True,source_hashes_preserved=True,
        generation_performed=False,paid_api_calls=0,job_id=os.environ['SLURM_JOB_ID'])
    write(OUT/'manifest.json',manifest)
    archive=OUT/'confirmation11-figures.zip'
    with zipfile.ZipFile(archive.with_suffix('.tmp.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix not in ('.zip','.lock'):
                z.write(path,str(path.relative_to(OUT)))
    with zipfile.ZipFile(archive.with_suffix('.tmp.zip')) as z:assert z.testzip() is None
    archive.with_suffix('.tmp.zip').replace(archive)
    print(json.dumps(dict(event='complete',figures=11,output=str(OUT),archive=str(archive),archive_bytes=archive.stat().st_size)),flush=True)


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Render figures in a finite CPU allocation'
    cv2.setNumThreads(1)
    verify_protocol(RUN)
    for name in ['png','pdf','previews','prompts','provenance']:(OUT/name).mkdir(parents=True,exist_ok=True)
    figures=collect()
    freeze(OUT/'provenance/selection.json',dict(figures=figures,cohort_path=str(COHORT),cohort_sha256=sha(COHORT)))
    cached={(v['case_id'],v['view_id']):v for v in read(RUN/'vlm/views.json')['views']}
    results=[compose(figure,cached) for figure in figures]
    package(results)


if __name__=='__main__':main()
