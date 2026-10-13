"""Seven-column figures and descriptive, matched-set reporting; no image normalization."""
import argparse
from studies.ournew_comparison.common import *
LABELS=('Official SphereDiff (20 steps)','DiffPano Old89 RGB','DiffPano Uniform89 RGB','Uniform89 select: coarse color','Uniform89 select: global mean','Uniform89 select: screened','Uniform89 select: ERP pixel')


def report():
    m=freeze_check();p=OUT/'evaluation/metrics.json';metrics=read(p) if p.exists() else None
    groups=metrics['groups'] if metrics else []
    lines=['# Four-prompt FLUX comparison','',
        'Seed 0; matched 20 denoising intervals; final ERP 4096 × 2048. DiffPano uses 89 fixed 80° × 80° cameras, RGB 1024 × 1024, native 128 × 128, GWTF and no final independent refinement.',
        'Uniform89 is the canonical Fibonacci phase-0 camera distribution. Fusion retains spherediff_center weighted RGB reduction in every DiffPano variant.',
        'The updated single-pixel constraint uses the current completed weighted RGB ERP at the fixed camera-selected pixel. Location is cached; RGB is recomputed for all 20 intervals and terminal assembly. Historical records retain patch-source semantics.',
        'Coarse color uses rtol 1e-4 consistently; other Poisson references use 1e-5. All use atol 1e-7, screened cap 200 and constrained cap 400.',
        '', '| Method / configuration | N panoramas scored | DS ↓ | CS ↑ | IS ↑ |','|---|---:|---:|---:|---:|']
    tex=['\\begin{tabular}{lrrrr}','Method / configuration & $N$ & DS $\\downarrow$ & CS $\\uparrow$ & IS $\\uparrow$ \\\\','\\hline']
    for v,label in zip(VARIANTS,LABELS):
        g=next((g for g in groups if g['variant']==v),None);n=g['scored_panorama_count'] if g else 0
        scores=[g.get(k) if g else None for k in ('mean_DS','mean_CS','IS')]
        values=['pending' if x is None else '%.6f'%x for x in scores]
        lines.append('| %s | %d/4 | %s |'%(label,n,' | '.join(values)))
        tex.append('%s & %d & %s \\\\'%(label,n,' & '.join('--' if x is None else '%.6f'%x for x in scores)))
    tex.append('\\end{tabular}')
    lines+=['','Each final row uses exactly bamboo_grove, lantern_market, sailboat_marina and tiled_courtyard. Incomplete collections have no aggregate score. IS is a single collection score over 32 fixed horizontal views per variant; no per-image IS or pooling across variants.','',
        '## Comparisons','',
        '- Official SphereDiff versus DiffPano Old89 RGB compares algorithms at matched steps. SphereDiff retains its official spherical representation, cameras, initialization and native resolutions; 20 steps differs from its official 28-step default. Equal steps do not imply equal compute.',
        '- Old89 RGB versus Uniform89 RGB compares camera layouts, including their changed noise transport, overlap and directional prompt assignment.',
        '- Uniform89 RGB versus the four Uniform89 Poisson modes compares blending/reference choices with fixed geometry, conditioning, schedule and starting states.',
        '', 'There is no matched legacy patch-anchor control in this suite. Differences from old patch-anchor results do not isolate anchor color semantics. These four-prompt ablations are descriptive; views are dependent and provide no population-level significance claim.',
        '', '## Per-prompt DS / CS','', '| Prompt | Configuration | Status | DS ↓ | CS ↑ |','|---|---|---|---:|---:|']
    if metrics:
        for r in metrics['rows']:
            lines.append('| %s | %s | %s | %s | %s |'%(r['prompt'],r['variant'],r['scoring_status'],*['pending' if r[k] is None else '%.6f'%r[k] for k in ('DS','CS')]))
    lines+=['','## Execution and provenance','', 'Manifest: [manifest.json](manifest.json). Gates: [validation](validation). Per-case configs, hashes, unclamped DiffPano tensors, terminal RGB references, counts and solver diagnostics: [cases](cases).',
        'Generation status and metric failures are separate. Figure placeholders explicitly identify incomplete generation. Fixed crops use the same PNG exposure, view directions, crop size and scale for all seven methods.',
        '', '```latex']+tex+['```','']
    if metrics:lines+=['Completion counts: '+json.dumps(metrics['counts'])+'.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines));(OUT/'table.tex').write_text('\n'.join(tex)+'\n')


def figures():
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from studies.panorama_metrics.render import render
    m=freeze_check();target=OUT/'figures';target.mkdir(exist_ok=True)
    fontpath='/usr/share/fonts/dejavu/DejaVuSans.ttf'
    try:font=ImageFont.truetype(fontpath,16)
    except OSError:font=ImageFont.load_default()
    cell=576;gap=8;header=50;width=7*(cell+gap)+gap
    directions=((0.,0.),(90.,0.),(180.,0.),(270.,0.),(0.,-35.),(180.,-35.))
    for name in PROMPTS:
        group=[next(r for r in m['rows'] if r['prompt']==name and r['variant']==v) for v in VARIANTS]
        erp=Image.new('RGB',(width,header+cell//2+gap),'white');detail=Image.new('RGB',(width,header+7*(cell+28+gap)),'white')
        for j,(row,label) in enumerate(zip(group,LABELS)):
            x=gap+j*(cell+gap);ImageDraw.Draw(erp).text((x,5),label,fill='black',font=font);ImageDraw.Draw(detail).text((x,5),label,fill='black',font=font)
            if not complete(row):
                for canvas in (erp,detail):ImageDraw.Draw(canvas).text((x,header+10),'Pending / failed generation',fill='black',font=font)
                continue
            with Image.open(folder(row)/'final.png') as image:rgb=np.asarray(image.convert('RGB'));small=image.convert('RGB').resize((cell,cell//2),Image.Resampling.LANCZOS)
            erp.paste(small,(x,header));views=render(rgb,directions,size=1024)
            for k,((yaw,pitch),view) in enumerate(zip(directions,views)):
                # Central native-detail crop: same source pixels and scale for every method.
                tile=Image.fromarray(view).crop((224,224,800,800));y=header+k*(cell+28+gap)
                ImageDraw.Draw(detail).text((x,y),'yaw %g°, pitch %g°; fixed detail'%(yaw,pitch),fill='black',font=font);detail.paste(tile,(x,y+28))
            # An explicit periodic longitude-seam crop, without filling or blending it.
            strip=np.concatenate((rgb[768:1280,-288:],rgb[768:1280,:288]),axis=1)
            y=header+6*(cell+28+gap);ImageDraw.Draw(detail).text((x,y),'ERP longitude seam, native pixels',fill='black',font=font);detail.paste(Image.fromarray(strip),(x,y+28))
        erp.save(target/(name+'-seven-way.png'));detail.save(target/(name+'-fixed-details.png'))
    write(target/'protocol.json',dict(variant_order=list(VARIANTS),directions=directions,perspective_fov=90,render_size=1024,
        crop=[224,224,800,800],ERP_seam=dict(y=[768,1280],left_right_width=288),exposure='unchanged uint8 final.png',
        per_method_normalization=False,detail_note='Fixed horizontal views expose bamboo/market/marina structures; lower-pitch views expose tiled paving and water detail; yaw180 and explicit seam include longitude boundary.'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--text-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);report()
    if not args.text_only:figures()
    preserved()
