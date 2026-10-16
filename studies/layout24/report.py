"""Nine-variant matched tables and three-by-three same-prompt ERP figures."""
import collections
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from studies.layout24.common import ROOT,OUT,VARIANTS,read,write,sha,folder,complete,freeze_check
from studies.followup24.common import PROMPTS24

LABELS={
    'spherediff_official_20':'SphereDiff',
    'diffpano_old89_rgb':'Old89 RGB (ours)',
    'diffpano_old70_rgb':'Old70 RGB (ours)',
    'diffpano_old50_rgb':'Old50 RGB (ours)',
    'diffpano_uniform89_rgb':'Uniform89 RGB (ours)',
    'diffpano_uniform70_rgb':'Uniform70 RGB (ours)',
    'diffpano_uniform50_rgb':'Uniform50 RGB (ours)',
    'diffpano_old89_screened_select':'Old89 lambda_blending + select (ours)',
    'diffpano_uniform89_screened_select':'Uniform89 lambda_blending + select (ours)',
}

def atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()));temporary.write_text(text)
    os.replace(str(temporary),str(path))

def fmt(value,places=3):return '--' if value is None else '%.*f'%(places,value)

def latex(groups):
    lines=[r'\begin{table}[t]',r'\centering',
        r'\caption{FLUX comparison on the matched 24 scenes (20 original and four new). All panoramas use seed 0, 20 denoising intervals, and $4096\times2048$ ERP. $N$ is the camera count. Incomplete groups have no aggregate score. Time is mean measured generation minutes among cases with recorded timing; historical missing times are excluded.}',
        r'\label{tab:flux_layout24}',r'\resizebox{\columnwidth}{!}{',r'\begin{tabular}{lccccc}',r'\toprule',
        r'Variant & $N$ & DS $\downarrow$ & CS $\uparrow$ & IS $\uparrow$ & Time (min) \\',r'\midrule']
    for group in groups:
        label=LABELS[group['variant']].replace('_',r'\_')
        time=group['mean_generation_seconds']
        lines.append(' & '.join((label,str(group['camera_count']),fmt(group['mean_DS'],4),fmt(group['mean_CS']),fmt(group['IS']),fmt(None if time is None else time/60,2)))+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}',r'}',r'\end{table}'])+'\n'

def figures(manifest):
    target=OUT/'figures';target.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf',24)
    saved=[];pending=[];provenance=[]
    old=read(target/'provenance.json') if (target/'provenance.json').exists() else {}
    previous={r['prompt']:r for r in old.get('figures',[])}
    for name in PROMPTS24:
        group=[next(r for r in manifest['rows'] if r['prompt']==name and r['variant']==variant) for variant in VARIANTS]
        if not all(complete(r) for r in group):pending.append(name);continue
        sources=[dict(variant=r['variant'],path=str(folder(r)/'final.png'),sha256=sha(folder(r)/'final.png'),metadata_sha256=sha(folder(r)/'metadata.json')) for r in group]
        path=target/(name+'.png');previous_record=previous.get(name,{})
        if previous_record.get('sources')==sources and path.exists() and previous_record.get('sha256')==sha(path):
            saved.append(name);provenance.append(previous_record);continue
        canvas=Image.new('RGB',(3*1040,3*572),'white');draw=ImageDraw.Draw(canvas)
        for index,(row,source) in enumerate(zip(group,sources)):
            x=(index%3)*1040+8;y=(index//3)*572+8
            draw.text((x,y),LABELS[row['variant']],fill='black',font=font)
            with Image.open(source['path']) as image:
                assert image.size==(4096,2048)
                canvas.paste(image.convert('RGB').resize((1024,512),Image.Resampling.LANCZOS),(x,y+44))
        temporary=path.with_suffix('.tmp.png');canvas.save(temporary);os.replace(str(temporary),str(path))
        saved.append(name);provenance.append(dict(prompt=name,path=str(path),sha256=sha(path),sources=sources))
    write(target/'provenance.json',dict(figures=provenance,complete_prompts=saved,pending_prompts=pending,
        order=list(VARIANTS),display='Original final PNG, identical Lanczos display resize only; no exposure or contrast changes'))
    return saved,pending

def main():
    manifest=freeze_check();counts=collections.Counter();variant_counts={v:collections.Counter() for v in VARIANTS}
    for row in manifest['rows']:
        path=folder(row)/'status.json';status=read(path) if path.exists() else {}
        if complete(row,verify=False):state='reused' if status.get('reused') else 'new_complete'
        else:state='invalid' if status.get('state')=='complete' else status.get('state','missing')
        counts[state]+=1;variant_counts[row['variant']][state]+=1
    metric_path=OUT/'evaluation/metrics.json';metrics=read(metric_path) if metric_path.exists() else None
    if metrics:groups=metrics['groups']
    else:
        groups=[dict(variant=v,camera_count=next(r['camera_count'] for r in manifest['rows'] if r['variant']==v),
            mean_DS=None,mean_CS=None,IS=None,mean_generation_seconds=None,timed_panorama_count=0,scored_panorama_count=0) for v in VARIANTS]
    atomic_text(OUT/'results.tex',latex(groups));saved,pending=figures(manifest)
    lines=['FLUX 24-scene panorama comparison','',
        'Generation status: '+str(dict(counts))+'.',
        'Metric status: '+str(metrics['counts'] if metrics else 'pending')+'.',
        'Figures: %d / 24 complete.'%len(saved),'',
        'Nine distinct matched collections use the same 24 scene prompts. IS is recomputed from 192 horizontal views per variant; original20 and new4 subset IS use 160 and 32 views independently. Missing or incomplete groups are null. Standalone perspective images never enter these metrics.','',
        '| Variant | N | Scored / 24 | DS ↓ | CS ↑ | IS ↑ | Measured minutes | Timed cases |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        seconds=g['mean_generation_seconds']
        lines.append('| %s | %d | %d | %s | %s | %s | %s | %d |'%(LABELS[g['variant']],g['camera_count'],g['scored_panorama_count'],fmt(g['mean_DS'],4),fmt(g['mean_CS']),fmt(g['IS']),fmt(None if seconds is None else seconds/60,2),g['timed_panorama_count']))
    lines+=['','Measured time excludes setup when generation-region timing is available. Unrecorded historical timing remains missing. Reused results retain their original metadata and source provenance. Identical numeric seed does not imply identical initial states across spherical, GWTF, and ordinary standalone representations.','',
        'Status and resume from /home/shig/diffpano:','```bash','python3 -m studies.followup24.manage status','python3 -m studies.followup24.manage launch --dry-run','python3 -m studies.followup24.manage launch --retry','```','',
        'Original prompt files, completed normalized mean-max results, and the outputs scratch symlink are preserved. No scientific comparison conclusion is inferred automatically.']
    atomic_text(OUT/'report.md','\n'.join(lines)+'\n')
    write(OUT/'status-summary.json',dict(counts=dict(counts),per_variant={k:dict(v) for k,v in variant_counts.items()},metrics=metrics['counts'] if metrics else None,
        figures=len(saved),pending_figures=pending,complete=bool(metrics and metrics['complete'] and len(saved)==24)))
    print('REPORT',dict(counts),'figures',len(saved),'metrics',metrics['counts'] if metrics else None,flush=True)

if __name__=='__main__':main()
