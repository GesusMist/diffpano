"""Honest partial/final tables and fixed-display six-panel comparisons."""
import collections
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from studies.mean_max_old89.common import *


def atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()));tmp.write_text(text);os.replace(str(tmp),str(path))


def latex(groups):
    lines=[r'\begin{table}[t]',r'\centering',r'\caption{Normalized-coefficient mean--max guidance with FLUX and the fixed Old89 cameras. Each complete row uses the same 15 prompts. $\alpha$ is the maximum-gradient coefficient. Time is mean generation time in minutes; setup is excluded. Incomplete groups have no aggregate scores.}',
        r'\label{tab:flux_mean_max}',r'\resizebox{\columnwidth}{!}{',r'\begin{tabular}{lcccccc}',r'\toprule',
        r'Reference & $\alpha$ & $N$ & DS $\downarrow$ & CS $\uparrow$ & IS $\uparrow$ & Time (min) \\',r'\midrule']
    def fmt(value,n=3):return '--' if value is None else ('%.*f'%(n,value))
    for group in groups:
        label=r'lambda\_blending' if group['reference_mode']=='screened' else r'coarse\_color'
        seconds=group['mean_generation_seconds']
        lines.append(' & '.join((label,fmt(group['gradient_mix_alpha'],2),str(group['scored_panorama_count']),fmt(group['mean_DS'],4),fmt(group['mean_CS']),fmt(group['IS']),fmt(None if seconds is None else seconds/60,2)))+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'}',r'\end{table}']
    return '\n'.join(lines)+'\n'


def figures(m):
    from studies.panorama_metrics.render import render
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf',30)
    font_small=ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf',22)
    saved=[];pending=[];sources=[]
    for name in PROMPTS:
        rows=[r for r in m['rows'] if r['prompt']==name]
        if not all(complete(r,verify=False) for r in rows):pending.append(name);continue
        erps=Image.new('RGB',(3*1552,2*832),'white');details=Image.new('RGB',(3*1552,2*590),'white')
        for row in rows:
            column=ALPHAS.index(row['gradient_mix_alpha']);line=REFERENCES.index(row['reference_mode']);directory_case=folder(row)
            metadata=read(directory_case/'metadata.json');assert sha(directory_case/'final.png')==metadata['artifacts']['final.png']
            with Image.open(directory_case/'final.png') as source:rgb=np.asarray(source.convert('RGB'));thumb=source.convert('RGB').resize((1536,768),Image.Resampling.LANCZOS)
            assert rgb.shape==(2048,4096,3)
            label=('lambda_blending' if row['reference_mode']=='screened' else 'coarse_color')+' (ours), alpha = %.2f'%row['gradient_mix_alpha']
            x=column*1552+8;y=line*832+8
            ImageDraw.Draw(erps).text((x,y),label,font=font,fill='black');erps.paste(thumb,(x,y+48))
            # Fixed world view and exact native-pixel crops for all prompts/variants.
            view=render(rgb,views=((0.,0.),),size=512)[0]
            center=rgb[768:1280,1792:2304]
            seam=np.concatenate((rgb[768:1280,-256:],rgb[768:1280,:256]),axis=1)
            dy=line*590+8;draw=ImageDraw.Draw(details);draw.text((x,dy),label,font=font,fill='black')
            for j,(array,caption) in enumerate(((view,'90 deg view, yaw 0, pitch 0'),(center,'ERP center, native 512 x 512'),(seam,'Longitude seam, native 512 x 512'))):
                draw.text((x+j*512,dy+37),caption,font=font_small,fill='black');details.paste(Image.fromarray(array),(x+j*512,dy+67))
            sources.append(dict(prompt=name,reference_mode=row['reference_mode'],gradient_mix_alpha=row['gradient_mix_alpha'],path=str(directory_case/'final.png'),sha256=metadata['artifacts']['final.png']))
        for kind,canvas in (('',erps),('-details',details)):
            target=directory/(name+kind+'.png');temp=target.with_suffix('.tmp.png');canvas.save(temp);os.replace(str(temp),str(target))
        saved.append(name)
    write(directory/'provenance.json',dict(completed_prompts=saved,pending_prompts=pending,sources=sources,
        display='Existing final.png uint8 conversion only; no independent contrast or exposure normalization',
        layout='Top screened alpha .25/.50/.75; bottom coarse_color alpha .25/.50/.75',
        detail=dict(perspective_yaw=0,perspective_pitch=0,fov=90,view_pixels=512,center_crop=[768,1280,1792,2304],seam_crop='rows768:1280, last256 + first256 columns')))
    return saved,pending


def main():
    m=freeze_check();preserved();counts=collections.Counter();failures=[];metadata=[]
    for row in m['rows']:
        path=folder(row);state=read(path/'status.json') if (path/'status.json').exists() else dict(state='missing')
        if complete(row,verify=False):counts['complete']+=1;metadata.append(read(path/'metadata.json'))
        else:
            label='invalid' if state['state']=='complete' else state['state'];counts[label]+=1
            if label in ('failed','invalid'):failures.append(dict(case_id=row['case_id'],error=state.get('error','Invalid provenance/counters')))
    metrics=read(OUT/'evaluation/metrics.json') if (OUT/'evaluation/metrics.json').exists() else None
    if metrics is None:
        from studies.mean_max_old89.evaluate import summarize
        rows=[dict(r,completion_status='missing',scoring_status='pending_generation',DS=None,CS=None) for r in m['rows']]
        groups=summarize(rows,{})
    else:groups=metrics['groups']
    atomic_text(OUT/'results.tex',latex(groups));saved,pending=figures(m)
    lines=['# Mean–max gradient experiment','',
        'Guidance is `(1-alpha)*g_mean + alpha*g_max`, before each Poisson solve. Alpha is the maximum-gradient coefficient. The same current warped predictions supply both statistics in one streaming pass. No magnitude normalization, clipping, contrast adjustment, or extra random draw is applied.','',
        'The exact matrix contains 90 new FLUX20/seed0 trajectories: 15 prompts × two references × three alpha values. All use the saved Old89 cameras at 80° × 80°, 1024² local RGB, 128² native states, and 2048×4096 ERP. GWTF initialization and the identity-preserving current-state VAE bridge are unchanged. Refinement fraction is explicitly zero; there are no replays or camera rotations.','',
        'Solver policy: screened lambda=0.1, rtol=1e-5, atol=1e-7, cap=200; coarse_color eta=0.1, 16×32 block means with a global mean constraint, rtol=1e-4, atol=1e-7, cap=400. Failed convergence remains failure.','',
        'Generation status: `'+json.dumps(dict(counts),sort_keys=True)+'`. Historical trajectories reused: 0.','',
        'Evaluation status: `'+json.dumps(metrics['counts'] if metrics else {},sort_keys=True)+'`. '+('All required metric groups are complete.' if metrics and metrics['complete'] else 'PROVISIONAL: incomplete matched groups have null aggregate scores.'),'',
        'IS uses six separate 15-panorama/120-horizontal-view collections. The original11 and new4 descriptive subsets are independently computed from 88 and 32 views respectively. Per-image IS is never averaged. Metrics, checkpoints, text association, preprocessing, and one-full-split protocol are inherited from the frozen evaluator.','',
        'Comparison figures completed: %d/15. Each has a companion fixed-detail figure with a yaw-0 perspective view, native ERP center crop, and periodic seam crop. Exposure is identical across variants.'%len(saved),'',
        '| Reference | Max coefficient | Scored N / 15 | DS ↓ | CS ↑ | IS ↑ | Generation minutes |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return 'pending' if x is None else '%.4f'%x
    for g in groups:lines.append('| %s | %.2f | %d / 15 | %s | %s | %s | %s |'%(g['reference_mode'],g['gradient_mix_alpha'],g['scored_panorama_count'],fmt(g['mean_DS']),fmt(g['mean_CS']),fmt(g['IS']),fmt(None if g['mean_generation_seconds'] is None else g['mean_generation_seconds']/60)))
    lines += ['', 'The reports record correction RMS/max, mean-color drift, raw ranges/outliers, convergence, stage times, and peak GPU memory in every case metadata file. Sharpness alone is not a quality conclusion: noise, double edges, halos, color drift, and seams require joint review of the fixed displays. No qualitative improvement is claimed by this automatic report.', '', 'Validation:']
    for name in ('cpu.json','weights.json','metrics-cpu.json','gpu.json'):
        path=OUT/'validation'/name;lines.append('- `%s`: %s'%(name,'passed' if path.exists() and read(path).get('passed') else 'pending'))
    if (OUT/'validation/smooth-legacy-comparison.json').exists():
        lines+=['','A smooth full-resolution numerical fixture was rejected by the unchanged coarse-color final FP32 normal-equation check; legacy weighted-mean guidance also fails that fixture. The original failed smoke and first preproduction freeze are archived. This is retained as a strict-rejection test. The positive smoke uses deterministic textured contributions and real FLUX integration; solver tolerances and production settings were not changed.']
    if metadata:
        minutes=[x['generation_seconds']/60 for x in metadata]
        lines+=['','Observed generation minutes: min %.2f, mean %.2f, max %.2f (setup separately recorded). Every accepted row has measured 1780 denoiser calls, 3560 VAE encodes, 1869 decodes and 21 converged Poisson fusions.'%(min(minutes),sum(minutes)/len(minutes),max(minutes))]
    if failures:lines+=['','Unresolved failures:','```json',json.dumps(failures,indent=2),'```']
    lines+=['','Reproduce/resume from `/home/shig/diffpano`:','```bash','python3 -m studies.mean_max_old89.manage status','python3 -m studies.mean_max_old89.manage launch --dry-run','python3 -m studies.mean_max_old89.manage launch --retry','```','',
        'The manager validates the freeze, requires all gates, rejects duplicate active suite jobs, and submits only missing or explicitly retried rows. Evaluation and reporting use Slurm dependencies. Source archive and hash provenance are under `history/`; original request is `REQUEST.txt`.']
    atomic_text(OUT/'report.md','\n'.join(lines)+'\n');preserved()
    print('REPORT',dict(counts),'figures',len(saved),'pending',len(pending),flush=True)

if __name__=='__main__':main()
