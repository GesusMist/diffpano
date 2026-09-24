"""Numerical/visual artifacts first; final research interpretation after review."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from studies.gwtf_noise.common import ROOT,BASE,CELL,BACKENDS,read,write,verify_preservation,require_validation
from studies.gwtf_noise.statistics import METHODS

LABELS={'direct':'Direct iid','nearest':'Nearest shared','gwtf':'GWTFlow-style'}


def plot_noise(backends):
    folder=ROOT/'report';folder.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(15,8));x=np.arange(5)
    specs=[('Variance',lambda s:s['variance']),('Horizontal covariance',lambda s:s['offsets']['h']['covariance']),
        ('Vertical covariance',lambda s:s['offsets']['v']['covariance']),('Moran-equivalent',lambda s:s['moran_equivalent']),
        ('Neighbor exact-duplicate rate',lambda s:max(s['offsets'][k]['exact_equality'] for k in ('h','v','diag'))),
        ('Matched-ray cross-view correlation',lambda s:s['matched_ray_correlation'])]
    for ax,(title,get) in zip(axes.flat,specs):
        for j,m in enumerate(METHODS):ax.bar(x+(j-1)*.25,[get(backends[b]['summary'][m]) for b in BACKENDS],.25,label=LABELS[m])
        ax.set_xticks(x,list(BACKENDS));ax.set_title(title);ax.grid(axis='y',alpha=.2)
        if title=='Variance':ax.set_ylim(.95,1.05);ax.axhline(1,color='black',linewidth=.7)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3)
    fig.suptitle('Initialization statistics: 512 realizations per camera, full native geometry',y=.96)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(folder/'noise-statistics.png',dpi=170);fig.savefig(ROOT/'noise-statistics.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(18,3.5))
    for ax,name in zip(axes,BACKENDS):
        rows=backends[name]['cross_view']
        angles=np.concatenate([r['angular_separation_degrees'] for r in rows.values()])
        edges=np.unique(np.quantile(angles,np.linspace(0,1,9)))
        for method in METHODS:
            corr=np.concatenate([r['methods'][method]['correlation_per_pair'] for r in rows.values()]);xs=[];ys=[]
            for lo,hi in zip(edges[:-1],edges[1:]):
                keep=(angles>=lo)&(angles<=hi if hi==edges[-1] else angles<hi)
                if keep.any():xs.append(angles[keep].mean());ys.append(corr[keep].mean())
            ax.plot(xs,ys,'o-',label=LABELS[method],markersize=3)
        ax.set_title(name);ax.set_xlabel('Ray separation (degrees)');ax.axhline(0,color='gray',linewidth=.6);ax.grid(alpha=.2)
    axes[0].set_ylabel('Matched-ray correlation');axes[-1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(folder/'correlation-vs-angular-separation.png',dpi=170);plt.close(fig)


def figures():
    manifest=require_validation(statistical=True);verify_preservation();folder=ROOT/'report';folder.mkdir(parents=True,exist_ok=True)
    stats={n:read(ROOT/'statistical-preflight'/(n+'.json')) for n in BACKENDS}
    rows=[];panels=[];generations={};missing=[]
    font=ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf',17)
    for name in BACKENDS:
        path=ROOT/name/'metadata.json'
        if not path.is_file():missing.append(name);continue
        new=read(path);assert new['actual_transformer_forward_invocations']==manifest['models'][name]['expected_predictions']
        oldfolder=Path(new['reused_current_control']);old=read(oldfolder/'metadata.json')
        assert new['config']==old['config']
        for key in ('model_checkpoint','model_revision','prepared_schedule','conditioning_sha256','prompt_indices','camera_sha256',
                    'camera_geometry_sha256','native_channels','local_native_resolution','noise_grid','native_scale','bridge_mode','vae_dtype'):
            assert new[key]==old[key],(name,key)
        generations[name]=dict(current=old,new=new,current_image=str(oldfolder/'final_result.png'),new_image=str(ROOT/name/'gwtf-final.png'))
        for method,m in (('nearest',old),('gwtf',new)):
            rows.append(dict(backend=name,method=method,**m['summary'],runtime_seconds=m['runtime_seconds'],job=m['environment']['job']))
        panel=Image.new('RGB',(1312,368),(245,245,245));draw=ImageDraw.Draw(panel)
        for col,(label,p) in enumerate((('Current nearest shared',oldfolder/'final_result.png'),('GWTFlow-style shared',ROOT/name/'gwtf-final.png'))):
            with Image.open(p) as image:
                assert image.width==2*image.height
                view=image.convert('RGB').resize((640,320),Image.Resampling.LANCZOS)
            panel.paste(view,(col*656+8,40));draw.text((col*656+8,8),name+' | '+label,font=font,fill='black')
        panel.save(folder/(name+'-current-vs-gwtf.png'));panels.append(panel)
    if panels:
        sheet=Image.new('RGB',(1312,368*len(panels)),(245,245,245))
        for i,p in enumerate(panels):sheet.paste(p,(0,368*i))
        sheet.save(folder/'all-backends-current-vs-gwtf.png')
    write(folder/'numerical-summary.json',dict(complete=not missing,missing=missing,
        initialization={n:s['summary'] for n,s in stats.items()},generations=generations,rows=rows,
        costs={n:{m:{k:v[k] for k in ('initialization_seconds','host_process_max_rss_gib','persistent_state_bytes')} for m,v in s['exact_seed0'].items()} for n,s in stats.items()},
        statistical_jobs={n:s['job'] for n,s in stats.items()}))
    if rows:
        with (folder/'summary.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    initrows=[]
    for n,s in stats.items():
        for m,v in s['summary'].items():
            initrows.append(dict(backend=n,method=m,mean=v['mean'],std=v['std'],h_cov=v['offsets']['h']['covariance'],v_cov=v['offsets']['v']['covariance'],
                diagonal_cov=v['offsets']['diag']['covariance'],moran_equivalent=v['moran_equivalent'],duplicate_rate=max(v['offsets'][k]['exact_equality'] for k in ('h','v','diag')),
                matched_ray_corr=v['matched_ray_correlation'],random_pair_abs_cov=v['offsets']['random']['mean_absolute_pair_covariance'],skewness=v['skewness'],excess_kurtosis=v['excess_kurtosis']))
    with (folder/'initialization-statistics.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(initrows[0]));w.writeheader();w.writerows(initrows)
    print('Comparison artifacts complete; missing backends:',missing,flush=True)
    if missing:raise SystemExit(1)


def finalize():
    folder=ROOT/'report';data=read(folder/'numerical-summary.json');review=read(folder/'visual-review.json')
    assert data['complete'] and set(review['backends'])==set(BACKENDS)
    verify_preservation();validation=read(ROOT/'validation.json');execution=read(ROOT/'execution.json')
    data.update(visual_review=review,validation=validation,execution=execution)
    write(folder/'summary.json',data)
    out=['# GWTFlow-style shared-noise initialization: five-backend study','',
        '## A. Method','',Path('studies/gwtf_noise/METHOD.md').read_text(),
        '## B. Statistical preflight','',
        'Each native geometry uses 89 cameras, 512 independent Gaussian realizations per camera/category, actual channel count, and the unchanged primary ERP grid. The sparse Monte Carlo preserves all conditional sibling edges. Diagnostics have separate RNG streams. Population-zero-centred neighbor covariance divided by variance is reported as Moran-equivalent; it is not a finite-image Moran significance test.','',
        '|Backend|Method|Mean|Std|H-cov|V-cov|Diagonal|Moran-equiv.|Max neighbor duplicate|Matched-ray corr|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in BACKENDS:
        for method in METHODS:
            v=data['initialization'][name][method]
            out.append('|{}|{}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.7f}|{:.5f}|'.format(name,LABELS[method],v['mean'],v['std'],v['offsets']['h']['covariance'],v['offsets']['v']['covariance'],v['offsets']['diag']['covariance'],v['moran_equivalent'],max(v['offsets'][k]['exact_equality'] for k in ('h','v','diag')),v['matched_ray_correlation']))
    out+=['','![Noise statistics](noise-statistics.png)','','![Correlation versus angular separation](correlation-vs-angular-separation.png)','',
          'Additional offsets, random-pair absolute covariance, exact equality, skewness, kurtosis, quantiles, KS effect sizes, per-camera estimates and category-specific angular separations are retained in each statistical-preflight JSON.','',
          '## C. Generation results','','|Backend|Method|Contrast|HF1/std|Pair MAE|Normalized pair MAE|View-consensus MAE|Out-of-range|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in data['rows']:
        out.append('|{backend}|{method}|{contrast:.5f}|{hf1_normalized:.5f}|{aligned_pair_mae:.5f}|{aligned_pair_normalized:.5f}|{view_to_consensus:.5f}|{out_of_range:.6f}|'.format(**r))
    for name in BACKENDS:
        v=review['backends'][name];new=data['generations'][name]['new']
        out += ['', '### '+name,'',v['observation'],'',
            f"Generation job {new['environment']['job']}; {new['actual_transformer_forward_invocations']} actual forwards; runtime {new['runtime_seconds']:.3f} s. Current control: `{new['reused_current_control']}`.",
            '',f'![{name} current vs GWTFlow]({name}-current-vs-gwtf.png)','',
            '|Step|View-consensus MAE|RGB modification|Native modification|Current-state error max|','|---:|---:|---:|---:|---:|']
        for r in new['milestones']:
            out.append('|{step}|{overlap_view_to_consensus_mean:.6f}|{rgb_consensus_delta_mean:.6f}|{native_consensus_delta_mean:.6f}|{current_state_error_max:.4g}|'.format(**r))
        out+=['','The metadata `audit.stage_audit` retains matched-pair MAE, normalized MAE, local contrast, absolute HF1, HF1/std and pre-clipping out-of-range RGB at these same milestones, using existing trajectory predictions.']
    out+=['','## D. Cross-backend interpretation','',review['cross_backend'],'','### Explicit research questions','']
    out+=['{}. {}'.format(i+1,answer) for i,answer in enumerate(review['answers'])]
    out+=['','## E. Cost and validation','',f"Full validation job {validation['job']}: {validation['suites']['focused']['count']} focused tests and {validation['suites']['full']['count']} full-suite tests passed; total validation {validation['seconds']:.3f} s. Compilation and diff checks passed. Source hashes were frozen and checked at every generation launch. Original code and matched historical images/metadata remain unchanged.",
          '','|Backend|Initializer|Seconds|Process max RSS GiB|Persistent local-state MiB|','|---|---|---:|---:|---:|']
    for name in BACKENDS:
        for method in METHODS:
            c=data['costs'][name][method]
            out.append('|{}|{}|{:.3f}|{:.3f}|{:.3f}|'.format(name,method,c['initialization_seconds'],c['host_process_max_rss_gib'],c['persistent_state_bytes']/2**20))
    out+=['','These CPU-only initializer timings exclude model generation. Process RSS peaks include geometry, diagnostics, Python and retained allocator memory; they are not exclusive per-method allocation deltas. Source/graph byte counts and geometry/sampling times are in initialization metadata. Zero denoiser or VAE calls occurred during initialization/preflight. No runtime speedup is inferred across shared nodes.','',
          '## F. Limitations','','One seed and one ruins prompt for generation; Monte Carlo characterizes the initializer, not a population of generated scenes. Matched-ray/local-overlap diagnostics do not establish global 3D coherence. HF can represent artifacts. PixelDiT is VAE-free evidence, but its differences from latent models do not isolate VAE causality. The implementation follows published Algorithm 1 with a one-hop ERP adaptation, not the exact released temporal video system.','',
          '## Baseline decision','',review['recommendation'],'','## Single next experiment (not implemented)','',review['next_experiment'],'']
    (folder/'report.md').write_text('\n'.join(out))
    print('FINAL REPORT COMPLETE',folder,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['figures','finalize']);a=p.parse_args()
    figures() if a.phase=='figures' else finalize()
