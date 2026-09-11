#!/usr/bin/env python3
"""Render a completed A-G ladder from saved controls; no Torch/model imports."""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageOps

spec=importlib.util.spec_from_file_location('seams',Path(__file__).resolve().parents[1]/'diffpano/seams.py')
seams=importlib.util.module_from_spec(spec);spec.loader.exec_module(seams)


def read(path):return json.loads(Path(path).read_text())
def save(fig,path):
    for ext in ('png','pdf'):fig.savefig(str(path)+'.'+ext,dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',default='configs/experiments/trajectory/sd35-ladder.json')
    args=parser.parse_args();protocol=read(args.protocol);out=Path(protocol['report']);out.mkdir(parents=True,exist_ok=True)
    controls={label:read(Path(folder)/(label+'_control.json')) for label,folder in protocol['folders'].items()}
    title={'A':'Native single patch','B':'Implied endpoint','C':'RGB roundtrip','D':'Roundtrip + residual',
           'E':'Native MultiDiffusion','F':'RGB consensus','G':'RGB consensus + residual'}
    fig,axes=plt.subplots(2,4,figsize=(16,9),layout='constrained')
    for label,ax in zip('ABCDEFG',axes.flat):
        source=Image.open(controls[label]['output']).convert('RGB')
        display=ImageOps.pad(source,(512,512),color='white',method=Image.Resampling.LANCZOS)
        ax.imshow(display);ax.set_title(label+' — '+title[label]);ax.axis('off')
    axes[1,3].axis('off');axes[1,3].text(.05,.6,'SD3.5 Medium\n40 steps · CFG 4.5\n1024 px local patch\nTraining-free correction',transform=axes[1,3].transAxes,fontsize=14)
    save(fig,out/'A-G-contact-sheet')
    for labels,name in [('ABCD','A-B-C-D'),('EFG','E-F-G')]:
        fig,axes=plt.subplots(1,len(labels),figsize=(5*len(labels),5),layout='constrained')
        for ax,label in zip(axes,labels):
            ax.imshow(Image.open(controls[label]['output']));ax.set_title(label+' — '+title[label]);ax.axis('off')
        save(fig,out/name)
    a,b=[read(Path(protocol['folders'][p])/(p+'_steps.json')) for p in 'AB']
    fig,axes=plt.subplots(3,1,figsize=(9,9),layout='constrained')
    for steps,label in ((a,'Same-prediction one-step oracle'),(b,'Independent full trajectories')):
        for ax,suffix in zip(axes,('mae','rmse','max_abs')):
            ax.plot([s['timestep'] for s in steps],[s['native_vs_implied_'+suffix] for s in steps],label=label)
            ax.set_ylabel('Native vs implied '+suffix);ax.invert_xaxis() if label.startswith('Same') else None
    for ax in axes:ax.legend();ax.set_yscale("symlog",linthresh=1e-8)
    axes[-1].set_xlabel('Actual scheduler timestep');save(fig,out/'A-B-errors')
    c,d=[read(Path(protocol['folders'][p])/(p+'_steps.json')) for p in 'CD']
    fig,axes=plt.subplots(3,1,figsize=(9,9),layout='constrained')
    for steps,label in ((c,'C'),(d,'D')):
        times=[s['timestep'] for s in steps]
        axes[0].plot(times,[s['roundtrip_mae'] for s in steps],label=label)
        axes[1].plot(times,[s['relative_residual_mae'] for s in steps],label=label)
    axes[2].plot([s['timestep'] for s in d],[s['corrected_vs_original_mae'] for s in d],label='D recovery')
    for ax,title_ in zip(axes,('Roundtrip residual MAE','MAE / clean latent std','Corrected-clean recovery MAE')):
        ax.set_ylabel(title_);ax.invert_xaxis();ax.legend()
    axes[-1].set_xlabel('Actual scheduler timestep');save(fig,out/'C-D-residuals')
    summary=dict(backend=protocol['backend'],controls=controls,
        A_one_step_mae_mean=float(np.mean([s['native_vs_implied_mae'] for s in a])),
        A_one_step_max_abs=max(s['native_vs_implied_max_abs'] for s in a),
        B_trajectory_mae_mean=float(np.mean([s['native_vs_implied_mae'] for s in b])),
        B_trajectory_max_abs=max(s['native_vs_implied_max_abs'] for s in b),
        D_recovery_mae_mean=float(np.mean([s['corrected_vs_original_mae'] for s in d])),
        D_recovery_max_abs=max(s['corrected_vs_original_max_abs'] for s in d))
    for label,steps in (('C',c),('D',d)):
        summary[label+'_residual']=dict(first=steps[0]['roundtrip_mae'],last=steps[-1]['roundtrip_mae'],
            peak=max(s['roundtrip_mae'] for s in steps),mean=float(np.mean([s['roundtrip_mae'] for s in steps])),
            normalized_mean=float(np.mean([s['relative_residual_mae'] for s in steps])))
    fig,axes=plt.subplots(3,1,figsize=(9,9),layout='constrained')
    for label in 'EFG':
        m=read(Path(protocol['folders'][label])/'metadata.json');steps=m['steps'];times=[s['scheduler_timestep'] for s in steps]
        axes[0].plot(times,[s['state_statistics']['pre_fusion_rgb_overlap_mae_mean'] for s in steps],label=label)
        if label in 'FG':axes[1].plot(times,[s['state_statistics']['consensus_correction_mae_mean'] for s in steps],label=label)
        if label=='G':
            for key in ('local_vae_residual_mae_mean','fused_residual_mae_mean','residual_correction_magnitude'):
                axes[2].plot(times,[s['state_statistics'][key] for s in steps],label=key)
        image=np.asarray(Image.open(controls[label]['output']).convert('RGB'))/255.
        summary[label+'_seams']=seams.boundary_gradient_metrics(image,controls[label]['geometry']['rgb']['patches'])
        keys=['pre_fusion_rgb_overlap_mae_mean','pre_fusion_rgb_overlap_mae_max']
        if label in 'FG':keys+=['consensus_correction_mae_mean']
        if label=='G':keys+=['local_vae_residual_mae_mean','local_vae_residual_mae_max','fused_residual_mae_mean','fused_residual_std','residual_correction_magnitude']
        summary[label+'_diagnostic_means']={key:float(np.mean([s['state_statistics'][key] for s in steps])) for key in keys}
    for ax,title_ in zip(axes,('Pre-fusion clean RGB overlap MAE','RGB consensus correction MAE','G native residual MAE')):
        ax.set_ylabel(title_);ax.invert_xaxis();ax.legend(fontsize=8)
    axes[-1].set_xlabel('Actual scheduler timestep');save(fig,out/'E-F-G-diagnostics')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Saved',out,flush=True)


if __name__=='__main__':main()
