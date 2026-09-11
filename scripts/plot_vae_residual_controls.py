#!/usr/bin/env python3
"""Render training-free C/D and E/F/G controls from saved artifacts."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_spec=importlib.util.spec_from_file_location('seams',Path(__file__).resolve().parents[1]/'diffpano/seams.py')
_seams=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_seams)


def read(path):return json.loads(Path(path).read_text())


def sheet(paths,titles,output):
    fig,axes=plt.subplots(1,len(paths),figsize=(6*len(paths),5),layout='constrained')
    for ax,path,title in zip(axes,paths,titles):
        ax.imshow(Image.open(path));ax.set_title(title);ax.axis('off')
    for extension in ('png','pdf'):fig.savefig(str(output)+'.'+extension,dpi=140)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--single-only',action='store_true');parser.add_argument('specs',nargs='+');args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);rows=[]
    for spec_path in args.specs:
        spec=read(spec_path);b=spec['backend'];folder=Path(spec['output']);previous=Path(spec['previous_pair']);cd=read(folder/'CD/comparison.json')
        assert len(set(cd['counts'].values()))==1 and cd['training_free']
        c,d=read(folder/'CD/C_steps.json'),read(folder/'CD/D_steps.json')
        assert [s['timestep'] for s in c]==[s['timestep'] for s in d]
        sheet([folder/'CD'/(label+'_final.png') for label in ('B','C','D')],
              [b.upper()+' B: implied',b.upper()+' C: RGB roundtrip',b.upper()+' D: residual correction'],out/(b+'-BCD'))
        row=dict(backend=b,source=str(folder),D_vs_B_final_rgb_mae=cd['D_vs_B']['final_rgb_mae'],C_vs_B_final_rgb_mae=cd['C_vs_B']['final_rgb_mae'])
        for label,steps in (('C',c),('D',d)):
            for key in ('roundtrip_residual_mae','roundtrip_residual_rmse','residual_std','relative_residual_mae'):
                row[label+'_'+key+'_mean']=float(np.mean([s[key] for s in steps]))
            row[label+'_residual_mae_first']=steps[0]['roundtrip_residual_mae'];row[label+'_residual_mae_last']=steps[-1]['roundtrip_residual_mae']
            row[label+'_residual_mae_peak']=max(s['roundtrip_residual_mae'] for s in steps)
        row['D_recovery_mae_mean']=float(np.mean([s['correction_recovery_mae'] for s in d]))
        row['D_recovery_max_abs']=max(s['correction_recovery_max_abs'] for s in d)
        fig,axes=plt.subplots(3,1,figsize=(8,8),layout='constrained')
        for steps,label in ((c,'C'),(d,'D')):
            times=[s['timestep'] for s in steps]
            axes[0].plot(times,[s['roundtrip_residual_mae'] for s in steps],label=label)
            axes[1].plot(times,[s['relative_residual_mae'] for s in steps],label=label)
        axes[2].plot([s['timestep'] for s in d],[s['correction_recovery_mae'] for s in d],label='D recovery MAE')
        for ax,title in zip(axes,('VAE residual MAE (native units)','Residual MAE / clean latent std','Corrected clean recovery MAE')):
            ax.set_ylabel(title);ax.invert_xaxis();ax.legend()
        axes[2].set_xlabel('Actual scheduler timestep');fig.suptitle(b.upper())
        for ext in ('png','pdf'):fig.savefig(out/(b+'-residuals.'+ext),dpi=140)
        plt.close(fig)
        if not args.single_only:
            g=read(folder/'G/comparison.json');assert g['training_free'] and g['same_initial_model_settings_schedule']
            files=[previous/'native_multidiffusion/result.png',previous/'implied_endpoint_consensus/result.png',folder/'G/generation/result.png']
            sheet(files,[b.upper()+' E: native MD',b.upper()+' F: RGB consensus',b.upper()+' G: residual correction'],out/(b+'-EFG'))
            patches=read(previous/'comparison.json')['geometry']['rgb']['patches']
            for label,path in zip(('E','F','G'),files):
                rgb=np.asarray(Image.open(path).convert('RGB'))/255.
                metric=_seams.boundary_gradient_metrics(rgb,patches)
                for key in ('boundary_gradient','nearby_gradient','boundary_excess','boundary_to_nearby_ratio'):row[label+'_'+key]=metric[key]
            gs=read(folder/'G/generation/metadata.json')['steps']
            fs=read(previous/'implied_endpoint_consensus/metadata.json')['steps']
            assert [s['scheduler_timestep'] for s in gs]==[s['scheduler_timestep'] for s in fs]
            fig,axes=plt.subplots(3,1,figsize=(9,9),layout='constrained')
            for steps,label in ((fs,'F'),(gs,'G')):
                times=[s['scheduler_timestep'] for s in steps]
                axes[0].plot(times,[s['state_statistics']['pre_fusion_rgb_overlap_mae_mean'] for s in steps],label=label)
                axes[2].plot(times,[s['state_statistics']['consensus_correction_mae_mean'] for s in steps],label=label+' RGB')
            times=[s['scheduler_timestep'] for s in gs]
            for key,label in (('local_vae_residual_mae_mean','Local residual'),('fused_residual_mae_mean','Fused residual'),('residual_correction_magnitude','Cropped correction')):
                axes[1].plot(times,[s['state_statistics'][key] for s in gs],label=label)
            for ax,title in zip(axes,('Pre-fusion RGB overlap MAE','G native residual MAE','RGB consensus correction MAE')):
                ax.set_ylabel(title);ax.invert_xaxis();ax.legend()
            axes[2].set_xlabel('Actual scheduler timestep');fig.suptitle(b.upper()+' synchronized residual correction')
            for ext in ('png','pdf'):fig.savefig(out/(b+'-G-diagnostics.'+ext),dpi=140)
            plt.close(fig)
            for key in ('pre_fusion_rgb_overlap_mae_mean','pre_fusion_rgb_overlap_mae_max','consensus_correction_mae_mean','local_vae_residual_mae_mean','local_vae_residual_mae_max','fused_residual_mae_mean','fused_residual_std','residual_correction_magnitude'):
                row['G_'+key+'_mean']=float(np.mean([s['state_statistics'][key] for s in gs]))
        rows.append(row)
    name='single-summary' if args.single_only else 'summary'
    (out/(name+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
    with (out/(name+'.csv')).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
