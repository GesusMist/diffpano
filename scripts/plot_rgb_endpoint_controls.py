#!/usr/bin/env python3
"""Render paired images, diagnostics, and common final-RGB boundary metrics."""
import argparse
import csv
import json
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load the pure NumPy helper without importing diffpano.__init__, which eagerly
# imports torch. Grace's lightweight matplotlib environment intentionally has no torch.
_spec=importlib.util.spec_from_file_location('diffpano_seam_metrics',Path(__file__).resolve().parents[1]/'diffpano/seams.py')
_seams=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_seams)
boundary_gradient_metrics=_seams.boundary_gradient_metrics


def run(sources,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];full=[]
    for source in sources:
        source=Path(source);pair=json.loads(source.read_text());folder=source.parent
        backend=pair['config']['model']['pipeline']
        assert all(v for k,v in pair['fairness'].items() if isinstance(v,bool))
        assert len(set(pair['model_evaluations'].values()))==1
        names=('native_multidiffusion','implied_endpoint_consensus')
        images=[np.asarray(Image.open(folder/name/'result.png').convert('RGB'))/255. for name in names]
        assert images[0].shape==images[1].shape
        patches=pair['geometry']['rgb']['patches']
        metrics=[boundary_gradient_metrics(rgb,patches) for rgb in images]
        fig,axes=plt.subplots(1,2,figsize=(16,4.7),layout='constrained')
        for ax,rgb,title in zip(axes,images,('Native MultiDiffusion','RGB implied-endpoint consensus')):
            ax.imshow(rgb);ax.set_title(backend.upper()+' — '+title);ax.axis('off')
        for ext in ('png','pdf'):fig.savefig(output/(backend+'-comparison.'+ext),dpi=140)
        plt.close(fig)
        metadata=json.loads((folder/names[1]/'metadata.json').read_text())
        steps=metadata['steps'];stats=[s['state_statistics'] for s in steps]
        ts=[s['scheduler_timestep'] for s in steps]
        keys=['pre_fusion_rgb_overlap_mae_mean','pre_fusion_rgb_overlap_mae_max','consensus_correction_mae_mean']
        if 'clean_native_roundtrip_mae' in stats[0]:keys.append('clean_native_roundtrip_mae')
        fig,axes=plt.subplots(len(keys),1,figsize=(8,2.3*len(keys)),layout='constrained',squeeze=False)
        for ax,key in zip(axes.flat,keys):
            ax.plot(ts,[s[key] for s in stats]);ax.set_ylabel(key.replace('_',' '));ax.invert_xaxis()
        axes[-1,0].set_xlabel('Actual scheduler timestep');fig.suptitle(backend.upper())
        for ext in ('png','pdf'):fig.savefig(output/(backend+'-diagnostics.'+ext),dpi=140)
        plt.close(fig)
        row=dict(backend=backend,job=pair['slurm_job_id'],node=pair['node'],source=str(source))
        for key in keys:
            row[key+'_step_mean']=float(np.mean([s[key] for s in stats]))
            row[key+'_first']=stats[0][key];row[key+'_last']=stats[-1][key]
        for name,metric in zip(names,metrics):
            for key in ('boundary_gradient','nearby_gradient','boundary_excess','boundary_to_nearby_ratio'):
                row[name+'_'+key]=metric[key]
            row[name+'_runtime_seconds']=pair['methods'][name]['runtime_seconds']
            row[name+'_allocated_gib']=pair['methods'][name]['peak_gpu_memory_gib']['allocated_gib']
            row[name+'_reserved_gib']=pair['methods'][name]['peak_gpu_memory_gib']['reserved_gib']
        fused_rgb=np.asarray(Image.open(folder/names[1]/'final_fused_clean.png').convert('RGB'))/255.
        assert fused_rgb.shape==images[1].shape
        fused_metric=boundary_gradient_metrics(fused_rgb,patches)
        row['terminal_vs_fused_clean_display_rgb_mae']=float(np.abs(images[1]-fused_rgb).mean())
        for key in ('boundary_gradient','nearby_gradient','boundary_excess','boundary_to_nearby_ratio'):
            row['final_fused_clean_'+key]=fused_metric[key]
        control_path=folder/'one_patch_roundtrip'/'control.json'
        control=None
        if control_path.exists():
            control=json.loads(control_path.read_text());row['roundtrip_final_rgb_mae']=control['differences']['final_rgb_mae']
            fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained')
            for ax,file,title in zip(axes,('no_roundtrip.png','roundtrip.png'),('Implied, no RGB roundtrip','Implied, RGB roundtrip each step')):
                ax.imshow(Image.open(control_path.parent/file));ax.set_title(title);ax.axis('off')
            fig.suptitle(backend.upper()+' one-patch control')
            for ext in ('png','pdf'):fig.savefig(output/(backend+'-roundtrip.'+ext),dpi=140)
            plt.close(fig)
        rows.append(row);full.append(dict(summary=row,seam_metrics=dict(zip(names,metrics)),final_fused_clean_seam_metrics=fused_metric,control=control))
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (output/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(full,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('sources',nargs='+');parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.sources,args.output)
