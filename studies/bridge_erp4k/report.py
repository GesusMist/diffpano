"""Paired resolution diagnostics; never infer a full factorial from a subset."""
import csv
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from diffpano.consensus_audit import thumbnail,view_metrics
from diffpano.bridge_factorial import make_operator
from dataclasses import replace
from studies.bridge_erp4k.common import *


def rgb(path):return torch.from_numpy(np.array(Image.open(path).convert('RGB'),copy=True)).permute(2,0,1)[None].float()/127.5-1


def check_pair(row,m,old,manifest,sources):
    if m['source_hashes']!=sources or m['config']!=row['resolved_config'] or m['config_sha256']!=row['resolved_config_sha256']:raise AssertionError('Follow-up settings/provenance mismatch')
    changes=config_diff(old['config'],m['config'])
    if changes!=row['scientific_setting_diff'] or {r['path'] for r in changes}!={'erp.height','erp.width'}:raise AssertionError('Non-ERP change')
    for k in ('prepared_schedule','conditioning_sha256','prompt_indices','prompt','camera_sha256','camera_geometry_sha256','native_channels','local_native_resolution','local_RGB_resolution','noise_grid','native_scale','bridge_mode','initialization','vae_bridge','transition_mode','warp_mode','reducer_mode','spatial_weight_mode','dpa_parameters','vae_calls'):
        if m[k]!=old[k]:raise AssertionError('Paired invariant changed: '+k)
    if m['camera_geometry_sha256']!=manifest['camera_geometry_sha256'] or m['clean_RGB_ERP_resolution']!=list(ERP_SIZE):raise AssertionError('Wrong geometry/ERP')
    calls=manifest['models'][row['backend']]['expected_guided_predictions']
    if m['actual_transformer_forward_invocations']!=calls or m['audit']['guided_predictions']!=calls:raise AssertionError('Model count changed')
    if not m['audit']['synchronous'] or m['audit']['transition']!='preserve_current_state' or not m['audit']['local_residual_never_warped']:raise AssertionError('Trajectory changed')
    if sorted(p.name for p in Path(row['output']).iterdir())!=['final_result.png','metadata.json']:raise AssertionError('Artifact budget exceeded')
    if Image.open(Path(row['output'])/'final_result.png').size!=(4096,2048):raise AssertionError('Output dimensions wrong')


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('CPU Slurm allocation required')
    torch.set_num_threads(2);manifest=require_validation();sources=source_hashes();rows=[];png_metrics=[];calls=0
    metric_keys=('contrast','hf1','hf1_normalized','out_of_range','aligned_pair_mae','aligned_pair_normalized','view_to_consensus','current_state_error')
    for name in SELECTED:
        selected=[r for r in manifest['cells'] if r['backend']==name];sheet=Image.new('RGB',(1280,35+200*len(selected)),(245,245,245));draw=ImageDraw.Draw(sheet)
        draw.text((8,7),name+' | baseline LEFT / 2048x4096 ERP RIGHT | same local rasters and 80x80 degree cameras',fill='black')
        for index,row in enumerate(selected):
            folder=Path(row['output']);base=Path(row['baseline_output']);m=read(folder/'metadata.json');old=read(base/'metadata.json');check_pair(row,m,old,manifest,sources)
            calls+=m['actual_transformer_forward_invocations'];entry=dict(backend=name,cell=row['cell'],code=row['code'])
            for k in metric_keys:
                entry['baseline_'+k]=old['summary'][k];entry['erp4k_'+k]=m['summary'][k];entry['delta_'+k]=m['summary'][k]-old['summary'][k]
            for k in ('runtime_seconds','total_seconds','peak_allocated_gib','peak_reserved_gib','host_max_rss_gib'):
                entry['baseline_'+k]=old[k];entry['erp4k_'+k]=m[k]
            rows.append(entry);c=make_config(name,row['cell']);cams=saved_cameras(c.view,ERP_SIZE);op=make_operator(replace(c,warp=replace(c.warp,mode='standard')))
            for col,(label,path,size) in enumerate((('baseline',base,old['clean_RGB_ERP_resolution']),('ERP 2048x4096',folder,list(ERP_SIZE)))):
                image=rgb(path/'final_result.png');x=640*col;y=35+200*index
                draw.text((x+8,y+3),row['cell']+' '+label+' '+str(size),fill='black');draw.text((x+334,y+19),'camera 7',fill='black');draw.text((x+486,y+19),'camera 59',fill='black')
                sheet.paste(thumbnail(image,(320,160)),(x+8,y+35));metrics=[]
                for j,slot in enumerate((7,59)):
                    view=op.erp_to_perspective(image,cams[slot]);sheet.paste(thumbnail(view,(146,146)),(x+334+152*j,y+35));metrics.append(dict(slot=slot,**view_metrics(view)))
                png_metrics.append(dict(backend=name,cell=row['cell'],variant=label,views=metrics,local_resolution=[c.view.height,c.view.width],FOV=[80,80]))
            del op
        sheet.save(ROOT/(name+'-paired.png'));print('PAIRED SHEET',name,flush=True)
    expected=sum(manifest['models'][r['backend']]['expected_guided_predictions'] for r in manifest['cells'])
    if len(rows)!=22 or calls!=expected:raise AssertionError('Incomplete selected matrix')
    old_files=read(ROOT/'preservation.json')['historical_artifacts']
    for p,v in old_files.items():
        if not Path(p).is_file() or Path(p).stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:raise AssertionError('Historical output changed: '+p)
    x=read(ROOT/'execution.json')
    changed=[p for p,h in x['prior_file_hashes'].items() if not Path(p).is_file() or sha(p)!=h]
    if set(changed)-{'docs/NATIVE_CONTROLS_REPORT.md'}:raise AssertionError('Preexisting scientific source/config modified: '+str(changed))
    import hashlib
    if hashlib.sha256(Path('docs/NATIVE_CONTROLS_REPORT.md').read_bytes()[:x['report_prefix_bytes']]).hexdigest()!=x['report_prefix_sha256']:raise AssertionError('Historical report prefix changed')
    with (ROOT/'paired-summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    audit=dict(passed=True,job=os.environ['SLURM_JOB_ID'],expected=22,completed=22,model_calls=calls,only_scientific_config_changes=['erp.height','erp.width'],all_initial_local_states_identical=True,all_angular_geometry_identical=True,historical_artifacts_preserved=len(old_files),prior_scientific_files_preserved=True)
    write_new(ROOT/'paired-summary.json',dict(study=manifest['study'],audit=audit,rows=rows,matched_PNG_view_metrics=png_metrics,
        notes=['Selected subset, not a complete factorial; no new factorial contrasts.','Primary HF/contrast comparisons use identical four diagnostic perspective rasters per backend.','Matched PNG sheet metrics use cameras 7/59 at unchanged local raster/FOV.','Raw ERP pixel-frequency and fixed-pixel seam metrics at different ERP resolutions are not directly comparable.','One prompt and seed; high HF can reflect artifacts, low disagreement can reflect blur.']))
    print('ERP4K FOLLOW-UP AUDIT PASSED',audit,flush=True)

if __name__=='__main__':main()
