"""Numerical pilot audit, complete 2^4 contrasts, and resolution-matched sheets."""
import argparse
import csv
import io
import itertools
import math
import os
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch
from PIL import Image,ImageDraw,ImageOps
from diffpano.bridge_factorial import BACKENDS,factor_cells,saved_cameras,make_config
from diffpano.consensus_audit import thumbnail,view_metrics
from diffpano.config import WarpConfig,FusionConfig
from diffpano.factorial_diagnostics import factorial_contrasts
from diffpano.projection import ProjectionCache
from diffpano.warp import StandardWarpOperator
from scripts.bridge_factorial_common import ROOT,read,require_validation,source_hashes,write_new,sha


def finite(value):
    if isinstance(value,float) and not math.isfinite(value):raise AssertionError('Nonfinite metric')
    if isinstance(value,dict):
        for v in value.values():finite(v)
    if isinstance(value,list):
        for v in value:finite(v)


def audit(manifest,pilot):
    rows=[];metadata={};initials={};common={};shapes={}
    expected=[r for r in manifest['cells'] if r['pilot'] or not pilot]
    for r in expected:
        folder=Path(r['output']);m=read(folder/'metadata.json');key=(r['backend'],r['cell']);metadata[key]=m
        if sorted(p.name for p in folder.iterdir())!=['final_result.png','metadata.json']:raise AssertionError('Per-cell artifact budget exceeded')
        if m['source_hashes']!=source_hashes() or m['config']!=r['resolved_config'] or m['config_sha256']!=r['resolved_config_sha256']:raise AssertionError('Cell provenance changed')
        if any(m[k]!=r[k] for k in 'ABCD'):raise AssertionError('Factor label mismatch')
        if m['camera_geometry_sha256']!=manifest['camera_geometry_sha256'] or m['prompt']['sha256']!=manifest['prompt']['sha256']:raise AssertionError('Geometry/prompt mismatch')
        if m['transition_mode']!='preserve_current_state' or m['audit']['transition']!='preserve_current_state':raise AssertionError('Wrong transition')
        if not m['audit']['synchronous'] or not m['audit']['local_residual_never_warped']:raise AssertionError('Nonlocal or asynchronous bridge')
        calls=manifest['models'][r['backend']]['expected_guided_predictions']
        if m['audit']['guided_predictions']!=m['actual_transformer_forward_invocations'] or m['audit']['guided_predictions']!=calls:raise AssertionError('Model count changed')
        if m['vae_bridge']!=r['bridge_mode']:raise AssertionError('Bridge changed')
        init=m['initialization']
        if init['source_draws']!=(1 if r['A'] else 89) or not init['source_released'] or init['vae_calls'] or init['persistent_erp_native_state'] or init['fixed_noise_renoising'] or init['empirical_normalization']:raise AssertionError('Noise semantics changed')
        ik=(r['backend'],r['A']);initials.setdefault(ik,init)
        if initials[ik]!=init:raise AssertionError('Initialization differs within A level')
        values={k:m[k] for k in ('model_checkpoint','model_revision','prepared_schedule','conditioning_sha256','prompt_indices','camera_sha256','camera_geometry_sha256','native_channels','local_native_resolution','local_RGB_resolution','noise_grid','native_scale','bridge_mode')}
        common.setdefault(r['backend'],values)
        if common[r['backend']]!=values:raise AssertionError('Nonfactor settings changed across cells')
        finite(m['summary']);finite(m['milestones']);finite(m['audit']['stage_audit']['aligned_local_overlap'])
        row=dict(backend=r['backend'],cell=r['cell'],**{k:r[k] for k in 'ABCD'},**m['summary'],runtime_seconds=m['runtime_seconds'],peak_allocated_gib=m['peak_allocated_gib'],host_max_rss_gib=m['host_max_rss_gib'])
        row.update({k+'_seconds':v for k,v in m['efficiency_seconds'].items() if isinstance(v,(float,int))})
        rows.append(row)
    for name in BACKENDS:
        left=initials[(name,0)];right=initials[(name,1)]
        for k in ('map_sha256','source_shape','native_local_shapes','scaling','first_camera_sha256'):
            if left[k]!=right[k]:raise AssertionError('A0/A1 matched sampling failed')
    result=dict(passed=True,cells=len(rows),expected=20 if pilot else 80,all_sources_match=True,all_scientific_settings_match=True,geometry_hash=manifest['camera_geometry_sha256'],model_calls=sum(m['actual_transformer_forward_invocations'] for m in metadata.values()),job=os.environ['SLURM_JOB_ID'])
    if pilot:write_new(ROOT/'pilot-audit.json',result)
    return rows,metadata,result


def load_rgb(path):
    return torch.from_numpy(np.array(Image.open(path).convert('RGB'),copy=True)).permute(2,0,1)[None].float()/127.5-1


def operator():return StandardWarpOperator(WarpConfig(mode='standard'),FusionConfig(mode='weighted_average',weight_mode='uniform'),ProjectionCache(max_entries=1))


def contact(name):
    c=make_config(name,'A0B0C0D0');cams=saved_cameras(c.view,(c.erp.height,c.erp.width));op=operator()
    cw,ch=328,356;sheet=Image.new('RGB',(cw*4,ch*4),(245,245,245));draw=ImageDraw.Draw(sheet)
    for i,cell in enumerate(factor_cells()):
        rgb=load_rgb(ROOT/'cells'/name/cell/'final_result.png');x=(i%4)*cw;y=(i//4)*ch
        draw.text((x+4,y+3),name+' '+cell,fill='black')
        sheet.paste(thumbnail(rgb,(320,160)),(x+4,y+20))
        for j,slot in enumerate((7,59)):
            panel=thumbnail(op.erp_to_perspective(rgb,cams[slot]),(158,158));sheet.paste(panel,(x+4+j*162,y+194))
            draw.text((x+4+j*162,y+181),'camera '+str(slot),fill='black')
    sheet.save(ROOT/(name+'-factorial.png'))


def choose_representatives(rows):
    # Descriptive display choices after full contrasts; no parameter tuning.
    eligible=[r for r in rows if r['out_of_range']<=.01]
    if not eligible:eligible=rows
    detail=max(eligible,key=lambda r:(r['hf1_normalized'],r['cell']))
    enough_contrast=[r for r in rows if r['contrast']>=median(v['contrast'] for v in rows)]
    agreement=min(enough_contrast,key=lambda r:(r['aligned_pair_normalized'],r['cell']))
    return dict(detail=detail['cell'],agreement=agreement['cell'])


def reference_sheet(selection):
    sheet=Image.new('RGB',(1050,6*286),(245,245,245));draw=ImageDraw.Draw(sheet);result={};index=0
    for name in ('flux','sana'):
        c=make_config(name,'A0B0C0D0');cams=saved_cameras(c.view,(c.erp.height,c.erp.width));op=operator();result[name]={}
        methods=[('detail '+selection[name]['detail'],ROOT/'cells'/name/selection[name]['detail']),('agreement '+selection[name]['agreement'],ROOT/'cells'/name/selection[name]['agreement']),('original SphereDiff',ROOT/'references'/name)]
        for label,folder in methods:
            rgb=load_rgb(folder/'final_result.png');y=index*286;index+=1
            draw.text((5,y+5),name+' '+label,fill='black');sheet.paste(thumbnail(rgb,(380,230)),(5,y+30));values=[]
            for j,slot in enumerate((7,59)):
                view=op.erp_to_perspective(rgb,cams[slot]);values.append(dict(slot=slot,**view_metrics(view)))
                draw.text((395+j*324,y+5),'camera '+str(slot)+'; 1024px / 80deg',fill='black')
                sheet.paste(thumbnail(view,(270,250)),(395+j*324,y+25))
            result[name][label]=dict(views=values,metric_domain='saved PNGs converted to RGB [-1,1], then projected at identical 1024x1024 perspective raster / 80-degree FOV; native ERP outputs unchanged')
    sheet.save(ROOT/'spherediff-comparison.png')
    return result


def main(pilot):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('CPU compute allocation required')
    torch.set_num_threads(2);manifest=require_validation();rows,metadata,audit_result=audit(manifest,pilot)
    if pilot:print('PILOT AUDIT PASSED',audit_result,flush=True);return
    for name in ('flux','sana'):
        folder=ROOT/'references'/name;m=read(folder/'metadata.json')
        if not m['external_reference'] or m['in_factorial_contrasts'] or m['source_commit']!=manifest['spherediff_source_commit'] or m['spec']!=manifest['references'][name]:raise AssertionError('Original reference provenance mismatch')
        if sorted(p.name for p in folder.iterdir())!=['final_result.png','metadata.json']:raise AssertionError('Reference artifact budget exceeded')
    metrics=[k for k,v in rows[0].items() if isinstance(v,(int,float)) and k not in 'ABCD']
    effects={};selection={}
    for name in BACKENDS:
        current=[r for r in rows if r['backend']==name]
        # PixelDiT VAE times are absent/not applicable; contrast common scalar metrics.
        keys=[k for k in metrics if all(k in r for r in current)]
        effects[name]=factorial_contrasts(current,keys);selection[name]=choose_representatives(current);contact(name)
    refs=reference_sheet(selection)
    with (ROOT/'factor-summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)
    summary=dict(study=manifest['study'],audit=audit_result,rows=rows,contrasts=effects,contrast_convention=manifest['factor_contrast_convention'],
        reference_selection=selection,selection_criterion='Detail: maximum HF1/std among cells with <=1% out-of-range fraction (all cells if none qualify). Agreement: minimum normalized aligned-pair MAE among cells at or above backend-median contrast. Descriptive tradeoff representatives, not universal best.',
        references_excluded_from_contrasts=True,reference_comparison_metrics=refs,
        one_seed=True,p_values_reported=False,notes=['Higher HF can reflect artifacts; low disagreement can reflect blur.','Raw factorial metrics and common-resolution PNG reference metrics have explicitly distinct domains.'])
    write_new(ROOT/'factor-summary.json',summary)
    previous=read(ROOT/'provenance.json')['historical_artifacts'];changed=[]
    for p,record in previous.items():
        if not Path(p).is_file() or Path(p).stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:changed.append(p)
    if changed:raise AssertionError('Historical artifacts changed: '+str(changed))
    print('COMPLETE FACTORIAL REPORT',audit_result,'historical artifacts preserved',len(previous),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');main(p.parse_args().pilot)
