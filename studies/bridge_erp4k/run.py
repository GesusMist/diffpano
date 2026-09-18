"""Reuse the exact completed run loop in a private module with study I/O hooks.

No baseline file/global is modified. Only configuration resolution, saved-camera
verification at the new ERP raster, provenance, and artifact destination differ.
"""
import argparse
import gc
import importlib.util
import time
from dataclasses import replace
from pathlib import Path

import torch
from diffpano.bridge_factorial import make_operator
from diffpano.dense_geometry import contributor_stats
from studies.bridge_erp4k.common import *


def spatial_probe(backend,cameras,c):
    """Full-size standard/LPW DPA-center allocations and coverage; zero model calls."""
    records={}
    for mode in ('standard','lpw'):
        cfg=replace(c,warp=replace(c.warp,mode=mode),fusion=replace(c.fusion,mode='detail_preserving_average',weight_mode='spherediff_center'))
        op=make_operator(cfg);rgb=torch.full((1,3,c.view.height,c.view.width),.2,device=backend.device)
        from diffpano.bridge_factorial import BridgeFactorialPipeline
        p=BridgeFactorialPipeline(backend=backend,cameras=cameras,erp_size=ERP_SIZE,warp_operator=op,backend_name=c.model.pipeline)
        start=time.perf_counter();acc=p._accumulator(1)
        for cam in cameras:p._accumulate(acc,rgb,cam,{})
        result=p._finalize(acc,{})
        if result.erp_rgb.shape!=(1,3,*ERP_SIZE) or not bool(torch.isfinite(result.erp_rgb).all()):raise AssertionError('Invalid target ERP fusion')
        coverage=contributor_stats(result.contributor_count)
        if coverage['coverage_percent']!=100 or coverage['minimum']<1:raise AssertionError('Uncovered target ERP pixels')
        returned=op.erp_to_perspective(result.erp_rgb,cameras[7])
        if returned.shape!=rgb.shape or not bool(torch.isfinite(returned).all()):raise AssertionError('Local raster changed')
        records[mode]=dict(coverage=coverage,ERP_resolution=list(ERP_SIZE),local_RGB_resolution=list(rgb.shape[-2:]),
            FOV=[80,80],constant_error_max=float((returned-rgb).abs().max()),seconds=time.perf_counter()-start)
        del result,returned,acc,p,op,rgb;gc.collect();torch.cuda.empty_cache()
    return records


def adapted_runner(phase):
    spec=importlib.util.spec_from_file_location('_erp4k_unchanged_runner',Path('scripts/bridge_factorial_run.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=ROOT;module.cell_record=lambda name,cell:cell_record(name,cell,allow_preflight=phase=='preflight')
    module.saved_cameras=saved_cameras;module.source_hashes=source_hashes
    original_load=module.load;spatial={}
    def load(name,cell):
        result=original_load(name,cell);manifest,row,c,b,p,conds,provenance=result
        old=read(BASE/'preflight'/(name+'.json'))
        if provenance!=old['provenance']:raise AssertionError('Native shape, camera, noise grid, schedule or conditioning changed from baseline')
        if phase=='preflight':spatial[name]=spatial_probe(b,p.cameras,c)
        return result
    module.load=load
    original_write=module.write_new
    def write(path,value):
        if phase=='preflight':
            old=read(BASE/'preflight'/(value['backend']+'.json'))
            if value['initialization']!=old['initialization']:raise AssertionError('Initial states changed with ERP resolution')
            value.update(clean_RGB_ERP_resolution=list(ERP_SIZE),spatial_probe=spatial[value['backend']],baseline_initialization_identical=True)
        else:
            old=read(BASE/'cells'/value['backend']/value['cell']/'metadata.json')
            if value['initialization']!=old['initialization']:raise AssertionError('Run initial states differ from paired baseline')
            changes=config_diff(old['config'],value['config'])
            if {r['path'] for r in changes}!={'erp.height','erp.width'}:raise AssertionError('Non-ERP scientific setting changed')
            value['audit']['stage_audit'].pop('final_local_overlap_thumbnails',None)
            value['audit']['stage_audit']['embedded_preview_storage']='omitted; numerical diagnostics retained'
            value.update(baseline_output=str(BASE/'cells'/value['backend']/value['cell']),scientific_setting_diff=changes,
                initial_local_states_identical_to_baseline=True,followup_type='selected-cell clean RGB ERP resolution only',
                limitations=['One fixed seed and selected cells; not a new full factorial.','Raw ERP pixel-frequency/seam metrics at different ERP resolutions are not directly comparable.','Local-view metrics retain identical FOV and raster dimensions; selected overlap is not global semantic coherence.'])
        original_write(path,value)
    module.write_new=write
    return module


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','run']);p.add_argument('backend');p.add_argument('cell',nargs='?');a=p.parse_args()
    m=adapted_runner(a.phase)
    if a.phase=='preflight':m.preflight(a.backend)
    else:m.run(a.backend,a.cell)

if __name__=='__main__':main()
