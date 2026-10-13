"""Bounded real-resolution geometry/solver/model checks, never a full extra trajectory."""
from dataclasses import replace
import gc,time
import torch
from studies.ournew_comparison.common import *
from studies.ournew_comparison.configuration import resolved
from studies.ournew_comparison.validate import result


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    m=freeze_check();gate('cpu.json');gate('weights.json');start=time.perf_counter()
    from diffpano.gradient_fusion import GradientSettings,reconstruct
    from diffpano.poisson_reference import ConnectivityCache
    from diffpano.projection import ProjectionCache
    from studies.tt_cea.canvas import CanvasOperator,CanvasSpec
    from studies.gradient_blending.operator import StudyCanvas
    coverage={};solver=[]
    for row in (m['rows'][1],m['rows'][2]):
        c,cameras,owners,slots,settings,geometry,old=resolved(row)
        base=CanvasOperator(CanvasSpec('erp',2048,4096),c.warp,c.fusion,'cuda',cache=ProjectionCache(max_entries=2,cpu_fallback=False))
        canvas=StudyCanvas(base,replace(settings,mode='poisson_select',reference_mode='single_pixel'),cameras,owners);canvas.stage='geometry-smoke'
        acc=canvas.make_accumulator(1);rgb=torch.zeros(1,3,1024,1024,device='cuda')
        for camera in cameras:canvas.accumulate(acc,rgb,camera)
        fused=canvas.finalize(acc);r=canvas.records[-1]
        assert bool((fused.contributor_count>0).all()) and bool((fused.weight_sum>0).all()) and r['converged']
        a=r['anchor'];expected=next(x['semantic']['anchor']['location'] for x in m['rows'] if x['reference_mode']=='single_pixel') if row['layout']=='uniform89' else None
        if expected is not None:
            for k in ('erp_y','erp_x','location_selection_camera_id'):assert a[k]==expected[k],(k,a,expected)
        coverage[row['layout']]=dict(geometry_sha256=geometry['angular_sha256'],minimum_contributors=int(fused.contributor_count.min()),
            minimum_weight=float(fused.weight_sum.min()),connectivity=r['connectivity'],anchor=a,
            coverage_resolution=[2048,4096],all_pixels_covered=True,all_supported_edges_connected=True)
        print('FULL_RESOLUTION_GEOMETRY',row['layout'],coverage[row['layout']],flush=True)
        del acc,fused,canvas,base,rgb;gc.collect();torch.cuda.empty_cache()
    path=ROOT/'outputs/10.6gradient-blending/seed0-v1/cases/ruins/rgb/intermediate/step-10/sufficient-statistics.pt'
    data=torch.load(path,map_location='cuda',weights_only=False);ref=data['reference'].float();g=tuple(x.float() for x in data['selected']);support=tuple(x.bool() for x in data['support'])
    assert ref.shape==(1,3,2048,4096);cache=ConnectivityCache();cache.check(support)
    a=coverage['uniform89']['anchor'];anchor=dict(a,color=ref[...,a['erp_y'],a['erp_x']].clone())
    for mode in ('screened','global_mean','single_pixel','coarse_color'):
        cfg=GradientSettings('poisson_select',reference_mode=mode,relative_tolerance=1e-4 if mode=='coarse_color' else 1e-5)
        torch.cuda.reset_peak_memory_stats();out,record=reconstruct(ref,g,support,cfg,anchor=anchor,connectivity=cache)
        assert bool(torch.isfinite(out).all()) and record['converged'];record['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        solver.append(record);print('SOLVER_SMOKE',mode,record,flush=True);del out
    del data,ref,g,support,cache,anchor;gc.collect();torch.cuda.empty_cache()
    from studies.ournew_comparison.runtime import load_backend,setup,initial,count_calls,safe
    b,table=load_backend(m['rows'][1]);groups={};initial_hashes={};intervals=[]
    # Bounded preflight encodes each full directional bank before the normal
    # release hook. Production still prepares one bank and releases immediately.
    from unittest.mock import patch
    from diffpano.initialization import load_directional_prompts
    from diffpano.planar_pipeline import _negative_prompt
    from diffpano.pipelines.flux import release_prompt_encoders
    banks={}
    with patch('diffpano.pipelines.flux.release_prompt_encoders'):
        for name in PROMPTS:
            row=next(r for r in m['rows'] if r['prompt']==name and r['variant']=='diffpano_old89_rgb')
            cfg=resolved(row)[0];lines=load_directional_prompts(str(prompt(name)[1]));negative=_negative_prompt(cfg)
            banks[(tuple(lines),negative)]=b.prepare_prompt_conditioning(lines,negative)
    release_prompt_encoders(b.pipeline)
    assert b.pipeline.text_encoder is None and b.pipeline.text_encoder_2 is None
    def cached_bank(lines,negative=''):
        return banks[(tuple(lines),negative)]
    b.prepare_prompt_conditioning=cached_bank
    for row in m['rows']:
        if row['variant'] not in ('diffpano_old89_rgb','diffpano_uniform89_rgb'):continue
        p,conditions,slots,config=setup(row,b,table)
        states,init=initial(b,row,p,slots);h=init['initial_local_sha256']
        if row['layout'] in initial_hashes:assert initial_hashes[row['layout']]==h
        initial_hashes[row['layout']]=h
        groups[row['prompt']+'/'+row['layout']]=dict(conditioning_sha256=config['provenance']['conditioning_sha256'],
            initial_local_sha256=h,geometry_sha256=config['geometry']['angular_sha256'],route=config['geometry']['route'],
            initialization=init,runtime_provenance=config['provenance'],allowed_runtime_differences=config['audit']['differences'])
        if row['prompt']==PROMPTS[0]:
            # One ordinary interval per real bank exercises conditioning, transport and bridge.
            with count_calls(b,'flux') as counts:
                next_states,details=p.advance_interval(states,conditions,table[0],pass_kind='initial')
            assert counts==dict(denoiser=89,encode=178,decode=89,initialize=0)
            assert all(bool(torch.isfinite(x).all()) for x in next_states)
            assert details['min_contributors']>0 and details['current_state_error_max']<1e-4
            intervals.append(dict(layout=row['layout'],counts=counts,details=safe(details)));del next_states
        del p,conditions,states;gc.collect();torch.cuda.empty_cache()
    assert len(groups)==8 and initial_hashes['old89']!=initial_hashes['uniform89']
    result('gpu.json',passed=True,coverage=coverage,solver=solver,groups=groups,initialization_hashes=initial_hashes,
        intervals=intervals,seconds=time.perf_counter()-start,solver_input=dict(path=str(path),sha256=sha(path)),
        extra_full_trajectories=0,gpu=torch.cuda.get_device_name())

if __name__=='__main__':main()
