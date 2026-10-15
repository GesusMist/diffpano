"""Full-resolution same-source arithmetic plus bounded real FLUX bridge smoke."""
from dataclasses import replace
import gc,time
import torch
from studies.mean_max_old89.common import *
from studies.mean_max_old89.configuration import resolved
from studies.mean_max_old89.validate import result


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    m=freeze_check();gate('cpu.json');gate('weights.json');start=time.perf_counter()
    from diffpano.gradient_fusion import reconstruct,GradientSolveError
    from diffpano.projection import ProjectionCache
    from studies.tt_cea.canvas import CanvasOperator,CanvasSpec
    from studies.gradient_blending.operator import StudyCanvas
    row=m['rows'][0];c,cameras,owners,slots,settings,geometry,old=resolved(row)
    solvers=[];roundoff_checks=[]
    for pattern in ('smooth','textured'):
        base=CanvasOperator(CanvasSpec('erp',2048,4096),c.warp,c.fusion,'cuda',cache=ProjectionCache(max_entries=2,cpu_fallback=False))
        canvas=StudyCanvas(base,settings,cameras,owners);canvas.stage='geometry-smoke'
        acc=canvas.make_accumulator(1)
        # Same-source one-pass mean/max, deterministic fixtures, never cached select.
        y,x=torch.meshgrid(torch.linspace(-1,1,1024,device='cuda'),torch.linspace(-1,1,1024,device='cuda'),indexing='ij')
        for i,camera in enumerate(cameras):
            rgb=torch.stack((.3*torch.sin((i%5+1)*x),.25*torch.cos((i%7+1)*y),.2*torch.sin(x+y+i*.1)))[None]
            if pattern=='textured':
                rgb+=torch.stack((.08*torch.sin(64*x+i*.1),.08*torch.cos(64*y+i*.13),.08*torch.sin(48*(x+y)+i*.17)))[None]
            canvas.accumulate(acc,rgb,camera)
        fused=canvas.finalize(acc);r=canvas.records[-1]
        assert bool((fused.contributor_count>0).all()) and bool((fused.weight_sum>0).all())
        coverage=dict(geometry_sha256=geometry['angular_sha256'],minimum_contributors=int(fused.contributor_count.min()),minimum_weight=float(fused.weight_sum.min()),
            connectivity=r['connectivity'],coverage_resolution=[2048,4096],all_pixels_covered=True,all_supported_edges_connected=True)
        stats=acc['gradient'];mean,support=stats.guidance('poisson_mean');maximum,_=stats.guidance('poisson_max')
        for reference in REFERENCES:
            for alpha in ALPHAS:
                cfg=replace(settings,gradient_mix_alpha=alpha,reference_mode=reference,relative_tolerance=1e-4 if reference=='coarse_color' else 1e-5)
                guidance=tuple((1-alpha)*a+alpha*b for a,b in zip(mean,maximum))
                try:
                    out,record=reconstruct(canvas.last_reference,guidance,support,cfg,connectivity=canvas.connectivity)
                except GradientSolveError as error:
                    # Keep the originally failing smooth fixture. Its very small
                    # RHS exposes final-output FP32 roundoff, also in legacy mean.
                    # Rejection is correct; NEVER accept its output or relax policy.
                    record=error.diagnostics
                    assert pattern=='smooth' and reference=='coarse_color'
                    assert not record['converged'] and record['failure']=='Final FP32 output fails normal equation or reference constraint'
                    assert record['output_normal_residual']>record['stopping_threshold']
                    roundoff_checks.append(dict(pattern=pattern,reference_mode=reference,gradient_mix_alpha=alpha,correctly_rejected=True,diagnostics=record))
                    print('EXPECTED_STRICT_REJECTION',alpha,record,flush=True)
                    continue
                assert record['converged'] and record['gradient_mix_alpha']==alpha and out.dtype==torch.float32 and bool(torch.isfinite(out).all())
                solvers.append(dict(pattern=pattern,**record));print('FULL_RESOLUTION_SOLVER',pattern,reference,alpha,record,flush=True);del out,guidance
        del stats,mean,maximum,support,acc,canvas,base,fused,rgb,x,y;gc.collect();torch.cuda.empty_cache()
    assert len([r for r in solvers if r['pattern']=='textured'])==6
    from studies.mean_max_old89.runtime import load_backend,setup,initial,count_calls,safe
    from unittest.mock import patch
    from diffpano.initialization import load_directional_prompts
    from diffpano.planar_pipeline import _negative_prompt
    from diffpano.pipelines.flux import release_prompt_encoders
    b,table=load_backend(row);banks={};groups={};intervals=[];baseline_hash=None
    with patch('diffpano.pipelines.flux.release_prompt_encoders'):
        for name in PROMPTS:
            sample=next(r for r in m['rows'] if r['prompt']==name);cfg=resolved(sample)[0]
            with effective_prompt(name) as (info,path):lines=load_directional_prompts(str(path))
            negative=_negative_prompt(cfg);banks[(tuple(lines),negative)]=b.prepare_prompt_conditioning(lines,negative)
    release_prompt_encoders(b.pipeline)
    assert b.pipeline.text_encoder is None and b.pipeline.text_encoder_2 is None
    b.prepare_prompt_conditioning=lambda lines,negative='':banks[(tuple(lines),negative)]
    # Setup all 90 rows to verify prompt/schedule/geometry identities. Initializer
    # is exercised once per prompt here and independently for EVERY production row.
    for name in PROMPTS:
        variants=[r for r in m['rows'] if r['prompt']==name];invariant=None
        for row in variants:
            p,conditions,slots,config=setup(row,b,table)
            now=(config['provenance']['conditioning_sha256'],config['geometry']['angular_sha256'],digest(config['provenance']['prepared_schedule']))
            if invariant is None:invariant=now
            assert invariant==now
            integration=name==PROMPTS[0] and row['gradient_mix_alpha']==.25
            if row==variants[0] or integration:
                states,init=initial(b,row,p,slots);h=init['initial_local_sha256']
                if baseline_hash is None:baseline_hash=h
                assert h==baseline_hash
                groups[name+'/old89']=dict(conditioning_sha256=now[0],geometry_sha256=now[1],schedule_sha256=now[2],initial_local_sha256=h,
                    initialization=init,runtime_provenance=config['provenance'],route=config['geometry']['route'])
                if integration:
                    with count_calls(b,'flux') as counts:
                        next_states,details=p.advance_interval(states,conditions,table[0],pass_kind='initial')
                        _,terminal,_=p.terminal(next_states)
                    assert counts==dict(denoiser=89,encode=178,decode=178,initialize=0),counts
                    assert details['min_contributors']>0 and details['current_state_error_max']<1e-4
                    assert all(bool(torch.isfinite(x).all()) for x in next_states) and bool(torch.isfinite(terminal).all())
                    assert len(p.canvas.records)==2 and p.canvas.records[-1]['terminal']
                    for f in p.canvas.records:assert f['gradient_mix_alpha']==row['gradient_mix_alpha'] and f['reference_mode']==row['reference_mode'] and f['converged']
                    intervals.append(dict(case_id=row['case_id'],counts=counts,details=safe(details),fusions=p.canvas.records));del next_states,terminal
                del states
            del p,conditions;gc.collect();torch.cuda.empty_cache()
        print('PROMPT_PREFLIGHT',name,invariant,flush=True)
    assert len(groups)==15 and len(intervals)==2
    result('gpu.json',passed=True,coverage=coverage,solvers=solvers,roundoff_rejections=roundoff_checks,groups=groups,intervals=intervals,
        seconds=time.perf_counter()-start,extra_full_trajectories=0,gpu=torch.cuda.get_device_name(),all_90_setup_identities_checked=True)

if __name__=='__main__':main()
