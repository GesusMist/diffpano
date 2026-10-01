"""Real-model oracle is the frozen run_dense loop with a one-interval adapter."""
import gc
import os
import time
import traceback
import torch
from diffpano.bridge_factorial import BridgeFactorialPipeline,make_operator
from studies.tt_cea.runtime import *
from studies.tt_cea.schedule import records,eligible_indices
from studies.tt_cea.time_travel import OriginalNoiseBank,backward_original_noise
from studies.tt_cea.audit import baseline_coefficients
from studies.tt_cea.cameras import routing

class OneIntervalBackend:
    """Only the frozen loop sees one timestep; real endpoint helpers see full schedule."""
    def __init__(self,backend,timestep):
        self.backend=backend;self.timesteps=timestep.reshape(1).clone();self.terminal_states=[]
    def __getattr__(self,name):return getattr(self.backend,name)
    def decode_native_canvas(self,value):
        self.terminal_states.append(value.detach().cpu().clone())
        return self.backend.decode_native_canvas(value)


def counts_check(counts,name,passes,terminal=False):
    e=89*passes
    expected=dict(denoiser=e,initialize=0,encode=0 if name=='pixeldit' else 2*e,
        decode=0 if name=='pixeldit' else e+(89 if terminal else 0))
    assert counts==expected,(counts,expected)


@torch.no_grad()
def run(name):
    require_gate();assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    geometry=read(ROOT/'geometry/coverage.json')
    assert geometry['source_hashes']==study_hashes()
    assert geometry['covers']['old89']['passed'],'Original cover geometry failed'
    out=ROOT/'preflight'/(name+'.json')
    if out.exists():raise FileExistsError(out)
    start=time.perf_counter();result=dict(backend=name,job=os.environ['SLURM_JOB_ID'],source_hashes=study_hashes(),
        interval_passed=False,ea_noise_passed=False,cea_passed_by_cover={'old89':False,'ea89':False},cea_errors={},
        numerical_prompt='ruins; fresh production runtime audits also check the underwater conditioning',calls={},environment=environment())
    try:
        c,b,old,row,table,groups=setup(name,'ruins','T1',covers=['old89','ea89'])
        oldgroup=groups['old89'];p=ExperimentalPipeline(b,oldgroup['cameras'],c)
        provenance,audit=check_runtime(c,b,p,oldgroup,old,'T1');result.update(provenance=provenance,runtime_audit=audit)
        expected=baseline_coefficients(old)
        for actual,saved in zip(records(table),expected):
            for key in ('model_timestep','alpha_high','sigma_high','alpha_low','sigma_low'):assert abs(actual[key]-saved[key])<=2e-7,(key,actual,saved)
        assert len(table)==STEPS[name]
        snapshot=b.timesteps.clone();k=len(eligible_indices(len(table)))
        tb=prepare_schedule(b,c,len(table)+k);result['schedules']=dict(original=records(table),matched_budget=records(tb))
        assert len(tb)==len(table)+k
        restored=prepare_schedule(b,c,len(table));assert restored==table and torch.equal(snapshot,b.timesteps)
        initial,init,init_audit=initialize(b,name,oldgroup,old)
        bank=OriginalNoiseBank(initial,b.native_initial_noise_sigma)
        result.update(initialization=init,initialization_audit=init_audit,original_noise_sha256=bank.initial_sha256,
            native_scale=float(b.native_initial_noise_sigma))
        ea=None
        try:
            ea,ea_init,ea_audit=initialize(b,name,groups['ea89'])
            repeat,repeat_record,repeat_audit=initialize(b,name,groups['ea89'],execution_order=list(reversed(range(89))))
            assert repeat_record['initial_local_sha256']==ea_init['initial_local_sha256'];del repeat
            result.update(ea_noise_passed=True,ea_initialization=ea_init,ea_initialization_audits=[ea_audit,repeat_audit],
                ea_routing=routing(groups['ea89']['cameras'],load_directional_prompts(c.prompt.path)))
        except Exception:result['ea_noise_error']=traceback.format_exc()
        first=next(t for t in table if t.eligible);high=initial
        with count_calls(b,name) as calls:
            for interval in table[:first.k]:
                high,summary=p.advance_interval(high,oldgroup['conditions'],interval,pass_kind='preflight_prefix')
                print('PREFLIGHT PREFIX',name,interval.k,flush=True)
        counts_check(calls,name,first.k);result['calls']['prefix']=dict(calls)
        # Independent frozen loop, not another invocation of the experimental helper.
        proxy=OneIntervalBackend(b,b.timesteps[first.k])
        ref=BridgeFactorialPipeline(backend=proxy,cameras=oldgroup['cameras'],erp_size=(2048,4096),warp_operator=make_operator(c),backend_name=name)
        p.canvas.clear_cache()
        with count_calls(b,name) as calls:
            reference=ref.run_dense([v.clone() for v in high],oldgroup['conditions'])
        counts_check(calls,name,1,terminal=True);result['calls']['frozen_interval_including_terminal']=dict(calls)
        reference_states=proxy.terminal_states
        del reference,ref,proxy;gc.collect();torch.cuda.empty_cache()
        with count_calls(b,name) as calls:
            low,summary=p.advance_interval([v.clone() for v in high],oldgroup['conditions'],first,pass_kind='preflight_equivalence')
        counts_check(calls,name,1);result['calls']['experimental_interval']=dict(calls)
        maxima=[];means=[]
        for left,right in zip(reference_states,low):
            torch.testing.assert_close(left,right,atol=2e-6,rtol=2e-6)
            maxima.append(float((left-right).abs().max()));means.append(float((left-right).abs().mean()))
        result['equivalence']=dict(passed=True,absolute_tolerance=2e-6,relative_tolerance=2e-6,
            native_state_max_abs_error=max(maxima),native_state_mean_abs_error=sum(means)/89,interval=records([first])[0],summary=summary,
            oracle='Unmodified BridgeFactorialPipeline.run_dense; adapter exposes one timestep and captures terminal inputs')
        del reference_states,high
        with count_calls(b,name,forbid=True) as calls:
            rng=torch.random.get_rng_state().clone()
            revisited=[backward_original_noise(v,bank[i].to(v),alpha_low=first.alpha_low,sigma_low=first.sigma_low,
                alpha_high=first.alpha_high,sigma_high=first.sigma_high) for i,v in enumerate(low)]
            assert torch.equal(rng,torch.random.get_rng_state())
        assert calls==dict(denoiser=0,encode=0,decode=0,initialize=0);result['calls']['backward']=dict(calls)
        result['backward']=dict(low_sha256=states_digest(low),revisited_sha256=states_digest(revisited),
            mean_abs_move=sum(float((a-v).abs().mean()) for a,v in zip(low,revisited))/89)
        with count_calls(b,name) as calls:
            replay,replay_summary=p.advance_interval(revisited,oldgroup['conditions'],first,pass_kind='preflight_replay')
        counts_check(calls,name,1);result['calls']['replay']=dict(calls)
        result['replay_summary']=replay_summary;assert bank.verify()==bank.initial_sha256
        assert torch.equal(snapshot,b.timesteps)
        result['interval_passed']=True
        del low,revisited,replay,bank;p.canvas.clear_cache();del p;gc.collect();torch.cuda.empty_cache()
        # Each cover has a separate feature gate; failures do not mask valid TT.
        geometry=read(ROOT/'geometry/coverage.json')
        for cover,states in [('old89',initial),('ea89',ea)]:
            if states is None or not geometry['covers'][cover]['passed']:
                result['cea_errors'][cover]='Required initialization or geometry gate failed';continue
            try:
                op=ExperimentalPipeline(b,groups[cover]['cameras'],c,projection='cea')
                with count_calls(b,name) as calls:
                    nxt,summary=op.advance_interval(states,groups[cover]['conditions'],table[0],pass_kind='preflight_cea')
                counts_check(calls,name,1);result['calls']['cea_'+cover]=dict(calls)
                result.setdefault('cea_summaries',{})[cover]=summary
                result['cea_passed_by_cover'][cover]=True;del nxt
            except Exception:result['cea_errors'][cover]=traceback.format_exc()
            finally:
                if 'op' in locals():op.canvas.clear_cache();del op
                gc.collect();torch.cuda.empty_cache()
        result['original_schedule_unchanged']=torch.equal(snapshot,b.timesteps)
        assert result['original_schedule_unchanged']
    except Exception:
        result['error']=traceback.format_exc();print(result['error'],flush=True)
    result['seconds']=time.perf_counter()-start
    result['total_recorded_calls']={key:sum(stage.get(key,0) for stage in result['calls'].values()) for key in ('denoiser','encode','decode','initialize')}
    result['peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
    atomic(out,historical_scheduler_json(result))
    print('BACKEND PREFLIGHT',name,'interval',result['interval_passed'],'CEA',result['cea_passed_by_cover'],flush=True)
    if not result['interval_passed']:raise SystemExit(1)
