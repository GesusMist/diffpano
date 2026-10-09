"""Interval extraction derived from frozen dense_consensus.py run_dense, lines 138-263.

The two-pass arithmetic and bridge helpers are reused; diagnostic-only repeated
ERP reprojections are omitted. No historical function or module global is edited.
"""
import hashlib
import time
from dataclasses import asdict
import torch
from diffpano.bridge_factorial import BridgeFactorialPipeline
from diffpano.refinement import NativeRefinement, cutoff, fraction_of
from diffpano.current_state_transition import interpolate_from_current_state
from diffpano.erp_local_consensus import camera_digest
from diffpano.erp_noise_initialization import states_digest
from diffpano.pipelines.base import reset_scheduler_step_state
from diffpano.trajectory import conditioning_digest
from studies.tt_cea.schedule import verify_prediction
from studies.tt_cea.time_travel import OriginalNoiseBank,backward_original_noise
from studies.tt_cea.canvas import CanvasOperator,CanvasSpec

class ExperimentalPipeline:
    def __init__(self,backend,cameras,config,projection='erp',*,size=(2048,4096)):
        self.backend=backend;self.cameras=tuple(cameras);self.camera_sha256=camera_digest(cameras)
        self.refinement_config=getattr(getattr(config,"global_pipeline",None),"refinement",None);self.refinement_tail=None
        self.name=config.model.pipeline;self.pixel=self.name=='pixeldit';self.flow=self.name!='sd2'
        self.canvas=CanvasOperator(CanvasSpec(projection,*size),config.warp,config.fusion,backend.device)
        self.bridge=BridgeFactorialPipeline(backend=backend,cameras=cameras,erp_size=size,
            warp_operator=self.canvas.standard,backend_name=self.name)
        self.bridge_mode=self.bridge.bridge_mode
    def _timed(self,timings,key,fn):
        if self.backend.device.type=='cuda':torch.cuda.synchronize(self.backend.device)
        started=time.perf_counter();v=fn()
        if self.backend.device.type=='cuda':torch.cuda.synchronize(self.backend.device)
        timings[key]=timings.get(key,0.)+time.perf_counter()-started
        return v
    @torch.no_grad()
    def advance_interval(self,states,conditionings,interval,canvas_operator=None,*,pass_kind,diagnostics=None):
        op=canvas_operator or self.canvas;b=self.backend
        if len(states)!=len(self.cameras) or len(conditionings)!=len(states):raise ValueError('One state/conditioning per view required')
        if camera_digest(self.cameras)!=self.camera_sha256:raise AssertionError('Camera mutation')
        if interval.k >= cutoff(len(b.timesteps),self.refinement_config):
            if pass_kind != 'initial':raise ValueError('Independent refinement cannot replay a denoising interval')
            if self.refinement_tail is None:
                self.refinement_tail=NativeRefinement(b,self.cameras,conditionings,interval.k)
            return self.refinement_tail.step(states,interval.k,geometry=self.cameras,conditions=conditionings)
        timings={};acc=op.make_accumulator(states[0].shape[0]);residuals=[None]*len(states)
        scheduler=getattr(getattr(b,'pipeline',None),'scheduler',None)
        before=b.guided_prediction_count if hasattr(b,'guided_prediction_count') else 0
        coefficients=None;rgb_out=0.;rgb_samples=0
        for i,camera in enumerate(self.cameras):
            if scheduler is not None:reset_scheduler_step_state(scheduler)
            x=states[i].to(b.device)
            if x.dtype!=torch.float32:raise ValueError('Native arithmetic must be FP32')
            t=b.timesteps[interval.k]
            if float(t)!=interval.model_timestep:raise AssertionError('Wrong interval timestep')
            calls=getattr(b,'guided_prediction_count',0)
            pair=self._timed(timings,'model',lambda:b.predict_clean_and_endpoint(x.clone(),t,conditionings[i]))
            assert b.guided_prediction_count-calls==1
            verify_prediction(pair,interval)
            coeff=tuple(v.detach().clone() if isinstance(v,torch.Tensor) else v for v in (pair.alpha,pair.sigma,pair.next_alpha,pair.next_sigma))
            if coefficients is None:coefficients=coeff
            else:assert all(float(a)==float(c) for a,c in zip(coefficients,coeff))
            rgb=pair.clean.float() if self.pixel else self._timed(timings,'decode',lambda:b.decode_clean(pair.clean))
            if rgb.shape!=(states[0].shape[0],3,camera.height,camera.width) or not bool(torch.isfinite(rgb).all()):raise ValueError('Invalid clean RGB')
            if not self.pixel:
                residuals[i]=self._timed(timings,'local_residual',lambda:self.bridge._local_clean_residual(pair.clean,rgb,{}))
            if diagnostics is not None:diagnostics('proposal',i,dict(clean=pair.clean,rgb=rgb,residual=residuals[i]))
            rgb_out+=float(((rgb < -1)|(rgb > 1)).float().mean());rgb_samples+=1
            self._timed(timings,'projection_fusion',lambda:op.accumulate(acc,rgb,camera))
            del pair,rgb,x
        assert b.guided_prediction_count-before==len(states)
        fused=self._timed(timings,'fusion_finalize',lambda:op.finalize(acc))
        next_states=[];update_sum=0.;update_max=0.;reconstruction_max=0.
        for i,camera in enumerate(self.cameras):
            x=states[i].to(b.device)
            crop=self._timed(timings,'canvas_to_view',lambda:op.sample_view(fused,camera))
            clean=crop.float() if self.pixel else self._timed(timings,'synchronized_encode',lambda:self.bridge._consensus_native_clean(crop,residuals[i],{}))
            if diagnostics is not None:diagnostics('bridge',i,dict(crop=crop,clean=clean))
            a,s,an,sn=coefficients
            nxt=interpolate_from_current_state(x,clean,a,s,an,sn,flow=self.flow)
            reconstructed=a*clean+s*((x-a*clean)/s) if float(s) else a*clean
            torch.testing.assert_close(reconstructed,x,atol=2e-6,rtol=2e-6)
            reconstruction_max=max(reconstruction_max,float((reconstructed-x).abs().max()))
            delta=(nxt-x).abs();update_sum+=float(delta.mean());update_max=max(update_max,float(delta.max()))
            if not bool(torch.isfinite(nxt).all()):raise ValueError('Nonfinite next native state')
            if diagnostics is not None:diagnostics('next',i,dict(state=nxt))
            next_states.append(nxt.detach().cpu());residuals[i]=None
            del x,crop,clean,nxt,reconstructed,delta
        for key in ('last_model_prediction','last_clean_prediction'):
            if hasattr(b,key):setattr(b,key,None)
        summary=dict(k=interval.k,pass_kind=pass_kind,model_timestep=interval.model_timestep,
            current_state_error_max=reconstruction_max,native_update_mean=update_sum/len(states),native_update_max=update_max,
            proposal_out_of_range_fraction=rgb_out/rgb_samples,min_contributors=int(fused.contributor_count.min()),
            minimum_weight_sum=float(fused.weight_sum.min()),stage_seconds=timings)
        if diagnostics is not None:diagnostics('canvas',None,dict(result=fused))
        return next_states,summary
    @torch.no_grad()
    def terminal(self,states):
        acc=self.canvas.make_accumulator(states[0].shape[0]);timings={}
        for s,camera in zip(states,self.cameras):
            rgb=s.to(self.backend.device).float() if self.pixel else self._timed(timings,'terminal_decode',lambda:self.backend.decode_native_canvas(s.to(self.backend.device)))
            self._timed(timings,'terminal_projection_fusion',lambda:self.canvas.accumulate(acc,rgb,camera))
        final=self.canvas.finalize(acc)
        erp=self._timed(timings,'terminal_export',lambda:self.canvas.export_erp(final,*final.rgb.shape[-2:]))
        return final,erp,timings
    @torch.no_grad()
    def run(self,states,conditionings,intervals,*,time_travel=False,progress=None):
        if time_travel and fraction_of(self.refinement_config):
            raise ValueError('Time travel with independent refinement is not an established combination')
        states=[s.detach().cpu().clone().float() for s in states]
        snapshot=self.backend.timesteps.clone();initial_hash=states_digest(states)
        cond=[conditioning_digest(c) for c in conditionings]
        bank=OriginalNoiseBank(states,self.backend.native_initial_noise_sigma) if time_travel else None
        before=getattr(self.backend,'guided_prediction_count',0);logs=[];boundaries=[];backward_seconds=0.
        for interval in intervals:
            low,summary=self.advance_interval(states,conditionings,interval,pass_kind='initial');logs.append(summary)
            if bank is not None and interval.eligible:
                mark=time.perf_counter();revisited=[];mean=maximum=0.
                selected=interval.k in (next(x.k for x in intervals if x.eligible),max(x.k for x in intervals if x.eligible))
                low_hash=states_digest(low) if selected else None
                for i,x in enumerate(low):
                    moved=backward_original_noise(x,bank[i].to(x),alpha_low=interval.alpha_low,sigma_low=interval.sigma_low,
                        alpha_high=interval.alpha_high,sigma_high=interval.sigma_high)
                    delta=(moved-x).abs();mean+=float(delta.mean());maximum=max(maximum,float(delta.max()));revisited.append(moved)
                elapsed=time.perf_counter()-mark;backward_seconds+=elapsed
                if selected:boundaries.append(dict(k=interval.k,low_sha256=low_hash,revisited_sha256=states_digest(revisited)))
                low,replay=self.advance_interval(revisited,conditionings,interval,pass_kind='replay')
                replay.update(backward_move_mean=mean/len(states),backward_move_max=maximum,backward_seconds=elapsed);logs.append(replay)
                del revisited
            states=low
            if progress:progress(interval.k+1,len(intervals),logs[-1])
        assert torch.equal(snapshot,self.backend.timesteps),'Prepared schedule mutated'
        assert cond==[conditioning_digest(c) for c in conditionings],'Conditioning mutated'
        expected=len(self.cameras)*len(logs);assert self.backend.guided_prediction_count-before==expected
        final,erp,terminal=self.terminal(states)
        return final,erp,dict(diagnostics=logs,initial_local_sha256=initial_hash,
            original_noise_sha256_before=bank.initial_sha256 if bank else None,
            original_noise_sha256_after=bank.verify() if bank else None,replay_boundary_hashes=boundaries,
            guided_predictions=expected,backward_seconds=backward_seconds,terminal_seconds=terminal,
            terminal_state_sha256=states_digest(states),projection=self.canvas.spec.projection,
            terminal_semantics='decode terminal local states then fuse in active RGB canvas; CEA exports once',
            persistent_state='CPU local native noisy states',residuals_local=True)
