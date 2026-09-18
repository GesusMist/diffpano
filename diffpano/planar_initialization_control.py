"""R exact-crop oracle: native step averaging vs clean/current-state averaging."""
import torch
from diffpano.current_state_transition import interpolate_from_current_state
from diffpano.native_multidiffusion import NativePlanarFusionAccumulator
from diffpano.planar import build_planar_patch_layout,extract_planar_patch


def average(values,layout):
    first=values[0]
    acc=NativePlanarFusionAccumulator(first.new_zeros(first.shape[0],first.shape[1],layout.canvas_height,layout.canvas_width))
    for value,patch in zip(values,layout.patches):acc.accumulate(value,patch)
    return acc.finalize()


def error(a,b):
    d=(a-b).abs()
    return dict(mae=float(d.mean()),max_abs=float(d.max()),rms=float(d.square().mean().sqrt()))


def overlap_error(states,layout):
    canvas=average(states,layout)
    return max(float((x-extract_planar_patch(canvas,p)).abs().max()) for x,p in zip(states,layout.patches))


@torch.no_grad()
def paired_run(backend,geometry,initial,conditioning,progress=None):
    layout=build_planar_patch_layout(geometry.canvas_height,geometry.canvas_width,geometry.patch_size,geometry.stride)
    global_state=initial.clone();shared=[extract_planar_patch(initial,p).clone() for p in layout.patches]
    records=[];before=backend.guided_prediction_count if hasattr(backend,'guided_prediction_count') else 0
    for index,t in enumerate(backend.timesteps):
        native_inputs=[extract_planar_patch(global_state,p).clone() for p in layout.patches]
        native=[];native_clean=[];shared_clean=[];pairs=[]
        for x in native_inputs:
            proposal,pair=backend.native_step_with_endpoints(x.clone(),t,conditioning)
            native.append(proposal);native_clean.append(pair.clean)
        for x in shared:
            pair=backend.predict_clean_and_endpoint(x.clone(),t,conditioning)
            shared_clean.append(pair.clean);pairs.append(pair)
        native_next=average(native,layout);clean_canvas=average(shared_clean,layout)
        shared_next=[interpolate_from_current_state(x,extract_planar_patch(clean_canvas,p),q.alpha,q.sigma,q.next_alpha,q.next_sigma,flow=True)
                     for x,p,q in zip(shared,layout.patches,pairs)]
        # A same-input algebraic oracle reuses native's predictions, so it costs
        # no extra model evaluations and separates roundoff from model drift.
        nclean=average(native_clean,layout);q=pairs[0]
        oracle=interpolate_from_current_state(global_state,nclean,q.alpha,q.sigma,q.next_alpha,q.next_sigma,flow=True)
        step=dict(step=index+1,timestep=float(t),coefficients=[float(v) for v in (q.alpha,q.sigma,q.next_alpha,q.next_sigma)],
            model_inputs=error(torch.stack(native_inputs),torch.stack(shared)),
            clean_predictions=error(torch.stack(native_clean),torch.stack(shared_clean)),
            next_states=error(native_next,average(shared_next,layout)),
            same_input_algebraic_oracle=error(native_next,oracle),
            shared_overlap_max_abs=overlap_error(shared_next,layout))
        records.append(step)
        # The local update uses float32; this bound covers the few additions and
        # multiplications in two equivalent evaluation orders, not model errors.
        scale=max(float(native_next.abs().max()),1.)
        if step['same_input_algebraic_oracle']['max_abs']>32*torch.finfo(torch.float32).eps*scale:
            raise AssertionError(f'Official first-order coefficient oracle failed at step {index+1}: {step}')
        global_state=native_next;shared=shared_next
        if progress:progress(step)
    calls=backend.guided_prediction_count-before
    if calls!=2*layout.num_patches*len(backend.timesteps):raise AssertionError('Unexpected paired prediction count')
    return global_state,average(shared,layout),records,calls


@torch.no_grad()
def independent_run(backend,geometry,states,conditioning,progress=None):
    layout=build_planar_patch_layout(geometry.canvas_height,geometry.canvas_width,geometry.patch_size,geometry.stride)
    records=[];before=backend.guided_prediction_count
    for index,t in enumerate(backend.timesteps):
        pairs=[backend.predict_clean_and_endpoint(x.clone(),t,conditioning) for x in states]
        clean=average([q.clean for q in pairs],layout)
        states=[interpolate_from_current_state(x,extract_planar_patch(clean,p),q.alpha,q.sigma,q.next_alpha,q.next_sigma,flow=True)
                for x,p,q in zip(states,layout.patches,pairs)]
        records.append(dict(step=index+1,overlap_max_abs=overlap_error(states,layout)))
        if progress:progress(records[-1])
    return average(states,layout),records,backend.guided_prediction_count-before
