"""Real-backend bridge preflights and guarded final factorial cells."""
import argparse
import hashlib
import os
import platform
import resource
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
from diffpano.bridge_factorial import BridgeFactorialPipeline, make_operator, saved_cameras, angular_geometry, digest
from diffpano.consensus_audit import camera_slots, image_metrics, view_metrics, milestones
from diffpano.dense_geometry import contributor_stats
from diffpano.erp_noise_initialization import ERPNoiseConfig,initialize_erp_noise,native_cameras,primary_noise_size,NearestERPIndexProjector,map_statistics
from diffpano.factorial_diagnostics import camera_boundary_metric
from diffpano.initialization import set_random_seed
from diffpano.noise_v_diagnostics import NoiseVAudit
from diffpano.pipelines import build_view_denoiser
from diffpano.trajectory import conditioning_digest
from diffpano.vae_residual import recover_identity,residual_metrics
from diffpano.diagnostics import tensor_to_pil
from scripts.generate import _configure_denoiser
from scripts.dense_erp_experiment import prepare,schedule_metadata,canonical_schedule,seam_metric
from scripts.bridge_factorial_common import ROOT,cell_record,read,write_new,source_hashes,sha


def load(name,cell):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('GPU Slurm allocation required')
    manifest,row,c=cell_record(name,cell)
    set_random_seed(0);backend=build_view_denoiser(c);_configure_denoiser(c,backend)
    cams=saved_cameras(c.view,(c.erp.height,c.erp.width));pipe=BridgeFactorialPipeline(backend=backend,cameras=cams,erp_size=(c.erp.height,c.erp.width),warp_operator=make_operator(c),backend_name=name)
    conditionings,slots=prepare(c,backend,pipe)
    if slots!=manifest['models'][name]['directional_prompt_indices']:raise AssertionError('Prompt routing changed')
    provenance=dict(model_checkpoint=getattr(backend,'checkpoint_path',None) or c.model.path or c.model.id,
        model_revision=getattr(backend,'official_commit',None) or c.model.revision,
        prepared_schedule=canonical_schedule(schedule_metadata(backend)),
        conditioning_sha256=hashlib.sha256(''.join(conditioning_digest(v) for v in conditionings).encode()).hexdigest(),
        prompt_indices=slots,camera_sha256=pipe.camera_sha256,camera_geometry_sha256=digest(angular_geometry(cams)),
        native_channels=backend.native_channels,local_native_resolution=list(backend.native_spatial_shape_for_rgb(c.view.height,c.view.width)),
        local_RGB_resolution=[c.view.height,c.view.width],noise_grid=list(primary_noise_size(native_cameras(backend,cams))),
        native_scale=float(backend.native_initial_noise_sigma),bridge_mode=pipe.bridge_mode,
        native_arithmetic_dtype='torch.float32',vae_dtype=str(backend.pipeline.vae.dtype) if name!='pixeldit' else None)
    if provenance['camera_geometry_sha256']!=manifest['camera_geometry_sha256']:raise AssertionError('Angular geometry changed')
    return manifest,row,c,backend,pipe,conditionings,provenance


def environment():
    return dict(job=os.environ['SLURM_JOB_ID'],node=platform.node(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version())


@torch.no_grad()
def preflight(name):
    output=ROOT/'preflight'/(name+'.json')
    if output.exists():raise FileExistsError(output)
    started=time.perf_counter();manifest,row,c,b,p,conditions,provenance=load(name,'A0B0C0D0')
    identity=[]
    if name!='pixeldit':
        for amplitude in (1.,3.):
            shape=(1,b.native_channels,*provenance['local_native_resolution'])
            z=torch.randn(shape,generator=torch.Generator().manual_seed(19),dtype=torch.float32).to(b.device)*amplitude
            rgb=b.decode_clean(z);rt=b.encode_clean(rgb);rt_again=b.encode_clean(rgb)
            if not torch.equal(rt,rt_again):raise AssertionError('VAE posterior-mode encode is not deterministic')
            recovered=recover_identity(z,rt)
            bound=8*torch.finfo(z.dtype).eps*(z.abs()+rt.abs()+1)
            error=(recovered-z).abs()
            identity.append(dict(amplitude=amplitude,**residual_metrics(z,rt,recovered),dtype=str(z.dtype),vae_dtype=str(b.pipeline.vae.dtype),
                error_max=float(error.max()),bound_max=float(bound.max()),max_error_over_element_bound=float((error/bound).max()),
                bound_formula='8*eps(native_arithmetic_dtype)*(abs(z0)+abs(E(D(z0)))+1)',deterministic_encoding=True))
            del z,rgb,rt,rt_again,recovered,bound,error
    noise={}
    for a in (0,1):
        states,record=initialize_erp_noise(b,p.cameras,ERPNoiseConfig('V-shared-erp' if a else 'V-independent-erp',*provenance['noise_grid'],0))
        noise[str(a)]=record;del states
    for key in ('map_sha256','first_camera_sha256','source_shape','native_local_shapes','scaling'):
        if noise['0'][key]!=noise['1'][key]:raise AssertionError('A0/A1 mismatch: '+key)
    projector=NearestERPIndexProjector(*provenance['noise_grid']);stats=[]
    for i,cam in enumerate(native_cameras(b,p.cameras)):stats.append(dict(slot=i,**map_statistics(projector.index_map(cam))))
    del projector
    aggregate={key:sum(r[key] for r in stats)/len(stats) for key in stats[0] if key!='slot'}
    coverage=contributor_stats(p.precompute_geometry(1))
    if getattr(b,'guided_prediction_count',0)!=0:raise AssertionError('Preflight must not denoise')
    result=dict(passed=True,backend=name,provenance=provenance,identity_oracle=identity,bridge_mode=p.bridge_mode,
        initialization=noise,noise_map_statistics=stats,noise_map_aggregate=aggregate,coverage=coverage,
        guided_predictions=0,environment=environment(),seconds=time.perf_counter()-started,source_hashes=source_hashes())
    write_new(output,result);print('PREFLIGHT PASSED',name,provenance,flush=True)


def metric_summary(final_metrics,milestones_record,audit,wrap):
    views=[v['full'] for v in final_metrics['views']]
    avg=lambda key:sum(key(v) for v in views)/len(views)
    pair=audit['aligned_local_overlap'][-1]
    last=milestones_record[-1]
    return dict(contrast=avg(lambda v:v['std']),hf1=avg(lambda v:v['hf']['1']['rms']),hf1_normalized=avg(lambda v:v['hf']['1']['normalized']),
        out_of_range=avg(lambda v:v['out_of_range_fraction']),view_to_consensus=last['overlap_view_to_consensus_mean'],
        aligned_pair_mae=pair['mae'],aligned_pair_normalized=pair['normalized_mae'],current_state_error=last['current_state_error_max'],
        wrap_gradient=wrap['boundary_gradient'],wrap_ratio=wrap['ratio'])


@torch.no_grad()
def run(name,cell):
    folder=ROOT/'cells'/name/cell
    if folder.exists():raise FileExistsError('Refusing to overwrite '+str(folder))
    # All real bridge/native-shape preflights must pass before any matrix cell.
    gates={n:read(ROOT/'preflight'/(n+'.json')) for n in ('sd2','sana','flux','sd35','pixeldit')}
    if not all(v['passed'] and v['source_hashes']==source_hashes() for v in gates.values()):raise AssertionError('All five current real-backend gates required')
    started=time.perf_counter();manifest,row,c,b,p,conditions,provenance=load(name,cell)
    if provenance!=gates[name]['provenance']:raise AssertionError('Cell settings/conditioning differ from frozen backend preflight')
    a=str(row['A']);init_start=time.perf_counter()
    states,init=initialize_erp_noise(b,p.cameras,ERPNoiseConfig('V-shared-erp' if row['A'] else 'V-independent-erp',*provenance['noise_grid'],0))
    init_seconds=time.perf_counter()-init_start
    if init!=gates[name]['initialization'][a]:raise AssertionError('Initializer changed between cells')
    module=b.model if name=='pixeldit' else (b.pipeline.unet if name=='sd2' else b.pipeline.transformer)
    forward_count=[0];original_forward=module.forward
    def counted(*args,**kwargs):forward_count[0]+=1;return original_forward(*args,**kwargs)
    module.forward=counted
    vae_counts={'decode':0,'encode':0};original_encode=original_decode=None
    if name!='pixeldit':
        original_encode=b.pipeline.vae.encode;original_decode=b.pipeline.vae.decode
        def encoded(*args,**kwargs):vae_counts['encode']+=1;return original_encode(*args,**kwargs)
        def decoded(*args,**kwargs):vae_counts['decode']+=1;return original_decode(*args,**kwargs)
        b.pipeline.vae.encode=encoded;b.pipeline.vae.decode=decoded
    audit=NoiseVAudit(p.cameras,len(b.timesteps));records=[];selected=milestones(len(b.timesteps))
    def progress(step,total,metrics):
        if step+1 in selected:records.append(dict(step=step+1,**metrics))
        print(name,cell,step+1,'/',total,'current error',metrics['current_state_error_max'],flush=True)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    result=p.run_dense(states,conditions,expected_initial_sha256=init['initial_local_sha256'],stage_audit=audit,progress=progress)
    torch.cuda.synchronize();runtime=time.perf_counter()-start;module.forward=original_forward
    expected=manifest['models'][name]['expected_guided_predictions'];n=len(p.cameras)
    if result.audit['guided_predictions']!=expected or forward_count[0]!=expected:raise AssertionError('Denoiser count changed')
    if name!='pixeldit':
        if vae_counts!={'encode':2*expected,'decode':expected+n}:raise AssertionError('Unexpected VAE bridge call count '+str(vae_counts))
        b.pipeline.vae.encode=original_encode;b.pipeline.vae.decode=original_decode
    final_metrics=dict(erp=image_metrics(result.erp_rgb,spherical=True),views=[dict(slot=i,**view_metrics(p.diagnostic_operator.erp_to_perspective(result.erp_rgb,p.cameras[i]))) for i in camera_slots(p.cameras)])
    seam=camera_boundary_metric(result.erp_rgb,p.cameras)
    image=tensor_to_pil(result.erp_rgb.detach().cpu().clone()[0]);wrap=seam_metric(image)
    stage=result.stage_seconds
    efficiency=dict(model=stage.get('model',0),vae_decode=stage.get('decode',0) if name!='pixeldit' else None,
        local_residual_roundtrip_encode=stage.get('local_residual_encode',0) if name!='pixeldit' else None,
        synchronized_clean_encode=stage.get('synchronized_clean_encode',0) if name!='pixeldit' else None,
        spatial_warp_fusion=sum(stage.get(k,0) for k in ('view_to_erp','erp_to_view','rgb_fusion','rgb_finalize','pyramid_construction','pyramid_level_fusion','erp_reconstruction')),
        note='Stage timers cover denoising-loop operations; total pipeline also includes terminal assembly and diagnostics.')
    result.audit['stage_audit']['note']='Matched local predicted-clean RGB diagnostics at four milestones; reused model predictions, zero extra diagnostic VAE/denoiser calls.'
    result.audit.update(initialization='one-time '+row['initialization_mode'],vae_bridge=p.bridge_mode,local_residual_never_warped=True,
        bridge_uses_same_decoded_RGB=True,transient_erp_native_field_at_initialization=True,persistent_erp_native_state=False)
    summary=metric_summary(final_metrics,records,result.audit['stage_audit'],wrap)
    summary.update(camera_boundary_gradient=seam['boundary_gradient'],camera_boundary_ratio=seam['ratio'])
    m=dict(study=manifest['study'],backend=name,cell=cell,**{k:row[k] for k in 'ABCD'},pilot=row['pilot'],
        prompt=manifest['prompt'],camera_count=n,FOV_x=80,FOV_y=80,**provenance,config=c.to_dict(),config_sha256=row['resolved_config_sha256'],
        common_settings_sha256=manifest['models'][name]['common_settings_sha256'],initialization=init,
        initialization_map_aggregate=gates[name]['noise_map_aggregate'],initialization_seconds=init_seconds,
        vae_bridge=p.bridge_mode,transition_mode='preserve_current_state',warp_mode=c.warp.mode,reducer_mode=c.fusion.mode,
        spatial_weight_mode=c.fusion.weight_mode,dpa_parameters=dict(alpha=c.fusion.alpha,power=c.fusion.power,epsilon=c.fusion.epsilon),
        reducer_values='current LPW pyramid coefficients' if row['B'] else 'RGB',current_LPW_adaptation=True if row['B'] else None,
        periodic_pyramid_reconstruction=bool(row['B']),clean_RGB_ERP_resolution=[c.erp.height,c.erp.width],
        audit=result.audit,milestones=records,aggregate_metrics=result.metrics,final_metrics=final_metrics,summary=summary,wrap=wrap,camera_boundary_seam=seam,
        efficiency_seconds=efficiency,stage_seconds=stage,runtime_seconds=runtime,total_seconds=time.perf_counter()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        host_max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        actual_transformer_forward_invocations=forward_count[0],vae_calls=vae_counts,environment=environment(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff'])).hexdigest(),source_hashes=source_hashes(),manifest_sha256=sha(ROOT/'manifest.json'),
        limitations=['One fixed seed; descriptive contrasts only.','Final ERP seam is a wrap-boundary diagnostic, not a global scene-coherence score.','Selected aligned local RGB pair is one overlap, not all camera pairs.'])
    folder.mkdir(parents=True,exist_ok=False);image.save(folder/'final_result.png');write_new(folder/'metadata.json',m)
    print('COMPLETED',name,cell,'seconds',runtime,'guided/forwards',expected,forward_count[0],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['preflight','run']);parser.add_argument('backend');parser.add_argument('cell',nargs='?');args=parser.parse_args()
    preflight(args.backend) if args.phase=='preflight' else run(args.backend,args.cell)
