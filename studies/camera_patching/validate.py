"""Model-free regressions, full geometry gates and read-only old89 audit."""
import os
import sys
import subprocess
from datetime import datetime,timezone
from .common import *
from .cameras import angular_hash,routing,cover
from .coverage import analyze_camera_cover


def baseline_audit():
    from studies.all_prompts.common import rows as old_rows
    from studies.all_prompts.audit import reusable_diffpano
    from studies.all_prompts.summary import reused_success,logged_success
    assert historical_fingerprint()==HISTORICAL_FINGERPRINT,'Historical generation source changed'
    required=[r for r in old_rows() if r['method']=='diffpano' and r['prompt'] in PROMPTS]
    records=[]
    for row in required:
        assert image_valid(row['output'],True),('Missing/corrupt old89 entry; exact recovery needed',row)
        if row['prompt'] in ('ruins','underwater'):
            proof=reusable_diffpano(row)
            assert proof is not None and sha(row['output'])==proof['image_sha256'] and job_success(proof['job'])
            assert reused_success(row)
            job=proof['job']
        else:
            job='19898963_'+str(row['index'])
            assert job_success(job) and logged_success(job,row)
            lines=(REPO/'logs'/('allp21-run.'+job+'.out')).read_text().splitlines()
            def event(name):return next(json.loads(x[len(name)+1:]) for x in lines if x.startswith(name+' '))
            start=event('WORKER_START');runtime=event('RUNTIME_PROVENANCE');prompt=event('PROMPT');done=event('GENERATION_VALIDATED')
            assert start['source_fingerprint']==HISTORICAL_FINGERPRINT
            assert runtime['row']==row and runtime['old89'] and not runtime['time_travel_enabled']
            assert prompt['original_sha256']==sha(REPO/'prompts'/(row['prompt']+'.txt'))
            assert done['active_projection_each_interval']==row['projection']
            assert done['counts']==expected_counts(89,row['steps'],row['backend']=='pixeldit')
            assert done['details']['backward_calls']==done['details']['replay_count']==0
        entry=dict(backend=row['backend'],projection=row['projection'],prompt=row['prompt'],path=row['output'],sha256=sha(row['output']),job=job)
        records.append(entry);emit('OLD89_VERIFIED',**entry)
    assert len(records)==24
    return records


def write_readmes(source,covers):
    ROOT.mkdir(parents=True,exist_ok=True)
    text=f'''# Camera Patching Strategy Study

Compare different 89-camera perspective patch-center distributions while keeping the rest of DiffPano fixed.
The comparison is old89, FibonacciN with N=89, and RandomN with N=89 and layout_seed=0.
New results are `fibonacci_n89` and `random_n89_seed0`; old89 images remain at
`{BASE}` and are neither copied nor regenerated.

Prompts: ruins, underwater, firework. Backends: sana, flux, sd35, pixeldit. Consensus projections: ERP and CEA.
Each new `<strategy>/<backend>/<projection>/<prompt>/` folder contains only `final.png` and compact `config.json`.
There are 48 requested new runs and 24 existing comparison images. A blocked geometry is never generated.

Fixed: diffusion seed 0, batch 1, 89 fixed cameras, roll 0, 80°×80° FoV, GWTFlow initialization,
center-weighted arithmetic RGB averaging (`weighted_average`, `spherediff_center`, temperature 0.1), standard warp,
validated local identity-preserving bridge (PixelDiT has no VAE), current-state transition, no time travel, no LPW, no DPA.
Canvas H=2048, W=4096; every final PNG is a 4096×2048 ERP. CEA runs use CEA during every consensus interval and export once at termination.
Model checkpoints, schedules, guidance and native/raster resolutions are inherited unchanged from the completed sweep.

Fibonacci: y_i=1-2*(i+0.5)/N; golden_angle=pi*(3-sqrt(5)); yaw_i=wrap_to_pi(i*golden_angle+phase);
pitch_i=asin(y_i); phase=0.0. This approximates uniform camera-center density, not an exact equal-area partition of perspective footprints.
Random: a dedicated CPU torch.Generator seeded with layout_seed=0 draws an N×2 FP32 tensor with torch.rand;
row i gives u_i and v_i, yaw_i=2*pi*u_i-pi, z_i=2*v_i-1, pitch_i=asin(z_i).
Thus yaw is uniform on [-pi,pi) and sin(pitch) is uniform on [-1,1].
One fixed random layout is shared by every prompt, backend, projection and denoising timestep; cameras are never resampled.
The layout RNG is separate from the diffusion/GWTFlow RNG.

Changing camera geometry also changes GWTFlow transport graphs, cross-view initial-noise covariance,
directional-prompt allocation and spatial center-weight overlap. Interpret this as a camera-patching strategy intervention.
Angular hashes exclude raster dimensions and use ordered yaw/pitch/roll/FoV values.
Geometry checks use every ERP and CEA pixel center, 1,000,003 deterministic spherical probes, and both exact poles.
Geometry details, validation, job IDs, errors and progress are in normal `logs/campatch-*.out/.err` Slurm logs.

Git commit: {source['git_commit']}; dirty working tree: {source['git_dirty']}.
Prepared UTC: {datetime.now(timezone.utc).isoformat()}. Completion is established by successful worker exits plus PNG/config verification, not this README.
'''
    (ROOT/'README.md').write_text(text)
    for strategy in STRATEGIES:
        folder=ROOT/strategy_name(strategy,89);folder.mkdir(exist_ok=True)
        common=f'''\nN=89; FoV=80×80 degrees; roll=0. Cameras remain fixed over every denoising step.
Identical angular layout is used across all four backends, all three prompts, and ERP/CEA.
Diffusion seed=0; no time travel. Only final.png and config.json are saved per successful run.
Angular geometry SHA256: `{covers[strategy]['angular_geometry_sha256']}`.
Coverage gate: {'PASSED' if covers[strategy]['passed'] else 'BLOCKED: uncovered rays; no production or parameter change allowed'}.
'''
        if strategy=='fibonacci':
            body='# FibonacciN\n\nstrategy=FibonacciN; phase=0.0. Deterministic angular geometry for arbitrary N.\n' \
                'y_i=1-2*(i+0.5)/N; golden_angle=pi*(3-sqrt(5)); yaw_i=wrap_to_pi(phase+i*golden_angle); pitch_i=asin(y_i).\n' \
                'Stable IDs: fibonacci:n{N:03d}:{i:03d}. This is camera-center density, not a perspective-footprint partition.\n'
        else:
            body='# RandomN\n\nstrategy=RandomN; layout_seed=0. One fixed randomly sampled layout with uniform spherical-area sampling.\n' \
                'Dedicated torch CPU generator; torch.rand((N,2), dtype=float32); row i gives (u,v).\n' \
                'yaw_i=2*pi*u-pi; z_i=2*v-1; pitch_i=asin(z_i). Stable IDs: random:n{N:03d}:seed{layout_seed}:{i:03d}.\n' \
                'The layout seed is separate from the diffusion seed. Never resample by prompt/backend/projection/timestep, or select a new seed after inspection.\n'
        (folder/'README.md').write_text(body+common)


def main():
    import torch
    from diffpano.config import ViewConfig
    from studies.tt_cea.cameras import load_cover
    before=fingerprint();source=source_record()
    emit('VALIDATION_START',source=source,job=os.environ.get('SLURM_JOB_ID'))
    assert os.environ.get('SLURM_JOB_ID'),'Run full validation in Slurm'
    subprocess.run([sys.executable,'-m','compileall','-q',str(STUDY)],check=True)
    subprocess.run(['git','diff','--check'],check=True)
    env=dict(os.environ);env['PYTHONPATH']=str(REPO/'tests')+':'+str(REPO)+':'+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(STUDY/'tests'),'-v'],check=True,env=env)
    regression=['test_gwtf_noise_initialization','test_camera_projection','test_current_state_transition',
        'test_fusion_lpw.FusionTests.test_weighted_average_arithmetic','test_fusion_lpw.FusionTests.test_spherediff_center_confidence_decreases_to_edge',
        'test_bridge_factorial.FactorialTests.test_bridge_identity_and_locality','studies.tt_cea.tests.test_canvas',
        'studies.tt_cea.tests.test_pipeline.PipelineTests.test_full_noop_against_actual_frozen_loop']
    subprocess.run([sys.executable,'-m','unittest','-v']+regression,check=True,env=env)
    baseline=baseline_audit();view=ViewConfig(height=1024,width=1024,fov_x=80,fov_y=80)
    all_covers={'old89':load_cover('old89',1024,1024)[0]}
    all_covers.update({s:cover(s,view,89,layout_seed=0,phase=0.) for s in STRATEGIES})
    torch.backends.cuda.matmul.allow_tf32=False
    coverage={}
    for name,cams in all_covers.items():
        r=routing(cams,prompt_record(REPO/'prompts/ruins.txt')[0]['effective_lines'])
        emit('CAMERA_LAYOUT',strategy=name,N=len(cams),angular_sha256=angular_hash(cams),semantic_band_counts=r['semantic_band_counts'])
        coverage[name]=analyze_camera_cover(cams,device='cuda' if torch.cuda.is_available() else 'cpu',
            progress=lambda domain,summary:emit('GEOMETRY_DOMAIN',strategy=name,domain=domain,summary=summary))
        emit('GEOMETRY_COVER',strategy=name,**coverage[name])
    assert coverage['old89']['passed']
    allowed=[s for s in STRATEGIES if coverage[s]['passed']];blocked=[s for s in STRATEGIES if not coverage[s]['passed']]
    write_readmes(source,coverage)
    assert fingerprint()==before
    emit('CAMERA_VALIDATION_PASSED',fingerprint=before,job=os.environ['SLURM_JOB_ID'],allowed_strategies=allowed,
        blocked_strategies=blocked,coverage=coverage,old89_baselines=baseline)

if __name__=='__main__':main()
