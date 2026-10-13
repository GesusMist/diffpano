"""Construct and freeze exactly 28 semantic rows; no historical result reuse."""
from dataclasses import asdict
from studies.ournew_comparison.common import *
from studies.ournew_comparison.configuration import resolved,anchor_location

def main():
    assert os.environ.get('SLURM_JOB_ID')
    assert not (OUT/'manifest.json').exists(),'Existing manifest must not be overwritten'
    from studies.flux_step_controls.runtime import original_spec
    from studies.original_spherediff.common import official_hashes,ROOT as ORIGINAL
    from studies.camera_patching.cameras import angular_hash
    from diffpano.gradient_fusion import GradientSettings
    assert GradientSettings(reference_mode='single_pixel').single_pixel_source=='fused_rgb_erp','Anchor change is not applied'
    prior=read(ORIGINAL/'manifest.json');assert prior['official_source_hashes']==official_hashes()
    baseline=read(ROOT/'outputs/flux-step-controls-seed0/validation/preflight-spherediff.json')
    assert baseline['passed'];official=baseline['record']
    weights=read(ROOT/'outputs/metrics-evaluation/seed0-v1/checkpoint_weight_audit.json');assert weights['all_shared_files_identical']
    source=sources();rows=[];locations={};prompts={p:prompt(p)[0] for p in PROMPTS}
    for row in definitions():
        info=prompts[row['prompt']]
        if row['method']=='diffpano':
            c,cameras,owners,slots,settings,geometry,old=resolved(row)
            if row['layout'] not in locations:locations[row['layout']]=anchor_location(cameras,owners)
            semantic=dict(prompt=info,configuration=asdict(c),geometry=geometry,solver=asdict(settings),
                anchor=dict(source=settings.single_pixel_source,semantics_version=settings.anchor_semantics_version,location_policy=settings.anchor_location_policy,location=locations[row['layout']]) if row['reference_mode']=='single_pixel' else None,
                schedule=old['prepared_schedule'],initializer='layout-aware GWTF with preserved camera noise-slot labels',
                model=dict(id=c.model.id,revision=c.model.revision,weight_audit_sha256=sha(ROOT/'outputs/metrics-evaluation/seed0-v1/checkpoint_weight_audit.json')))
        else:
            spec,delta=original_spec();assert spec['call']['num_inference_steps']==20
            semantic=dict(prompt=info,configuration=spec,official_defaults_step_difference=delta,schedule=official['schedule'],
                geometry=dict(layout='official',camera_count=89,source_sha256=digest(official_hashes()),projection='spherical'),
                initializer=official['initialization'],model=dict(id=spec['model_source'],revision=spec['revision'],weight_audit_sha256=sha(ROOT/'outputs/metrics-evaluation/seed0-v1/checkpoint_weight_audit.json')),
                solver=None,anchor=None,refinement=None)
        semantic['case']={k:v for k,v in row.items() if k not in ('index','case_id')}
        row.update(semantic=semantic,semantic_sha256=digest(semantic),source_sha256=digest(source));rows.append(row)
    assert len(rows)==len({r['case_id'] for r in rows})==len({r['semantic_sha256'] for r in rows})==28
    write(OUT/'manifest.json',dict(rows=rows,prompts=prompts,source_hashes=source,logical_cases=28,expected_new=28,reused=0,
        official_source_hashes=official_hashes(),official_reference=official,weight_audit=weights,seed=0,steps=20,
        solver_policy=dict(coarse_color_rtol=1e-4,other_rtol=1e-5,atol=1e-7,screened_cap=200,constrained_cap=400),
        max_concurrent_generation_gpus=2,working_tree_status=subprocess.check_output(['git','status','--short'],cwd=str(ROOT)).decode(),implementation_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip()))
    # Protect completed earlier records; images are recorded by their immutable metadata hashes.
    protected={}
    for root in (ROOT/'outputs/10.8gradient-refinement-flux/seed0-v1',ROOT/'outputs/10.7gradient-blending-flux/seed0-v1',ROOT/'outputs/10.9gradient-metrics/seed0-v1'):
        for path in root.rglob('*.json'):
            if 'logs' not in path.parts:protected[str(path)]=sha(path)
        for path in root.glob('*.csv'):protected[str(path)]=sha(path)
    write(OUT/'history/protected.json',protected)
    write(OUT/'history/prior-coarse-retries.json',dict(validation=read(ROOT/'outputs/10.8gradient-refinement-flux/seed0-v1/retries/rtol-1e-4/validation.json'),
        statuses=[read(ROOT/'outputs/10.8gradient-refinement-flux/seed0-v1/cases'/key/'status.json') for key in (
        'flux/seed0/air_balloons/coarse_color/poisson_select/refine000','flux/seed0/air_balloons/coarse_color/poisson_max/refine000',
        'flux/seed0/underwater/coarse_color/poisson_select/refine000','flux/seed0/underwater/coarse_color/poisson_select/refine010','flux/seed0/underwater/coarse_color/poisson_max/refine000')]))
    assert all(x['state']=='complete' for x in read(OUT/'history/prior-coarse-retries.json')['statuses'])
    print('MANIFEST_READY',28,'new trajectories',flush=True)
if __name__=='__main__':main()
