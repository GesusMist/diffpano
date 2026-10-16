"""Validate exact identities, audit existing artifacts, then freeze only missing runs."""
import argparse,subprocess,tarfile,unittest
from dataclasses import asdict
from studies.layout24.common import *
from studies.layout24.configuration import resolved,coverage_evidence


def expected_initialization(row,old):
    if row['layout']=='old89':init=old['initialization'];source=None
    elif row['layout']=='uniform89':
        source=ROOT/'outputs/10.9ournew-flux28/seed0-v1/validation/gpu.json'
        g=read(source);assert g['passed'];init=g['groups']['bamboo_grove/uniform89']['initialization']
    else:
        strategy='old' if row['layout'].startswith('old') else 'fibonacci'
        source=ROOT/'outputs/camera-count-erp-seed0-v2/preflight'/('diffpano-%s-n%d-flux.json'%(strategy,row['camera_count']))
        g=read(source);assert g['passed'];init=dict(g['record']['initialization'])
        init['initial_local_sha256']=init.pop('initial_state_sha256')
    values={k:safe(init[k]) for k in ('initial_local_sha256','source_sha256','source_shape','scaling','scaling_applications')}
    if source is not None:values.update(evidence_path=str(source),evidence_sha256=sha(source))
    return values


def validate_geometry_sources():
    """Only scientifically relevant old implementations must equal their archived versions."""
    p=ROOT/'outputs/10.9ournew-flux28/seed0-v1/manifest.json';manifest=read(p)
    paths=('diffpano/projection.py','diffpano/warp.py','diffpano/fusion.py','studies/tt_cea/canvas.py',
        'diffpano/camera.py','studies/camera_patching/cameras.py','diffpano/erp_noise_initialization.py','diffpano/gwtf_noise_initialization.py')
    checked={}
    for name in paths:
        assert sha(ROOT/name)==manifest['source_hashes'][name],'Scientific geometry/noise source changed: '+name
        checked[name]=sha(ROOT/name)
    return dict(source_manifest=str(p),source_manifest_sha256=sha(p),exact_relevant_sources=checked,
        full_repository_equality_required=False)


def build_rows():
    from studies.flux_step_controls.runtime import original_spec
    from studies.original_spherediff.common import official_hashes,ROOT as ORIGINAL
    prompts={name:prompt(name)[0] for name in PROMPTS};source=sources();rows=definitions()
    preflight=read(ROOT/'outputs/flux-step-controls-seed0/validation/preflight-spherediff.json')
    assert preflight['passed'];official=preflight['record']
    historical=read(ORIGINAL/'manifest.json');assert official_hashes()==historical['official_source_hashes']
    weights_path=ROOT/'outputs/metrics-evaluation/seed0-v1/checkpoint_weight_audit.json'
    weights=read(weights_path);assert weights['all_shared_files_identical']
    banks={}
    for row in rows:
        info=prompts[row['prompt']]
        if row['method']=='diffpano':
            c,cameras,owners,slots,settings,geometry,old=resolved(row)
            semantic=dict(prompt=info,configuration=safe(asdict(c)),geometry=safe(geometry),solver=settings.to_record(),
                schedule=old['prepared_schedule'],expected_initialization=expected_initialization(row,old),
                initializer='layout-aware GWTF; canonical per-camera noise-slot labels; integer gradient owners separately mapped',
                model=dict(id=c.model.id,revision=c.model.revision,weight_audit_sha256=sha(weights_path)),
                refinement=dict(last_fraction=0.,coupled_intervals=20,independent_intervals=0),
                guidance=None if settings.mode=='rgb' else dict(mode='poisson_select',selection='greatest minimum endpoint geometric center-confidence; stable camera-ID tie; whole signed RGB edge'))
            if row['layout'] not in banks:banks[row['layout']]=dict(geometry=geometry,expected_initialization=semantic['expected_initialization'])
        else:
            spec,delta=original_spec();assert spec['call']['num_inference_steps']==20
            semantic=dict(prompt=info,configuration=spec,official_defaults_step_difference=delta,schedule=official['schedule'],
                geometry=dict(layout='official',camera_count=89,source_sha256=digest(official_hashes()),projection='spherical'),
                initializer=official['initialization'],model=dict(id=spec['model_source'],revision=spec['revision'],weight_audit_sha256=sha(weights_path)),
                solver=None,refinement=None)
        semantic['case']={k:v for k,v in row.items() if k not in ('index','case_id')}
        row.update(semantic=safe(semantic),semantic_sha256=digest(semantic),source_sha256=digest(source))
    assert len({r['semantic_sha256'] for r in rows})==216
    return rows,prompts,banks,preflight,weights


def main(validate_only=False):
    assert os.environ.get('SLURM_JOB_ID'),'Preparation requires a Slurm CPU allocation'
    tests=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in ('studies.layout24.tests_geometry','studies.layout24.reuse_tests'))
    result=unittest.TextTestRunner(verbosity=2).run(tests);assert result.wasSuccessful()
    scientific_sources=validate_geometry_sources();rows,prompts,banks,official,weights=build_rows()
    coverage=coverage_evidence(rows)
    if validate_only:
        write(OUT/'validation/preliminary.json',dict(passed=True,tests_run=result.testsRun,coverage=coverage,scientific_sources=scientific_sources))
        print('LAYOUT24_PRELIMINARY_VALID',len(rows),flush=True);return
    assert not (OUT/'manifest.json').exists(),'Refusing to overwrite the frozen Task B manifest'
    from studies.layout24.reuse import audit,adopt
    audit_record,plan=audit(rows=rows,publish=True)
    assert len(plan['rows'])==216 and {r['index'] for r in plan['rows']}==set(range(216))
    source=sources();assert all(r['source_sha256']==digest(source) for r in rows),'Source changed during prepare'
    manifest=dict(rows=rows,prompts=prompts,logical_cases=216,source_hashes=source,seed=0,steps=20,
        official_preflight=official,weight_audit=weights,geometry_banks=banks,coverage=coverage,
        expected_generation_counts={str(n):expected_counts(dict(camera_count=n)) for n in (89,70,50)},
        solver_policy=dict(mode='poisson_select',reference_mode='screened',lambda_color=.1,rtol=1e-5,atol=1e-7,max_iterations=200),
        max_concurrent_generation_gpus_across_tasks=2,outputs_symlink=os.readlink(ROOT/'outputs'),
        generation_plan_sha256=sha(OUT/'generation_plan.json'),reuse_audit_sha256=sha(OUT/'reuse_audit.json'),
        implementation_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip(),
        scientific_sources=scientific_sources)
    write(OUT/'manifest.json',manifest)
    for item in plan['rows']:
        if item['state']=='reused':adopt(rows[item['index']],item['source'])
    with tarfile.open(str(OUT/'frozen-source.tar.gz'),'w:gz') as archive:
        for name in source:archive.add(str(ROOT/name),arcname=name)
    write(OUT/'validation/prepared.json',dict(passed=True,tests_run=result.testsRun,source_sha256=digest(source),
        manifest_sha256=sha(OUT/'manifest.json'),coverage=coverage,scientific_sources=scientific_sources,
        source_archive_sha256=sha(OUT/'frozen-source.tar.gz'),extra_full_trajectories=0,
        GPU_profile_policy='First full production case of each missing path occupies an authorized target; runtime checks all measured counts and solver/coverage gates'))
    gate();print('LAYOUT24_MANIFEST_READY',plan.get('counts'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--validate-only',action='store_true');args=parser.parse_args();main(args.validate_only)
