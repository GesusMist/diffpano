"""Freeze the exact experiment after development tests; never overwrite a freeze."""
from dataclasses import asdict
import tarfile
from studies.mean_max_old89.common import *
from studies.mean_max_old89.configuration import resolved


def main():
    assert os.environ.get('SLURM_JOB_ID')
    assert not (OUT/'manifest.json').exists(),'Existing production freeze must not be overwritten'
    evidence=read(OUT/'validation/development-cpu.json');assert evidence['passed']
    source=sources();assert evidence['source_sha256']==digest(source),'Run development tests on final source before freezing'
    weights=read(ROOT/'outputs/metrics-evaluation/seed0-v1/checkpoint_weight_audit.json');assert weights['all_shared_files_identical']
    rows=[];prompts={p:prompt(p)[0] for p in PROMPTS}
    for row in definitions():
        c,cameras,owners,slots,settings,geometry,old=resolved(row)
        semantic=dict(case={k:v for k,v in row.items() if k not in ('index','case_id')},prompt=prompts[row['prompt']],
            configuration=asdict(c),geometry=geometry,solver=settings.to_record(),schedule=old['prepared_schedule'],
            initializer=dict(name='GWTF',noise_slots=slots,baseline_initial_local_sha256=old['initialization']['initial_local_sha256']),
            model=dict(id=c.model.id,revision=c.model.revision,weight_audit_sha256=sha(ROOT/'outputs/metrics-evaluation/seed0-v1/checkpoint_weight_audit.json')),
            guidance=dict(formula='(1-alpha)*g_mean + alpha*g_max',alpha_is='max coefficient',semantics_version=1),
            refinement=dict(last_fraction=0.,time_travel=False,replays=0,dynamic_cameras=False))
        row.update(semantic=safe(semantic),semantic_sha256=digest(semantic),source_sha256=digest(source));rows.append(row)
    assert len(rows)==len({r['case_id'] for r in rows})==len({r['semantic_sha256'] for r in rows})==90
    (OUT/'effective-prompts').mkdir(exist_ok=True)
    prior=ROOT/'outputs/10.9ournew-flux28/seed0-v1'
    runtimes=[read(p)['generation_seconds'] for p in (prior/'cases').rglob('metadata.json') if '/spherediff_official/' not in str(p)]
    assert len(runtimes)==24
    import statistics
    protected={};links={}
    for root in (ROOT/'outputs/10.8gradient-refinement-flux/seed0-v1',ROOT/'outputs/10.7gradient-blending-flux/seed0-v1',ROOT/'outputs/10.9gradient-metrics/seed0-v1',prior):
        for path in root.rglob('*'):
            if path.is_symlink():links[str(path)]=os.readlink(path)
            elif path.is_file() and path.suffix in ('.json','.csv','.md','.tex') and 'logs' not in path.parts:protected[str(path)]=sha(path)
    write(OUT/'history/protected.json',protected);write(OUT/'history/protected-symlinks.json',links)
    write(OUT/'history/prior-generation-profile.json',dict(count=24,seconds=runtimes,minimum=min(runtimes),median=statistics.median(runtimes),maximum=max(runtimes),
        walltime_seconds=10800,reason='Prior actual 24 DiffPano runs plus conservative allowance for mixed guidance/setup; first matrix row must pass before remaining launch'))
    write(OUT/'manifest.json',dict(rows=rows,prompts=prompts,source_hashes=source,weight_audit=weights,logical_cases=90,expected_new=90,reused=0,
        subsets=dict(original11=66,new4=24),formula='(1-alpha)*g_mean + alpha*g_max',alpha_is='max coefficient',default_alpha=.5,
        solver_policy=dict(screened_rtol=1e-5,coarse_color_rtol=1e-4,atol=1e-7,screened_cap=200,constrained_cap=400),
        expected_generation_counts=COUNTS,expected_poisson_solves=21,max_concurrent_generation_gpus=2,
        implementation_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip(),
        working_tree_status=subprocess.check_output(['git','status','--short'],cwd=str(ROOT)).decode(),development_tests=evidence))
    with tarfile.open(str(OUT/'history/frozen-source.tar.gz'),'w:gz') as archive:
        for name in source:archive.add(str(ROOT/name),arcname=name)
    (OUT/'history/implementation.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=str(ROOT)))
    write(OUT/'history/freeze.json',dict(manifest_sha256=sha(OUT/'manifest.json'),source_sha256=digest(source),archive_sha256=sha(OUT/'history/frozen-source.tar.gz')))
    print('MANIFEST_READY: 90 new trajectories; 15 prompts; 6 variants per prompt',flush=True)

if __name__=='__main__':main()
