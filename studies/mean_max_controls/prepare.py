"""Freeze the user-requested missing controls without touching historical results."""
import argparse,subprocess,tarfile,unittest
from dataclasses import asdict,replace
from studies.mean_max_controls.common import *
from studies.mean_max_controls.configuration import resolved


def prepare(input_path):
    assert os.environ.get('SLURM_JOB_ID'),'Preparation requires a Slurm CPU allocation'
    assert not (OUT/'manifest.json').exists(),'Refusing to overwrite the control freeze'
    tests=unittest.defaultTestLoader.loadTestsFromName('studies.mean_max_controls.tests')
    result=unittest.TextTestRunner(verbosity=2).run(tests)
    assert result.wasSuccessful()
    prior=base.freeze_check()
    for name in ('cpu.json','weights.json','metrics-cpu.json'):base.gate(name)
    gpu=base.gate('gpu.json');base.preserved()
    inputs=read(input_path);rows=definitions(inputs);source=sources()
    prompts={row['prompt']:prompt(row['prompt'])[0] for row in rows}
    checks=[]
    for row in rows:
        c,cameras,owners,slots,settings,geometry,old=resolved(row)
        previous=next(item for item in prior['rows'] if item['prompt']==row['prompt'] and item['reference_mode']==row['reference_mode'] and item['gradient_mix_alpha']==.5)
        assert safe(asdict(c))==previous['semantic']['configuration']
        assert safe(geometry)==previous['semantic']['geometry']
        mixed_solver=dict(previous['semantic']['solver']);mixed_solver.pop('gradient_mix_alpha');mixed_solver['mode']=MODE
        assert settings.to_record()==mixed_solver
        expected=gpu['groups'][row['prompt']+'/old89']
        assert expected['initial_local_sha256']==old['initialization']['initial_local_sha256']
        assert expected['schedule_sha256']==digest(old['prepared_schedule'])
        semantic=dict(case={k:v for k,v in row.items() if k not in ('index','case_id')},prompt=prompts[row['prompt']],
            configuration=asdict(c),geometry=geometry,solver=settings.to_record(),schedule=old['prepared_schedule'],
            initializer=previous['semantic']['initializer'],model=previous['semantic']['model'],
            guidance=dict(mode=MODE,selection='existing geometric-confidence edge selection',display_alias='select_closest'),
            refinement=previous['semantic']['refinement'])
        row.update(semantic=safe(semantic),semantic_sha256=digest(semantic),source_sha256=digest(source))
        checks.append(dict(case_id=row['case_id'],mean_max_equivalent_configuration=True,mean_max_equivalent_geometry=True,
            only_solver_changes=['mode','removed gradient_mix_alpha'],reused_gpu_group=row['prompt']+'/old89'))
    write(OUT/'requested-missing.json',inputs)
    manifest=dict(rows=rows,prompts=prompts,logical_cases=len(rows),source_hashes=source,
        mean_max_manifest_sha256=sha(base.OUT/'manifest.json'),mean_max_gpu_validation_sha256=sha(base.OUT/'validation/gpu.json'),
        missing_input_sha256=sha(input_path),implementation_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip(),
        expected_generation_counts=COUNTS,expected_poisson_solves=21,max_concurrent_generation_gpus=2,
        solver_policy=dict(screened_rtol=1e-5,coarse_color_rtol=1e-4,atol=1e-7,screened_cap=200,constrained_cap=400),
        historical_results_modified=False,checks=checks)
    write(OUT/'manifest.json',manifest)
    with tarfile.open(str(OUT/'frozen-source.tar.gz'),'w:gz') as archive:
        for name in source:archive.add(str(ROOT/name),arcname=name)
    write(OUT/'validation/prepared.json',dict(passed=True,tests_run=result.testsRun,checks=checks,
        source_sha256=digest(source),manifest_sha256=sha(OUT/'manifest.json'),source_archive_sha256=sha(OUT/'frozen-source.tar.gz')))
    validation()
    print('CONTROL_MANIFEST_READY: %d missing Old89 select controls'%len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--missing',required=True);args=parser.parse_args()
    prepare(Path(args.missing))
