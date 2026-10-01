"""Explicitly authorized, separate reexecution to recover omitted pole metadata."""
import traceback
from studies.tt_cea.common import *
from studies.tt_cea.metadata import assert_same_recovery_sample, validate_pole_record


def recover(backend,authorization_file):
    authorization=read(authorization_file)
    assert authorization['approved'] and [backend,'ruins','C0'] in authorization['authorized_cases']
    assert authorization['user_instruction'].strip()
    require_gate(backend,'C0')
    original_folder=sample_dir(backend,'ruins','C0')
    original_status=read(original_folder/'status.json');original=read(original_folder/'metadata.json')
    assert original_status['state']=='complete' and sha(original_folder/'metadata.json')==original_status['metadata_sha256']
    assert original.get('terminal_pole_rgb') is None
    target=ROOT/'pole-metadata-recovery'/backend/'attempt-1'
    certificate=target.parent/'certificate.json'
    assert not target.exists() and not certificate.exists(),'No silent retries or duplicate recovery'
    from studies.tt_cea import run as runner
    # This fresh process redirects only the NEW study runner's output directory.
    # No historical module, model, schedule, tensor, or numerical helper is patched.
    original_directory_function=runner.sample_dir
    def recovery_directory(name,prompt,case):
        assert (name,prompt,case)==(backend,'ruins','C0')
        return target
    try:
        runner.sample_dir=recovery_directory
        runner.run(backend,'ruins','C0')
    finally:
        runner.sample_dir=original_directory_function
    try:
        status=read(target/'status.json');candidate=read(target/'metadata.json')
        assert status['state']=='complete' and sha(target/'metadata.json')==status['metadata_sha256']
        assert_same_recovery_sample(original,candidate)
        for file in ('final.png','final_cea.png','viewports.png','schedule.csv'):
            assert sha(original_folder/file)==original['artifacts'][file]
            assert sha(target/file)==candidate['artifacts'][file]
        assert sha(original_folder/'metadata.json')==original_status['metadata_sha256']
        atomic(certificate,dict(passed=True,purpose='terminal pole metadata recovery only',
            backend=backend,prompt='ruins',case='C0',job=status['job'],
            original_folder=str(original_folder),candidate_folder=str(target),
            original_metadata_sha256=original_status['metadata_sha256'],candidate_metadata_sha256=status['metadata_sha256'],
            terminal_state_sha256=candidate['diagnostics']['terminal_state_sha256'],
            terminal_pole_rgb=validate_pole_record(candidate['terminal_pole_rgb']),
            authorization_sha256=sha(authorization_file),source_hashes=study_hashes()))
        print('POLE METADATA RECOVERED',backend,flush=True)
    except Exception:
        atomic(certificate,dict(passed=False,backend=backend,error=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=parser(__doc__);p.add_argument('--backend',choices=BACKENDS,required=True)
    p.add_argument('--authorization-file',required=True)
    a=p.parse_args();load_settings(a.config);recover(a.backend,a.authorization_file)
