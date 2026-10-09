"""Bounded CPU validation of retry scope, archived failures, and FP32 solving."""
from dataclasses import asdict, replace
import torch
from diffpano.gradient_fusion import GradientSettings, edges, reconstruct
from studies.gradient_refinement_retry.common import *

@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Validation requires a Slurm allocation'
    torch.set_num_threads(4)
    plan = read(RETRY / 'plan.json')
    check_scope(plan)
    assert sha(RETRY / 'original-manifest.json') == plan['original_manifest_sha256']
    recorded = []
    for attempt in plan['failed_attempts']:
        for path, expected in attempt['archived_artifacts'].items():
            assert sha(path) == expected
        failure = attempt['failure']
        threshold = 1e-7 + RTOL * failure['initial_true_residual']
        assert failure['output_normal_residual'] <= threshold
        recorded.append(dict(index=attempt['index'], old_threshold=failure['stopping_threshold'],
            new_threshold=threshold, failed_output_residual=failure['output_normal_residual']))
    torch.manual_seed(208)
    reference = torch.randn(1, 3, 64, 128) * .2
    guidance = (torch.randn_like(reference), torch.randn(1, 3, 63, 128))
    support = (torch.ones(1, 1, 64, 128, dtype=torch.bool), torch.ones(1, 1, 63, 128, dtype=torch.bool))
    records = []
    for mode in ('poisson_select', 'poisson_max'):
        row = next(r for r in plan['rows'] if r['gradient_mode'] == mode)
        settings = GradientSettings(**row['semantic']['fusion'])
        assert asdict(settings) == row['semantic']['fusion']
        actual, record = reconstruct(reference, guidance, support, settings)
        assert record['converged'] and actual.dtype == torch.float32 and torch.isfinite(actual).all()
        assert record['output_normal_residual'] <= record['stopping_threshold']
        assert record['mean_constraint_error'] <= record['constraint_tolerance']
        strict, _ = reconstruct(reference.double(), tuple(g.double() for g in guidance), support,
            replace(settings, relative_tolerance=1e-10, absolute_tolerance=1e-11), dtype=torch.float64)
        torch.testing.assert_close(actual.double(), strict, atol=2e-3, rtol=2e-3)
        record.update(mode=mode, max_error_to_fp64=float((actual.double()-strict).abs().max()))
        records.append(record)
    check_scope(plan)
    write(RETRY / 'validation.json', dict(passed=True, source_hashes=plan['source_hashes'],
        plan_sha256=sha(RETRY / 'plan.json'), job=os.environ['SLURM_JOB_ID'],
        scope='Only five failed rows; only relative tolerance changes',
        archived_failure_threshold_checks=recorded, numerical_checks=records))
    print('RETRY_VALIDATION_PASSED', records, flush=True)

if __name__ == '__main__':
    main()
