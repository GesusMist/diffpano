"""Evaluate the resumed Uniform cases without modifying historical metrics."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

from studies.faster_uniform_resume.common import ROOT, OUT, PLAN, load_plan
from studies.faster_uniform_resume.completion import load_verifier
from studies.layout24 import common as original

EVAL = OUT / 'evaluation'
UNIFORM_VARIANTS = tuple(v for v in original.VARIANTS if '_uniform' in v)


def historical_candidates(identity, existing):
    """Add only the authenticated, protocol-compatible 101-image evaluation."""
    root = original.OUT / 'evaluation'
    protocol_path, metrics_path = root / 'metric_protocol.json', root / 'metrics.json'
    if not protocol_path.is_file() or not metrics_path.is_file():
        return existing
    protocol, metrics = original.read(protocol_path), original.read(metrics_path)
    if (protocol.get('gpu_identity') != identity
            or metrics.get('protocol_sha256') != original.digest(protocol)):
        return existing
    candidates = {key: list(values) for key, values in existing.items()}
    for row in metrics['rows']:
        if row.get('scoring_status') != 'complete':
            continue
        if not all(row.get(key) for key in
                   ('image_sha256', 'effective_prompt_sha256',
                    'feature_path', 'feature_sha256')):
            continue
        candidates.setdefault(row['image_sha256'], []).insert(
            0, dict(row, record_source=str(metrics_path)))
    return candidates


@contextmanager
def adapted_evaluator(verifier):
    """Inject bookkeeping only; use the original DS/CS/IS implementation."""
    from studies.layout24 import evaluate as legacy
    saved = {name: getattr(legacy, name)
             for name in ('EVAL', 'complete', 'read', 'write', 'feature_candidates')}
    rows = original.read(original.OUT / 'manifest.json')['rows']
    statuses = {original.folder(row) / 'status.json': row for row in rows}
    destination = OUT.resolve()

    def read(path):
        path = Path(path)
        if path in statuses and not path.exists():
            row = statuses[path]
            if verifier.protected_complete(row):
                # This flag only reports historical provenance; all actual
                # completion evidence comes from the retained-image verifier.
                return dict(state='complete',
                            reused=verifier.protected_records[row['index']]['reused'],
                            status_pruned_by_authorized_cleanup=True)
        return saved['read'](path)

    def write(path, value):
        path = Path(path)
        if path == original.OUT / 'completion.json':
            path = OUT / 'full-matrix-completion.json'
        assert path.resolve().is_relative_to(destination), (
            'Evaluator attempted to write outside the new resume output', path)
        return saved['write'](path, value)

    legacy.EVAL = EVAL
    legacy.complete = verifier.checked_complete
    legacy.read = read
    legacy.write = write
    legacy.feature_candidates = lambda identity: historical_candidates(
        identity, saved['feature_candidates'](identity))
    try:
        yield legacy
    finally:
        for name, value in saved.items():
            setattr(legacy, name, value)


def resume_metrics(metrics):
    """Full-matrix completeness and this Uniform-only scope stay distinct."""
    uniform_rows = [row for row in metrics['rows']
                    if row['variant'] in UNIFORM_VARIANTS]
    uniform_groups = [group for group in metrics['groups']
                      if group['variant'] in UNIFORM_VARIANTS]
    assert len(uniform_rows) == 96 and len(uniform_groups) == 4
    counts = dict(
        valid_generations=sum(row['completion_status'] == 'complete'
                              for row in uniform_rows),
        DS=sum(row.get('DS') is not None for row in uniform_rows),
        CS=sum(row.get('CS') is not None for row in uniform_rows),
        matched_IS=sum(group.get('IS') is not None for group in uniform_groups))
    failed = [row['case_id'] for row in uniform_rows
              if row['scoring_status'] == 'failed']
    complete = counts == dict(valid_generations=96, DS=96, CS=96, matched_IS=4)
    expected_all = dict(valid_generations=176, DS=176, CS=176, matched_IS=7)
    return dict(
        uniform_complete=complete and not failed,
        evaluation_complete=(complete and not metrics.get('errors')
                             and metrics['counts'] == expected_all),
        uniform_counts=counts,
        expected_uniform_counts=dict(valid_generations=96, DS=96, CS=96, matched_IS=4),
        all_matrix_counts=metrics['counts'],
        expected_after_uniform_resume=expected_all,
        full_216_matrix_complete=metrics['complete'],
        failed_uniform_cases=failed,
        pending_uniform_cases=[row['case_id'] for row in uniform_rows
                               if row['completion_status'] != 'complete'],
        note='The 40 unfinished Old70/Old50 cases are outside this Uniform-only resume.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    load_plan()
    verifier = load_verifier(strict_complete=original.complete)
    adapter_sources = {str(path): original.sha(path)
                       for path in sorted(Path(__file__).parent.glob('*.py'))}
    plan_hash = original.sha(PLAN)
    previous_args = sys.argv
    try:
        # The original evaluator covers all 216 rows. Forty are deliberately
        # outside this continuation, so its full-matrix partial result is normal.
        sys.argv = [str(Path(__file__)), '--allow-partial']
        with adapted_evaluator(verifier) as legacy:
            legacy.main()
    finally:
        sys.argv = previous_args
    metrics = original.read(EVAL / 'metrics.json')
    summary = resume_metrics(metrics)
    assert original.sha(PLAN) == plan_hash, 'Resume plan changed during evaluation'
    assert all(original.sha(path) == value for path, value in adapter_sources.items())
    original.write(EVAL / 'resume-summary.json', dict(
        **summary, adapter_source_hashes=adapter_sources, plan_sha256=plan_hash,
        metrics_sha256=original.sha(EVAL / 'metrics.json'),
        original_evaluator_sha256=original.sha(ROOT / 'studies/layout24/evaluate.py')))
    print('UNIFORM_EVALUATION', summary, flush=True)
    if not summary['evaluation_complete'] and not args.allow_partial:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
