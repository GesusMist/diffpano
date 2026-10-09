"""Original report plus explicit mixed-tolerance retry provenance."""
from studies.gradient_refinement_retry.common import *
from studies.gradient_refinement import report

def main():
    plan = read(RETRY / 'plan.json')
    note = dict(indices=plan['indices'], previous_relative_tolerance=1e-5,
        relative_tolerance=RTOL, absolute_tolerance=1e-7, plan=str(RETRY / 'plan.json'),
        caveat='Five coarse-color cases rerun with relaxed relative tolerance after FP32 output-residual failures; the remaining 59 cases retain original settings.')
    summary = OUT / 'evaluation/summary.json'
    if summary.exists():
        value = read(summary)
        value['tolerance_retries'] = note
        write(summary, value)
    report.main()
    path = OUT / 'report.txt'
    lines = ['', 'Tolerance-only retry exceptions (explicit user request):', note['caveat'],
        'rtol 1e-5 -> 1e-4; atol remains 1e-7; iteration limit remains 400.',
        'Original numerical and generation implementations are unchanged.',
        'Archived failed attempts and retry validation: ' + str(RETRY)]
    lines += ['RETRY ' + row['key'] for row in plan['rows']]
    lines += ['These five cases use: python3 -m studies.gradient_refinement_retry.run --index <index> --retry',
        'Their manifest settings differ from the original runner defaults.']
    path.write_text(path.read_text() + '\n'.join(lines) + '\n')
    completion = OUT / 'completion.json'
    if completion.exists():
        value = read(completion)
        value['report_sha256'] = sha(path)
        value['evaluation_sha256'] = sha(summary)
        value['tolerance_retries'] = note
        write(completion, value)

if __name__ == '__main__':
    main()
