"""Partial or completed factual report; no unobserved outcomes are claimed."""
from studies.gradient_refinement.common import *
from studies.gradient_refinement.status import inventory

def main():
    status=inventory();write(OUT/'latest-status.json',status);manifest=read(OUT/'manifest.json')
    lines=['DiffPano FLUX Poisson reference / selection / independent final refinement',
        'Implementation base revision: '+manifest['implementation_revision'],
        '64 logical cases: four references x two gradients x four prompts x two fractions.',
        'Audited historical reuse: %d; requested new generations: %d.'%(manifest['audited_reuse'],manifest['new_generation']),
        'Current states: '+json.dumps(status['counts'],sort_keys=True),
        'Fixed: seed 0, original 20 FLUX intervals, saved ordered Old89, 80x80 degree FOV, local RGB 1024x1024, ERP H2048xW4096, GWTF.',
        'screened: sum M|DI-g|^2 + 0.1 sum |I-R|^2 (original solver unchanged).',
        'global_mean: min sum M|DI-g|^2, mean(I)=mean(R), explicit zero-mean solve; no ridge.',
        'single_pixel: same zero-mean solution, offset c0-J(p0), current selected source at the selected ERP ray.',
        'coarse_color: sum M|DI-g|^2 + 0.1 sum_b n_b |B_b(I-R)|^2, mean(I)=mean(R).',
        'Coarse 16x32 blocks and eta=.1 are fixed initial choices, not tuned optimum values.',
        'Select copies the signed RGB vector of highest confidence. Max copies highest RGB L2 magnitude among the same valid edges; stable camera-ID ties.',
        'Refinement 0.10: coupled intervals 1-18; native independent 19-20; one terminal fusion with the case gradient AND reference.',
        'Expected counts off/on: denoiser 1780/1780, encode 3560/3204, decode 1869/1691, fusions 21/19.',
        'Four reused cases lack a historical raw FP32 image; new cases retain unclamped final FP32 and terminal RGB reference.',
        'Metrics use frozen panorama implementations. Four prompts and dependent report views support descriptive comparisons only.']
    for name in ('validation.json','gpu-smoke.json'):
        if (OUT/name).exists():
            m=read(OUT/name);lines.append(name+': passed='+str(m['passed'])+' job='+str(m.get('job')))
    completed=[r for r in manifest['rows'] if case_complete(r)]
    for r in completed:
        m=read(folder(r)/'metadata.json')
        lines.append('%s | reused=%s | counts=%s | seconds=%.2f | solves=%d'%(r['key'],m['reused'],json.dumps(m['counts'],sort_keys=True),m['generation_seconds'],len(m['fusions'])))
    for r in status['cases']:
        if r['state'] in ('failed','interrupted','submitted_inactive'):lines.append('INCOMPLETE '+json.dumps(r))
    if (OUT/'evaluation/summary.json').exists():
        evaluation=read(OUT/'evaluation/summary.json');lines+=['Four-prompt descriptive aggregates:']+[json.dumps(r,sort_keys=True) for r in evaluation['aggregates']]
    else:lines.append('Paired numerical outcomes are pending generation and evaluation; no quality conclusion yet.')
    lines+=['Figures: '+str(OUT/'figures'),'Metrics: '+str(OUT/'evaluation'),
        'Status: python3 -m studies.gradient_refinement.status',
        'Dry run: python3 -m studies.gradient_refinement.launch',
        'Resume missing: python3 -m studies.gradient_refinement.launch --submit',
        'Retry failed/incomplete explicitly: python3 -m studies.gradient_refinement.launch --submit --retry']
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    preserved()
    if len(completed)==64 and (OUT/'evaluation/summary.json').exists() and (OUT/'figures/inventory.json').exists():
        figures=read(OUT/'figures/inventory.json');assert len(figures['comparisons'])==80
        evaluation=read(OUT/'evaluation/summary.json');assert len(evaluation['rows'])==64
        write(OUT/'completion.json',dict(passed=True,complete=64,reused=manifest['audited_reuse'],generated=64-manifest['audited_reuse'],
            manifest_sha256=sha(OUT/'manifest.json'),evaluation_sha256=sha(OUT/'evaluation/summary.json'),report_sha256=sha(OUT/'report.txt'),figures=figures['png_count']))
if __name__=='__main__':main()
