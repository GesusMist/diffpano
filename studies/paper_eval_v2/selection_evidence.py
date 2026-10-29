"""Record development-only evidence for the predeclared confirmation fallback."""
from .common import *


def prepare(run=RUN):
    import statistics
    groups=read(run/'group1/group_metrics.json')
    pairs=read(run/'group1/paired_comparisons.json')
    outcomes=('DS','CS-H','CS-directional','MUSIQ')
    chosen=[r for r in groups if r['namespace']=='common_v2' and r['cohort_id']=='development15'
        and r['method_id'] in PRIMARY_METHODS and r['metric'] in outcomes]
    if len(chosen)!=16 or any(r['prompt_count']!=15 for r in chosen):
        raise ValueError('Complete matched development15 scalar evidence required')
    inventory=read(run/'inventory/records.json')
    cohorts=read(run/'inventory/cohorts.json')
    ids={row['case_id'] for row in cohorts['selected_cells']}
    prompts=set(cohorts['cohorts']['development15']['prompt_ids'])
    runtime={}
    for method in PRIMARY_METHODS:
        rows=[row for row in inventory if row['case_id'] in ids and row['method_id']==method and row['prompt_id'] in prompts and row['state']=='complete']
        values=[row['runtime']['generation_seconds'] for row in rows if row.get('runtime',{}).get('generation_seconds') is not None]
        runtime[method]=dict(observed_generation_seconds=values,median_generation_seconds=statistics.median(values) if values else None,
            observed_count=len(values),limitations='Historical setup/memory instrumentation is incomplete; no inferred missing times or cross-hardware normalization.')
    evidence=dict(cohort_id='development15',prompt_ids=sorted(prompts),groups=chosen,
        paired_comparisons=[r for r in pairs if r['namespace']=='common_v2' and r['cohort_id']=='development15' and r['metric'] in outcomes],
        runtime=runtime,source_protocol_registry_sha256=sha(run/'protocol_registry.json'),
        source_group_metrics_sha256=sha(run/'group1/group_metrics.json'),source_paired_comparisons_sha256=sha(run/'group1/paired_comparisons.json'),
        outcome='use_predeclared_unrefined_screened_select_fallback',selected_method='select',
        rationale=['Mean-max alpha0.50 leads the matched development means for MUSIQ and both semantic scores; it has the highest discontinuity score among the four primary candidates.',
            'RGB has the lowest discontinuity score and the lowest MUSIQ mean; low DS alone does not establish superior perceptual or geometric quality.',
            'These development measures do not establish one configuration as best across perceptual quality, geometry, semantics and cost. The planned subjective geometry evidence remains incomplete, and pilot sensitivity was weak.',
            'Carry the explicitly predeclared unrefined screened-select Old89 fallback into confirmation. This is not a claim that select won development metrics. Mean-max remains an identified ablation.'],
        confirmation_outcomes_consulted=False,refinement_included=False,missing_required_development_cases=[],
        calibration_review_sha256=sha(run/'vlm/calibration_review.json'),recorded_utc=now())
    path=run/'generation/final_method_evidence.json'
    if path.exists():return read(path)
    freeze(path,evidence)
    return dict(path=str(path),sha256=sha(path),state='frozen')

if __name__=='__main__':print(prepare())
