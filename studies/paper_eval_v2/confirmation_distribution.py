"""Predeclared confirmation contrasts using frozen full-dimensional estimators."""
from .common import *
COMPARISONS=(('sphere20','select'),('spherope','select'))


def evaluate(run=RUN,max_bootstrap_seconds=300.):
    import numpy as np
    from .cli import verify_protocol
    from .metrics import load_features
    from .distribution import collection_scores,paired_bootstrap,paired_omni_bootstrap
    verify_protocol(run)
    plan=read(run/'generation/manifest.json')
    cases={r['case_id']:r for r in plan['rows'] if r['tier']=='core'}
    expected=sorted({r['prompt_id'] for r in cases.values()})
    rows=[r for r in read(run/'group1/per_panorama_metrics.json') if r['case_id'] in cases
          and r['namespace']=='common_v2' and 'feature_path' in r]
    by_method={m:{r['prompt_id']:r for r in rows if r['method_id']==m} for m in ('sphere20','select','spherope')}
    common=sorted(set.intersection(*(set(v) for v in by_method.values())))
    folder=run/'group2/confirmation';folder.mkdir(parents=True,exist_ok=True)
    reference_file=run/'group2/reference_features.json'
    if not reference_file.exists() or not read(reference_file).get('complete') or len(common)<2:
        result=dict(state='pending_matched_features',matched_prompt_ids=common,missing_prompt_ids=sorted(set(expected)-set(common)))
        write(folder/'status.json',result);return result
    ref=read(reference_file)
    source=read(run/'real_dataset_manifest.json')
    inputs=run/'group2/real_reference_inputs.json';adapted=read(inputs)
    expected_references={r['scene_id']:r['image_sha256'] for r in source['rows']}
    actual_references={r['reference_id']:r['image_sha256'] for r in ref['records']}
    if (ref.get('namespace')!='common_v2' or ref.get('reference_manifest_sha256')!=sha(inputs)
            or Path(ref['reference_manifest_path']).resolve()!=inputs.resolve()
            or adapted.get('source_manifest_sha256')!=sha(run/'real_dataset_manifest.json')
            or len(ref['records'])!=24 or len(actual_references)!=24 or actual_references!=expected_references):
        raise ValueError('Real reference features do not identify the frozen24 ODI-SR source images')
    reference=[load_features(r) for r in ref['records']]
    features={m:{p:load_features(by_method[m][p]) for p in common} for m in by_method}
    shared=dict(cohort_id='confirmation24',namespace='common_v2',reference_kind='real',
        reference_manifest_sha256=ref['reference_manifest_sha256'],reference_features_index_sha256=sha(reference_file),
        source_capture_coverage_sha256=sha(run/'dataset/capture_coverage.json'),
        prompt_ids=common,expected_prompt_ids=expected,missing_prompt_ids=sorted(set(expected)-set(common)),
        matched_prompt_count=len(common),core_seed=0,scope='predeclared_confirmation_contrast_exploratory_distribution',
        reference_estimand='fixed frozen ODI-SR24 reference collection; interval varies generated prompt clusters only',
        interpretation='Compact dependent-view collections; twenty-four scenes are not independent crops.')
    groups=[dict(shared,method_id=m,**collection_scores([features[m][p] for p in common],reference)) for m in features]
    write(folder/'group_metrics.json',groups)
    reference_x=np.concatenate([r['inception'] for r in reference]);pairs=[]
    for baseline,comparison in COMPARISONS:
        first={p:features[baseline][p]['inception'] for p in common}
        second={p:features[comparison][p]['inception'] for p in common}
        for metric in ('FID','KID'):
            result=paired_bootstrap(first,second,reference_x,metric,max_seconds=max_bootstrap_seconds)
            pairs.append(dict(shared,baseline=baseline,comparison=comparison,metric=metric,**{k:v for k,v in result.items() if k not in shared}))
            write(folder/'paired_comparisons.json',pairs)
        result=paired_omni_bootstrap({p:[features[baseline][p]] for p in common},
            {p:[features[comparison][p]] for p in common},reference,max_seconds=max_bootstrap_seconds)
        pairs.append(dict(shared,baseline=baseline,comparison=comparison,metric='OmniFID',**{k:v for k,v in result.items() if k not in shared}))
        write(folder/'paired_comparisons.json',pairs)
    for name,values in (('group_metrics',groups),('paired_comparisons',pairs)):
        csv_write(folder/(name+'.csv'),values,sorted({k for r in values for k in r}))
    result=dict(state='complete' if len(common)==24 and all(r['state']=='complete' for r in pairs) else 'partial',
        matched_prompt_ids=common,groups=len(groups),contrasts=len(pairs),source_sha256=sha(__file__),
        incomplete_uncertainty=sum(r['state']!='complete' for r in pairs))
    write(folder/'status.json',result);verify_protocol(run);return result

if __name__=='__main__':print(evaluate())
