"""Equal-prompt summaries and paired prompt-cluster uncertainty, never crop CIs."""
from collections import defaultdict
import numpy as np

BOOTSTRAP_SEED = 20261026
BOOTSTRAP_REPLICATES = 2000
PRIMARY_METHODS = ('sphere20', 'rgb', 'select', 'meanmax050')
PRIMARY_COMPARISONS = (('sphere20', 'select'), ('rgb', 'select'), ('select', 'meanmax050'))
PRIMARY_OUTCOMES = ('DS', 'CS-H', 'CS-directional', 'MUSIQ')


def protocol_definitions():
    return dict(statistics=dict(unit='prompt cluster', panorama='average relevant views first',
        seeds='average panoramas/seeds within each prompt', prompts='equal weight',
        bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
        interval='percentile 2.5/97.5', paired='same sampled prompt IDs for both methods',
        missing='excluded with explicit common prompt IDs and coverage; never zero',
        primary_comparisons=PRIMARY_COMPARISONS, primary_outcomes=PRIMARY_OUTCOMES,
        difference='comparison minus baseline'))


def prompt_values(rows, metric):
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        value = row.get('metrics', {}).get(metric, row.get(metric))
        if value is None:
            continue
        value = float(value)
        if not np.isfinite(value):
            raise ValueError('Nonfinite metric value: '+metric)
        identity = (row['method_id'], row['prompt_id'], row.get('case_id', row.get('panorama_id')))
        if identity in seen:
            raise ValueError('Duplicate panorama in prompt aggregation: '+str(identity))
        seen.add(identity)
        grouped[(row['method_id'], row['prompt_id'])].append(value)
    return {key:float(np.mean(values)) for key, values in grouped.items()}


def bootstrap_mean(values, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('A nonempty finite vector is required')
    rng = np.random.RandomState(seed)
    samples = values[rng.randint(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    interval = np.percentile(samples, [2.5, 97.5]).tolist() if len(values) >= 2 else [None, None]
    return dict(mean=float(values.mean()), ci95=interval, prompt_count=len(values),
                bootstrap_replicates=replicates if len(values) >= 2 else 0, bootstrap_seed=seed,
                uncertainty_status='computed' if len(values) >= 2 else 'insufficient_prompt_clusters')


def summarize(rows, metric, cohort_id, prompt_ids, comparisons=PRIMARY_COMPARISONS):
    allowed = set(prompt_ids)
    selected = [row for row in rows if row['prompt_id'] in allowed]
    values = prompt_values(selected, metric)
    groups, pairs = [], []
    for method in sorted({row['method_id'] for row in selected}):
        present = sorted(p for m, p in values if m == method)
        group = dict(method_id=method, metric=metric, cohort_id=cohort_id,
            prompt_ids=present, missing_prompt_ids=sorted(allowed-set(present)),
            panorama_count=sum(row['method_id'] == method and row['prompt_id'] in present for row in selected),
            per_prompt=[dict(prompt_id=p, value=values[(method, p)]) for p in present])
        if present:
            group.update(bootstrap_mean([values[(method, p)] for p in present]))
        else:
            group.update(mean=None, ci95=[None, None], prompt_count=0, uncertainty_status='missing')
        groups.append(group)
    for baseline, comparison in comparisons:
        common = sorted(p for p in allowed if (baseline, p) in values and (comparison, p) in values)
        pair = dict(baseline=baseline, comparison=comparison, metric=metric, cohort_id=cohort_id,
            prompt_ids=common, missing_prompt_ids=sorted(allowed-set(common)),
            scope='primary' if metric in PRIMARY_OUTCOMES else 'exploratory',
            difference='comparison minus baseline')
        if common:
            deltas = [values[(comparison, p)]-values[(baseline, p)] for p in common]
            pair.update(bootstrap_mean(deltas), baseline_mean=float(np.mean([values[(baseline,p)] for p in common])),
                comparison_mean=float(np.mean([values[(comparison,p)] for p in common])),
                per_prompt=[dict(prompt_id=p, difference=d) for p,d in zip(common,deltas)])
        else:
            pair.update(mean=None, ci95=[None,None], prompt_count=0, uncertainty_status='missing_common_cohort')
        pairs.append(pair)
    return groups, pairs
