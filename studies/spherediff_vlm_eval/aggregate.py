"""Equal-weight panorama averages and paired, prompt-level uncertainty."""
from collections import Counter, defaultdict
import math
from statistics import mean

import numpy as np

CRITERIA = ('image_quality', 'aesthetic_appeal', 'distortion', 'connectivity')
METHOD_LABELS = {
    'sphere20': 'SphereDiff official FLUX20',
    'rgb': 'DiffPano RGB Old89',
    'select': 'DiffPano Select + lambda_blending Old89',
    'meanmax050': 'DiffPano Mean-Max + lambda_blending Old89, alpha=0.50',
    'meanmax025': 'DiffPano Mean-Max + lambda_blending Old89, alpha=0.25',
    'meanmax075': 'DiffPano Mean-Max + lambda_blending Old89, alpha=0.75',
    'sphere28': 'SphereDiff official FLUX28 (additional baseline)',
}
PRIMARY_METHODS = ('sphere20', 'rgb', 'select', 'meanmax050')


def valid_scores(row):
    scores = row.get('scores') or {}
    return row.get('status') == 'success' and set(scores) == set(CRITERIA) and all(
        type(scores[k]) is int and 1 <= scores[k] <= 5 for k in CRITERIA)


def paired_bootstrap(differences, *, draws=10000, seed=20261016):
    """Resample matched prompt differences, never individual camera views."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite paired prompt differences')
    if len(values) < 2:
        return None, None
    if draws < 100:
        raise ValueError('Use at least 100 bootstrap replicates')
    rng = np.random.default_rng(seed)
    estimates = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    low, high = np.quantile(estimates, [.025, .975])
    return float(low), float(high)


def aggregate(inventory, view_records, *, cameras, bootstrap_draws=10000, bootstrap_seed=20261016):
    """Never replace missing ratings with zero or compare unequal prompt sets."""
    expected = {camera['view_id'] for camera in cameras}
    if len(expected) != 14:
        raise ValueError('The frozen evaluation protocol must contain fourteen cameras')
    by_panorama = defaultdict(dict)
    for row in view_records:
        key = row['view_id']
        if key not in expected:
            raise ValueError('Unexpected evaluation camera: ' + key)
        group = by_panorama[row['panorama_id']]
        if key in group:
            raise ValueError('Duplicate panorama/view score: ' + row['panorama_id'] + '/' + key)
        group[key] = row
    panoramas, panorama_lookup, diagnostics = [], {}, []
    camera_lookup = {row['view_id']: row for row in cameras}
    for source in inventory['records']:
        rows = by_panorama[source['panorama_id']]
        if any(row.get('image_sha256') != source['image_sha256'] for row in rows.values()):
            raise ValueError('Scores refer to a different source panorama')
        successful = {key: row for key, row in rows.items() if valid_scores(row)}
        statuses = Counter(row.get('status', 'pending') for row in rows.values())
        complete = set(successful) == expected
        models = sorted({row.get('resolved_model') for row in successful.values() if row.get('resolved_model')})
        if len(models) > 1:
            complete = False
        record = {key: source.get(key) for key in ('panorama_id', 'method_id', 'prompt_id', 'prompt_name', 'image_sha256')}
        record.update(status='complete' if complete else 'mixed_models' if len(models) > 1 else 'incomplete',
                      expected_views=14, successful_views=len(successful),
                      missing_views=14 - len(successful), failed_views=statuses['failed'],
                      blocked_views=statuses['blocked'], resolved_models=models)
        record.update({criterion: mean(row['scores'][criterion] for row in successful.values()) if complete else None
                       for criterion in CRITERIA})
        panoramas.append(record)
        identity = (source['method_id'], source['prompt_id'])
        if identity in panorama_lookup:
            raise ValueError('Inventory contains multiple selected panoramas for a method/prompt')
        panorama_lookup[identity] = record
        groups = defaultdict(set)
        for key, camera in camera_lookup.items():
            groups[camera['elevation_group']].add(key)
            if abs(camera['pitch']) == 90:
                groups['polar_combined'].add(key)
            if camera['is_seam']:
                groups['seam_centered'].add(key)
        for group_name, keys in groups.items():
            group_complete = keys <= set(successful) and len(models) <= 1
            diag = dict(record, group=group_name, expected_views=len(keys),
                        successful_views=len(keys & set(successful)), status='complete' if group_complete else 'incomplete')
            diag.update({criterion: mean(successful[key]['scores'][criterion] for key in keys) if group_complete else None
                         for criterion in CRITERIA})
            diagnostics.append(diag)

    summaries, comparisons, rankings, diagnostic_summaries, cohort_states = [], [], [], [], {}
    for name, cohort in inventory['cohorts'].items():
        methods = cohort['method_ids']
        prompts = sorted(cohort['prompt_ids'])
        complete_prompts = [p for p in prompts if all(
            panorama_lookup.get((m, p), {}).get('status') == 'complete' for m in methods)]
        # An alias moving to a new snapshot must not silently mix judges.
        snapshots = sorted({snapshot for p in complete_prompts for m in methods
                            for snapshot in panorama_lookup[m, p]['resolved_models']})
        if len(snapshots) > 1:
            complete_prompts = []
            state = 'mixed_models'
        else:
            state = ('complete' if prompts and len(complete_prompts) == len(prompts)
                     else 'preliminary' if complete_prompts else 'not_evaluated' if prompts else 'no_matched_inputs')
        cohort_states[name] = dict(available_prompts=len(prompts), evaluated_prompts=len(complete_prompts),
                                  status=state, evaluated_prompt_ids=complete_prompts, resolved_models=snapshots)
        for method in methods:
            summary = dict(cohort=name, method_id=method, method=METHOD_LABELS.get(method, method),
                           n_prompts=len(complete_prompts), available_prompts=len(prompts), status=state)
            summary.update({c: mean(panorama_lookup[method, p][c] for p in complete_prompts) if complete_prompts else None
                            for c in CRITERIA})
            summaries.append(summary)
        baselines = [m for m in ('sphere20', 'rgb') if m in methods]
        if name.startswith('alpha') and 'meanmax050' in methods:
            baselines.append('meanmax050')
        for baseline in baselines:
            for method in methods:
                if method == baseline:
                    continue
                for c in CRITERIA:
                    delta = [panorama_lookup[method, p][c] - panorama_lookup[baseline, p][c] for p in complete_prompts]
                    ci = paired_bootstrap(delta, draws=bootstrap_draws, seed=bootstrap_seed) if delta else (None, None)
                    comparisons.append(dict(cohort=name, method_id=method, baseline=baseline, criterion=c,
                                            n_prompts=len(delta), available_prompts=len(prompts), status=state,
                                            mean_difference=mean(delta) if delta else None,
                                            ci95_low=ci[0], ci95_high=ci[1], bootstrap_unit='prompt',
                                            bootstrap_draws=bootstrap_draws, bootstrap_seed=bootstrap_seed,
                                            uncertainty_note='No CI with fewer than two prompts' if len(delta) < 2 else 'Percentile paired bootstrap; small cohorts may give unstable intervals'))
        for prompt in prompts:
            for criterion in CRITERIA:
                values = {m: panorama_lookup.get((m, prompt), {}).get(criterion) for m in methods}
                comparable = prompt in complete_prompts
                for method, value in values.items():
                    rank = 1 + sum(v > value for v in values.values()) if comparable else None
                    rankings.append(dict(cohort=name, prompt_id=prompt, method_id=method, criterion=criterion,
                                         score=value if comparable else None, rank=rank,
                                         status='complete' if comparable else 'incomplete', ties='competition ranking'))
        # Reuse the same complete prompt cohort; groups never alter paid requests.
        for group_name in sorted({row['group'] for row in diagnostics}):
            for method in methods:
                group_rows = [row for row in diagnostics if row['method_id'] == method and
                              row['prompt_id'] in complete_prompts and row['group'] == group_name]
                summary = dict(cohort=name, method_id=method, group=group_name,
                               n_prompts=len(group_rows), available_prompts=len(prompts), status=state)
                summary.update({c: mean(row[c] for row in group_rows) if group_rows else None for c in CRITERIA})
                diagnostic_summaries.append(summary)
    alpha_rows = [r for r in summaries if r['method_id'].startswith('meanmax') and
                  (r['cohort'].startswith('alpha') or r['cohort'] == 'primary')]
    return dict(panoramas=panoramas, summaries=summaries, comparisons=comparisons, alpha_ablation=alpha_rows,
                diagnostics=diagnostics, diagnostic_summaries=diagnostic_summaries, rankings=rankings,
                cohorts=cohort_states, total_views_expected=14 * len(panoramas),
                successful_views=sum(r['successful_views'] for r in panoramas),
                failed_views=sum(r['failed_views'] for r in panoramas),
                blocked_views=sum(r['blocked_views'] for r in panoramas),
                missing_views=sum(r['missing_views'] for r in panoramas))
