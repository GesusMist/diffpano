"""Resume saved native-reference bootstrap draws using the frozen estimator.

This orchestration adapter never recalculates completed contrasts. It appends to
saved draws in the original RandomState order and publishes the same percentile
interval only once all predeclared replicates exist. Scientific definitions stay
in distribution.py; interrupted continuations retain a checkpoint per contrast.
"""
import argparse
import contextlib
import copy
import fcntl
import json
import os
import time
from pathlib import Path

import numpy as np
from .common import RUN, csv_write, digest, freeze, now, read, sha, update_component, write
from .distribution import frechet, kid
from .statistics import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED

KEYS = ('namespace', 'reference_kind', 'bank', 'cohort_id', 'baseline', 'comparison', 'metric')


def pair_key(pair):
    return digest({key: pair.get(key) for key in KEYS})


def target(pair):
    return (pair.get('reference_kind') == 'native'
            and pair.get('namespace') == 'native_five_v2'
            and pair.get('metric') in ('FID', 'KID')
            and pair.get('state') == 'pending_uncertainty_compute'
            and len(pair.get('prompt_ids', [])) >= 2
            and pair.get('achieved_replicates', 0) < pair.get('requested_replicates', 0))


def validate_saved(pair):
    draws = pair.get('replicates', [])
    if (len(draws) != pair.get('achieved_replicates')
            or not 0 <= len(draws) <= pair['requested_replicates']
            or not np.isfinite(draws).all()):
        raise ValueError('Saved replicate count or finite values are invalid')
    if pair.get('state') == 'complete' and len(draws) != pair['requested_replicates']:
        raise ValueError('Completed contrast is missing replicates')
    return draws


def longest_prefix(original, candidates):
    """Accept only append-only progression from an immutable original record."""
    best = original
    original_draws = validate_saved(original)
    for candidate in candidates:
        if candidate is None:
            continue
        if (pair_key(candidate) != pair_key(original)
                or candidate['prompt_ids'] != original['prompt_ids']
                or candidate['requested_replicates'] != original['requested_replicates']
                or candidate['bootstrap_seed'] != original['bootstrap_seed']):
            raise ValueError('Contrast identity changed during continuation')
        draws = validate_saved(candidate)
        if draws[:len(original_draws)] != original_draws:
            raise ValueError('Saved bootstrap prefix changed')
        prior = best['replicates']
        if draws[:min(len(draws), len(prior))] != prior[:min(len(draws), len(prior))]:
            raise ValueError('Published and checkpoint bootstrap draws disagree')
        if len(draws) > len(prior):
            best = candidate
    return copy.deepcopy(best)


def resume_linked(pair, first, second, reference, *, deadline=None, checkpoint=None,
                  checkpoint_every=25, clock=time.monotonic):
    """Same linked-reference calculation and prompt draws as paired_bootstrap."""
    result = copy.deepcopy(pair)
    draws = list(validate_saved(result))
    if len(draws) == result['requested_replicates']:
        return result
    ids = sorted(set(first) & set(second) & set(reference))
    if ids != result['prompt_ids'] or len(ids) < 2:
        raise ValueError('Matched prompt clusters differ from the saved estimator')
    if result['metric'] not in ('FID', 'KID'):
        raise ValueError('Only frozen linked-reference FID/KID are supported')
    score = frechet if result['metric'] == 'FID' else lambda x, y: kid(x, y)['mean']

    def calculate(chosen):
        x = np.concatenate([first[p] for p in chosen])
        z = np.concatenate([second[p] for p in chosen])
        y = np.concatenate([reference[p] for p in chosen])
        return float(score(z, y) - score(x, y))

    started = clock()
    observed = calculate(ids)
    if not np.isclose(observed, result['mean_difference'], rtol=1e-9, atol=1e-10):
        raise ValueError('Frozen observed contrast differs from the saved result')
    rng = np.random.RandomState(result['bootstrap_seed'])
    # Replay only the cheap RNG steps, checking the numerical first/last saved
    # draws to catch altered feature inputs before any new draws are appended.
    for index, value in enumerate(draws):
        chosen = [ids[i] for i in rng.randint(len(ids), size=len(ids))]
        if index in (0, len(draws)-1) and not np.isclose(calculate(chosen), value, rtol=1e-9, atol=1e-10):
            raise ValueError('Frozen estimator does not reproduce the saved bootstrap prefix')
    initial_count = len(draws)
    old_seconds = float(result.get('seconds', 0.))

    def snapshot():
        complete = len(draws) == result['requested_replicates']
        result.update(state='complete' if complete else 'pending_uncertainty_compute',
            replicates=list(draws), achieved_replicates=len(draws),
            ci95=np.percentile(draws, [2.5, 97.5]).tolist() if complete else None,
            partial_percentiles=np.percentile(draws, [2.5, 97.5]).tolist() if draws and not complete else None,
            seconds=old_seconds + clock()-started)
        if checkpoint is not None:
            checkpoint(copy.deepcopy(result))
        return result

    for unused in range(initial_count, result['requested_replicates']):
        if deadline is not None and clock() >= deadline:
            break
        chosen = [ids[i] for i in rng.randint(len(ids), size=len(ids))]
        draws.append(calculate(chosen))
        if (len(draws)-initial_count) % checkpoint_every == 0:
            snapshot()
    return snapshot()


def pair_inputs(run, pair, all_rows, feature_cache):
    from .metrics import load_features
    ids = pair['prompt_ids']
    selected = [r for r in all_rows if r.get('namespace') == 'common_v2'
                and r.get('prompt_id') in ids and r.get('method_id') in (pair['baseline'], pair['comparison'])
                and 'feature_path' in r]
    index_path = run/'group2'/('native_'+pair['bank']+'_features.json')
    index = read(index_path)
    if not index.get('complete'):
        raise ValueError('Native reference feature index is incomplete')
    reference_rows = {p: sorted([r for r in index['records'] if r['prompt_id'] == p],
                               key=lambda r: r['direction_index']) for p in ids}
    if any([r['direction_index'] for r in values] != list(range(5)) for values in reference_rows.values()):
        raise ValueError('Native reference direction slots changed')
    descriptor = dict(native_index_sha256=sha(index_path),
        generated=[{k:r[k] for k in ('case_id', 'prompt_id', 'method_id', 'feature_path', 'feature_sha256')} for r in selected],
        reference=[{k:r[k] for k in ('prompt_id', 'direction_index', 'feature_path', 'feature_sha256')}
                   for p in ids for r in reference_rows[p]])

    def feature(row):
        key = (row['feature_path'], row['feature_sha256'])
        if key not in feature_cache:
            feature_cache[key] = load_features(row)['native_five']
        return feature_cache[key]

    first, second = {}, {}
    for method, destination in ((pair['baseline'], first), (pair['comparison'], second)):
        for p in ids:
            values = [feature(r) for r in selected if r['method_id'] == method and r['prompt_id'] == p]
            if not values:
                raise ValueError('Generated features missing for a saved matched prompt')
            destination[p] = np.concatenate(values)
    reference = {p: np.concatenate([feature(r) for r in reference_rows[p]]) for p in ids}
    if any(values.shape != (5, 2048) for values in reference.values()):
        raise ValueError('Reference must retain the frozen five full-dimensional views')
    return first, second, reference, descriptor


@contextlib.contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another Group2 uncertainty continuation is active') from exc
        yield


def execute(run=RUN, max_seconds=10000.):
    from .cli import verify_protocol
    run = Path(run)
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run bootstrap continuation in a finite Slurm CPU allocation')
    if not 0 < max_seconds <= 10000:
        raise ValueError('Continuation is bounded to at most10000 seconds per allocation')
    registry = verify_protocol(run)
    directory = run/'group2/uncertainty_resume'
    with exclusive(directory/'worker.lock'):
        initial_path = directory/'original_paired_comparisons.json'
        original_status_path = directory/'original_status.json'
        if not initial_path.exists():
            freeze(initial_path, read(run/'group2/paired_comparisons.json'))
        if not original_status_path.exists():
            freeze(original_status_path, read(run/'group2/status.json'))
        originals = read(initial_path)
        original_by_key = {pair_key(row):row for row in originals}
        selected = [row for row in originals if target(row)]
        if any(row['requested_replicates'] != BOOTSTRAP_REPLICATES or row.get('bootstrap_seed') != BOOTSTRAP_SEED for row in selected):
            raise ValueError('Saved uncertainty parameters differ from the frozen protocol')
        all_rows = read(run/'group1/per_panorama_metrics.json')
        feature_cache, inputs = {}, {}
        for pair in selected:
            key = pair_key(pair)
            inputs[key] = pair_inputs(run, pair, all_rows, feature_cache)
        manifest = dict(original_pair_file_sha256=sha(initial_path), original_status_file_sha256=sha(original_status_path),
            registry_sha256=sha(run/'protocol_registry.json'), scientific_sources=registry['source_hashes'],
            helper_source_sha256=sha(__file__), targets=[dict(key=pair_key(row), inputs=inputs[pair_key(row)][3]) for row in selected],
            seed=BOOTSTRAP_SEED, requested_replicates=BOOTSTRAP_REPLICATES,
            preservation='Every completed contrast and every saved partial bootstrap prefix remain unchanged; only missing linked-reference draws are appended.')
        freeze(directory/'manifest.json', manifest)
        deadline = time.monotonic()+max_seconds
        completed = 0
        appended = 0
        for pair in selected:
            key = pair_key(pair)
            current = read(run/'group2/paired_comparisons.json')
            published = {pair_key(row):row for row in current}
            if len(published) != len(current) or set(published) != set(original_by_key):
                raise ValueError('Group2 contrast set changed; explicit reconciliation required')
            checkpoint_path = directory/'checkpoints'/(key+'.json')
            checkpoint_record = read(checkpoint_path) if checkpoint_path.exists() else None
            if checkpoint_record is not None and checkpoint_record['inputs_sha256'] != digest(inputs[key][3]):
                raise ValueError('Checkpoint features differ from the immutable continuation inputs')
            saved = longest_prefix(pair, [published[key], checkpoint_record['result'] if checkpoint_record else None])
            before = saved['achieved_replicates']

            def save(result):
                write(checkpoint_path, dict(updated_utc=now(), job_id=os.environ['SLURM_JOB_ID'],
                    inputs_sha256=digest(inputs[key][3]), result=result))

            first, second, reference, unused = inputs[key]
            result = resume_linked(saved, first, second, reference, deadline=deadline, checkpoint=save)
            # Detect a conflicting writer before publishing; read fresh status so
            # unrelated fields added by reporting or orchestration are preserved.
            latest = read(run/'group2/paired_comparisons.json')
            if digest(latest) != digest(current):
                raise RuntimeError('Concurrent Group2 writer changed published comparisons')
            replacement = [result if pair_key(row) == key else row for row in latest]
            status = read(run/'group2/status.json')
            status.update(paired_comparisons=replacement,
                state='partial' if status.get('blockers') or any(row['state'] != 'complete' for row in replacement) else 'complete')
            write(run/'group2/paired_comparisons.json', replacement)
            csv_write(run/'group2/paired_comparisons.csv', replacement, sorted({field for row in replacement for field in row}))
            write(run/'group2/status.json', status)
            completed += result['state'] == 'complete'
            appended += result['achieved_replicates']-before
            progress = dict(state='complete' if completed == len(selected) else 'partial', targets=len(selected),
                completed_targets=completed, appended_this_job=appended, job_id=os.environ['SLURM_JOB_ID'],
                updated_utc=now(), manifest_sha256=sha(directory/'manifest.json'))
            write(directory/'status.json', progress)
            print(json.dumps(dict(contrast={k:pair.get(k) for k in KEYS}, achieved=result['achieved_replicates'], requested=result['requested_replicates'], state=result['state'])), flush=True)
            if time.monotonic() >= deadline:
                break
        # Count all checkpoints/publications again so restart status is accurate.
        latest = read(run/'group2/paired_comparisons.json')
        published = {pair_key(row):row for row in latest}
        remaining = [pair_key(row) for row in selected if published[pair_key(row)]['state'] != 'complete']
        result = dict(state='complete' if not remaining else 'partial', targets=len(selected), completed_targets=len(selected)-len(remaining),
            remaining_contrast_ids=remaining, appended_this_job=appended, updated_utc=now(), job_id=os.environ['SLURM_JOB_ID'],
            manifest_sha256=sha(directory/'manifest.json'), external_asset_blockers_preserved=True)
        write(directory/'status.json', result)
        verify_protocol(run)
        update_component(run, 'group2_uncertainty_resume', result['state'], result=result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--max-seconds', type=float, default=10000.)
    args = parser.parse_args()
    print(json.dumps(execute(args.run, args.max_seconds), indent=2))


if __name__ == '__main__':
    main()
