"""Independent inventory → views → offline plan → explicitly authorized API run."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.request

from .io_utils import read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path('outputs/10.16spherediff-vlm-eval/seed0-v1')


def resolve_output(root, argument):
    path = Path(argument).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def prepare_output(output):
    """Require a study marker before reusing any nonempty output directory."""
    output = Path(output)
    marker = output / 'study.json'
    if marker.exists():
        if read_json(marker).get('study') != 'spherediff_vlm_eval':
            raise ValueError('Output directory belongs to a different study')
    else:
        existing = {p.name for p in output.iterdir()} if output.exists() else set()
        # inventory.py is independently executable; its own audit proves ownership.
        own_inventory = False
        if (output / 'inventory.json').exists() and (output / 'provenance_audit.json').exists():
            candidate = read_json(output / 'inventory.json')
            own_inventory = (candidate.get('schema_version') == 1 and
                             candidate.get('method_ids') == ['sphere20', 'rgb', 'select', 'meanmax050', 'meanmax025', 'meanmax075', 'sphere28'] and
                             'cohorts' in candidate and 'records' in candidate)
        if existing - {'startup_preservation.json'} and not own_inventory:
            raise ValueError('Refusing to write into an existing non-study directory; choose a new version')
        output.mkdir(parents=True, exist_ok=True)
        write_json(marker, dict(study='spherediff_vlm_eval', version=1,
                               created_utc=datetime.now(timezone.utc).isoformat(), paid_execution_default=False))
    return output


@contextmanager
def output_lock(output):
    with (output / '.study.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another study command is writing this output directory') from None
        yield


def save_run_configuration(output, args, root):
    path = output / 'run_configuration.json'
    record = dict(root=str(root), historical_roots=[str(Path(p).expanduser().resolve()) for p in args.historical_root],
                  path_remappings=dict(item.split('=', 1) for item in args.path_map))
    if path.exists():
        if read_json(path) != record:
            raise ValueError('Source roots/remappings changed; use a fresh versioned output directory')
    else:
        write_json(path, record)
    return record


def preservation_check(root, output):
    """Verify the recorded repository and all read historical evidence hashes."""
    checks, errors = 0, []
    startup = output / 'startup_preservation.json'
    if startup.exists():
        record = read_json(startup)
        for relative, expected in record['files'].items():
            path = root / relative
            checks += 1
            if not path.exists() or sha256(path) != expected:
                errors.append(str(path))
        if (root / 'outputs').is_symlink() and os.readlink(root / 'outputs') != record['outputs_symlink']:
            errors.append('outputs symlink changed')
    audit = output / 'provenance_audit.json'
    if audit.exists():
        for filename, expected in read_json(audit).get('source_files', {}).items():
            checks += 1
            path = Path(filename)
            if not path.exists() or sha256(path) != expected:
                errors.append(filename)
    result = dict(passed=not errors, checked_files=checks, changed=errors,
                  checked_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'preservation_check.json', result)
    if errors:
        raise ValueError('Historical source preservation failed: ' + ', '.join(errors[:10]))
    return result


def environment_check(output, network=False):
    """No paid request; optional unauthenticated HEAD checks API reachability."""
    previous_path = output / 'environment_preflight.json'
    previous = read_json(previous_path) if previous_path.exists() else {}
    result = dict(credential_available=bool(os.environ.get('OPENAI_API_KEY', '').strip()),
                  credential_source='OPENAI_API_KEY environment variable; value never recorded',
                  network_access=previous.get('network_access', 'not_checked'), model_access='not_checked', paid_requests=0)
    if not network:
        result.update({key: previous[key] for key in ('unauthenticated_http_status', 'error_type') if key in previous})
    if network:
        request = urllib.request.Request('https://api.openai.com/v1/models', method='HEAD')
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                result.update(network_access='reachable', unauthenticated_http_status=response.status)
        except urllib.error.HTTPError as error:
            result.update(network_access='reachable', unauthenticated_http_status=error.code)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            result.update(network_access='unavailable_from_this_process', error_type=type(error).__name__)
    write_json(output / 'environment_preflight.json', result)
    return result


def freeze_implementation(output):
    """Preserve the exact uncommitted evaluator as well as paper/render identities."""
    study = Path(__file__).resolve().parent
    files = {str(path.relative_to(study)): sha256(path) for path in sorted(study.rglob('*'))
             if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.txt', '.json', '.md', '.png')}
    path = output / 'implementation.json'
    if path.exists():
        if read_json(path)['files'] != files:
            raise ValueError('Evaluator implementation changed after freeze; choose a fresh output version')
    else:
        write_json(path, dict(files=files, frozen_utc=datetime.now(timezone.utc).isoformat()))
        import tarfile
        with tarfile.open(output / 'implementation.tar.gz', 'w:gz') as archive:
            for relative in files:
                archive.add(study / relative, arcname='studies/spherediff_vlm_eval/' + relative)
    return files


def select_views(inventory, views, scope):
    names = list(inventory['cohorts']) if scope == 'all' else ['primary', 'alpha025', 'alpha075'] if scope == 'ablations' else [scope]
    identifiers = {p for name in names for p in inventory['cohorts'][name]['panorama_ids']}
    return [row for row in views if row['panorama_id'] in identifiers]


def judge_configuration(output, args, protocol):
    from .protocol import digest
    config = dict(model=args.model, image_detail=args.image_detail,
                  api_parameters=dict(temperature=0, top_p=1, seed=args.judge_seed, max_tokens=args.max_output_tokens),
                  protocol_hash=digest(protocol), max_retries=args.max_retries)
    path = output / 'judge_configuration.json'
    if path.exists() and read_json(path) != config:
        raise ValueError('Judge configuration changed; use a fresh versioned output to avoid mixing protocols')
    if not path.exists():
        write_json(path, config)
    return config


def plan_counts(inventory, views, config):
    from .client import cache_identity, cost_estimate
    primary_keys = {cache_identity(row, config)[0] for row in select_views(inventory, views, 'primary')}
    result = {}
    for name, cohort in inventory['cohorts'].items():
        subset = select_views(inventory, views, name)
        keys = {cache_identity(row, config)[0] for row in subset}
        result[name] = dict(matched_prompts=len(cohort['prompt_ids']), panoramas=len(cohort['panorama_ids']),
                            view_records=len(subset), unique_requests=len(keys),
                            additional_requests_after_primary=len(keys - primary_keys),
                            cost_estimate=cost_estimate(config, len(keys)),
                            additional_cost_after_primary=cost_estimate(config, len(keys - primary_keys)))
    ablation_keys = {cache_identity(row, config)[0] for row in select_views(inventory, views, 'ablations')}
    result['combined_ablations'] = dict(additional_requests_after_primary=len(ablation_keys - primary_keys),
                                        additional_cost_after_primary=cost_estimate(config, len(ablation_keys - primary_keys)))
    return result


def report(output, inventory, views, config, args, evaluation=None):
    from .aggregate import aggregate
    from .client import evaluate_views
    from .protocol import CAMERAS, protocol_record
    from .report import write_reports
    # Re-read all cached scopes offline, so reports retain earlier successful cohorts.
    all_scores = evaluate_views(views, output, config, dry_run=True,
                               max_requests=args.max_requests, budget_usd=args.budget_usd)
    aggregated = aggregate(inventory, all_scores['records'], cameras=CAMERAS,
                           bootstrap_draws=args.bootstrap_draws, bootstrap_seed=args.bootstrap_seed)
    preflight_path = output / 'environment_preflight.json'
    preflight = read_json(preflight_path) if preflight_path.exists() else None
    write_reports(output, inventory, all_scores['records'], aggregated, protocol=protocol_record(),
                  evaluation=evaluation or all_scores, preflight=preflight)
    return aggregated


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('inventory', 'prepare', 'dry-run', 'evaluate', 'report', 'check-environment', 'verify-preservation'))
    p.add_argument('--root', type=Path, default=ROOT, help='Current checkout, defaults to the module location')
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='New versioned directory, relative to checkout unless absolute')
    p.add_argument('--historical-root', action='append', default=[], help='Additional relocated outputs root; repeatable')
    p.add_argument('--path-map', action='append', default=[], metavar='OLD=NEW', help='Explicit historical path-prefix remapping')
    p.add_argument('--scope', choices=('primary', 'alpha025', 'alpha075', 'ablations', 'sphere28', 'all'), default='primary')
    p.add_argument('--model', default='gpt-4o')
    p.add_argument('--image-detail', choices=('high', 'low', 'auto'), default='high')
    p.add_argument('--judge-seed', type=int, default=0)
    p.add_argument('--max-output-tokens', type=int, default=700)
    p.add_argument('--max-retries', type=int, default=3)
    p.add_argument('--budget-usd', type=float, default=0, help='Hard cumulative request cost guard; zero by default')
    p.add_argument('--max-requests', type=int, default=0, help='Hard cumulative actual-attempt guard including retries')
    p.add_argument('--execute', action='store_true', help='Explicitly enable paid requests, only for evaluate')
    p.add_argument('--network', action='store_true', help='Unauthenticated HEAD probe, only for check-environment')
    p.add_argument('--bootstrap-draws', type=int, default=10000)
    p.add_argument('--bootstrap-seed', type=int, default=20261016)
    p.add_argument('--no-figures', action='store_true', help='Skip human-facing figure assembly during prepare')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.execute and args.command != 'evaluate':
        raise ValueError('--execute is valid only for the evaluate command')
    if args.network and args.command != 'check-environment':
        raise ValueError('--network is valid only for check-environment')
    root = args.root.expanduser().resolve()
    output = prepare_output(resolve_output(root, args.output))
    with output_lock(output):
        if args.command == 'check-environment':
            print(json.dumps(environment_check(output, args.network), indent=2))
            return
        if args.command == 'verify-preservation':
            print(json.dumps(preservation_check(root, output), indent=2))
            return
        if args.command in ('inventory', 'prepare'):
            run = save_run_configuration(output, args, root)
            if not (output / 'inventory.json').exists():
                from .inventory import build_inventory
                inventory = build_inventory(root, output, run['historical_roots'], run['path_remappings'])
            else:
                inventory = read_json(output / 'inventory.json')
                print('RESUME frozen inventory; use a new output version to change cohort membership', flush=True)
            if args.command == 'inventory':
                print(json.dumps(inventory.get('counts', {}), indent=2))
                return
            from .preprocessing import preprocess
            views = preprocess(inventory['records'], output)
            write_json(output / 'views.json', views)
            if not args.no_figures:
                from .protocol import CAMERAS
                from .report import make_figures
                figures = make_figures(output, inventory, views, cameras=CAMERAS)
                print('FIGURES', len(figures), flush=True)
            environment_check(output, network=False)
        else:
            inventory = read_json(output / 'inventory.json')
            views = read_json(output / 'views.json')
        from .client import evaluate_views
        from .protocol import protocol_record
        freeze_implementation(output)
        environment_check(output, network=False)
        protocol = protocol_record()
        config = judge_configuration(output, args, protocol)
        selected = select_views(inventory, views, args.scope)
        if args.execute:
            preservation_check(root, output)
        execution = evaluate_views(selected, output, config, dry_run=not args.execute,
                                   max_requests=args.max_requests, budget_usd=args.budget_usd)
        plan = {key: value for key, value in execution.items() if key != 'records'}
        plan.update(scope=args.scope, cohort_plans=plan_counts(inventory, views, config),
                    selected_panorama_count=len({row['panorama_id'] for row in selected}),
                    recorded_utc=datetime.now(timezone.utc).isoformat())
        plan_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        write_json(output / 'execution_plans' / (plan_stamp + '.json'), plan)
        write_json(output / 'execution_plan.json', plan)
        aggregated = report(output, inventory, views, config, args, execution)
        if args.command == 'prepare':
            print('PRESERVATION', json.dumps(preservation_check(root, output)), flush=True)
        print(json.dumps(dict(output=str(output), scope=args.scope,
                              pending_requests=execution['pending_unique_requests'],
                              estimated_cost_usd=execution['estimated_cost_usd'],
                              conservative_max_cost_usd=execution['conservative_max_cost_usd'],
                              requests_attempted=execution['requests_attempted'],
                              primary=aggregated['cohorts']['primary']), indent=2))


if __name__ == '__main__':
    main()
