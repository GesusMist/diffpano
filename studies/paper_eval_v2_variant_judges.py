"""Isolated extension of the frozen panorama judges to all >=10-image variants.

Uses the original ledger, lock, image identities, rubric and response parser.
The original production manifests and scientific modules remain untouched.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import statistics
import time

from studies.paper_eval_v2.common import ROOT, RUN, read, write, freeze, digest, sha, now, atomic_text, csv_write
from studies.paper_eval_v2 import vlm, vlm_protocol as vp

EXT = Path('/home/shig/paper-eval-v2/20261027-variant-judging-v1')
ORDER = ['sphere20', 'rgb', 'select', 'uniform50_rgb', 'uniform70_rgb',
         'uniform89_rgb', 'uniform89_select', 'mean', 'meanmax025', 'meanmax050',
         'meanmax075', 'select_coarse_color_rtol0.0001',
         'meanmax025_coarse_color_rtol0.0001', 'meanmax050_coarse_color_rtol0.0001',
         'meanmax075_coarse_color_rtol0.0001', 'select_refine05of28', 'select_refine08of28']
NAMES = dict(zip(ORDER, ['SphereDiff', 'Old89 RGB', 'Old89 Select', 'Uniform50 RGB',
    'Uniform70 RGB', 'Uniform89 RGB', 'Uniform89 Select', 'Old89 mean',
    r'Old89 mean/max $\alpha=0.25$', r'Old89 mean/max $\alpha=0.50$',
    r'Old89 mean/max $\alpha=0.75$', 'Old89 Select + CC',
    r'Old89 mean/max $\alpha=0.25$ + CC', r'Old89 mean/max $\alpha=0.50$ + CC',
    r'Old89 mean/max $\alpha=0.75$ + CC', r'Old89 Select + 20\% refinement',
    r'Old89 Select + 30\% refinement']))
NAMES['spherope'] = 'SpheRoPE (50 steps)'
METRICS = [('quality', 'image_quality', 'Q'), ('aesthetics', 'aesthetic_appeal', 'A'),
           ('distortion', 'geometric_distortion', 'G'),
           ('polar_distortion', 'geometric_distortion', 'PG'),
           ('wrap_continuity', 'local_continuity', 'WC'),
           ('all_view_continuity', 'local_continuity', 'AC')]


def verify():
    from studies.paper_eval_v2.cli import verify_protocol
    verify_protocol(RUN)
    frozen = read(RUN / 'vlm/pilot_protocol.json')
    assert read(RUN / 'vlm/pilot_result.json')['production_ready']
    assert frozen == read(RUN / 'vlm/production_protocol.json')
    for name, checksum in frozen['implementation_sha256'].items():
        assert sha(ROOT / 'studies/paper_eval_v2' / name) == checksum, name
    return frozen


def observations():
    with sqlite3.connect('file:' + str(RUN / 'api/ledger.sqlite3') + '?mode=ro', uri=True) as db:
        return {key: json.loads(value) for key, value in db.execute('SELECT request_id,value FROM observations')}


def prepare():
    from studies.paper_eval_v2.rendering import cache_views
    frozen = verify()
    cohorts = read(RUN / 'cohorts.json')
    allowed = set(cohorts['cohorts']['development24']['prompt_ids'])
    cells = [c for c in cohorts['selected_cells'] if c['prompt_id'] in allowed]
    counts = Counter(c['method_id'] for c in cells)
    assert set(ORDER) == {m for m, n in counts.items() if n >= 10}
    chosen = {c['case_id'] for c in cells if c['method_id'] in ORDER}
    records = [r for r in read(RUN / 'inventory/records.json') if r['case_id'] in chosen]
    assert len(records) == len(chosen) == 339
    assert all(r['state'] == 'complete' and r['eligible'] for r in records)
    EXT.mkdir(parents=True, exist_ok=True)
    (EXT / 'vlm').mkdir(exist_ok=True)
    for name in ('pilot_protocol.json', 'pilot_result.json', 'production_protocol.json'):
        freeze(EXT / 'vlm' / name, read(RUN / 'vlm' / name))
    api = EXT / 'api'
    if not api.exists():
        api.symlink_to((RUN / 'api').resolve(), target_is_directory=True)
    assert api.resolve() == (RUN / 'api').resolve()
    freeze(EXT / 'selection.json', dict(records=records, counts={m: counts[m] for m in ORDER},
        selection_rule='All pre-existing selected development24 methods with at least 10 complete panoramas; no score-based filtering.',
        protocol_sha256=frozen['protocol_sha256']))
    # An isolated render database avoids races with the main controller.
    existing = {v['case_id']: [] for v in read(RUN / 'vlm/views.json')['views']}
    for view in read(RUN / 'vlm/views.json')['views']:
        existing[view['case_id']].append(view)
    views = []
    for index, record in enumerate(sorted(records, key=lambda r: r['case_id'])):
        cached = existing.get(record['case_id'], [])
        if len(cached) == 14 and all(v['source_image_sha256'] == record['image_sha256'] for v in cached):
            views.extend(cached)
        else:
            views.extend(cache_views(EXT, [record]))
        if index % 20 == 0:
            print(json.dumps(dict(event='render_progress', panoramas=index+1, total=len(records))), flush=True)
    assert len(views) == 339 * 14
    for case in chosen:
        items = [v for v in views if v['case_id'] == case]
        assert len({v['view_id'] for v in items}) == 14
    repeats = [v for v in views if int(vp.digest({'image': v['image_sha256'], 'seed': 20261026})[:8], 16) < int(.05 * 2**32)]
    manifest = dict(views=views, cohort_prompt_ids=sorted(allowed),
        cohorts=[dict(cohort_id='development_available_variants', scope='exploratory',
            method_ids=ORDER, prompt_ids=sorted(allowed), case_ids=sorted(chosen))],
        repeat_view_identities=[dict(image_sha256=v['image_sha256'], view_id=v['view_id']) for v in repeats],
        selection_before_extension_scores=True, registry_sha256=sha(RUN / 'protocol_registry.json'))
    freeze(EXT / 'vlm/production_views.json', manifest)
    requests = vlm.request_records(views, frozen) + vlm.request_records(repeats, frozen, 'repeat-1')
    cached = observations()
    missing = [r for r in requests if r['request_id'] not in cached]
    summary = dict(created_utc=now(), panoramas=339, methods=len(ORDER), views=len(views),
        primary_requests=len(vlm.request_records(views, frozen)), planned_requests=len(requests),
        reused_valid_requests=len(requests)-len(missing), new_requests=len(missing),
        maximum_new_request_reservation_usd=str(sum(vlm.money(r['max_cost_usd']) for r in missing)),
        protocol_sha256=frozen['protocol_sha256'], manifest_sha256=digest(manifest),
        shared_ledger=str((RUN / 'api/ledger.sqlite3').resolve()),
        request_ids=[r['request_id'] for r in requests])
    freeze(EXT / 'preparation.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'request_ids'}), flush=True)
    export()


def export():
    frozen = read(EXT / 'vlm/pilot_protocol.json')
    manifest = read(EXT / 'vlm/production_views.json')
    cached = observations()
    # Confirmation rows use only scenes fully judged for all three methods/judges.
    core = read(RUN / 'vlm/production_views.json')
    confirmation = next(c for c in core['cohorts'] if c['cohort_id'] == 'confirmation24')
    confirm_cases = set(confirmation['case_ids'])
    views = manifest['views'] + [v for v in core['views'] if v['case_id'] in confirm_cases]
    grouped = defaultdict(list)
    for view in views:
        grouped[view['case_id']].append(view)
    panoramas = []
    valid_requests = 0
    for case, items in sorted(grouped.items()):
        assert len(items) == 14 and len({v['view_id'] for v in items}) == 14
        for model in vp.MODELS:
            scored = [(v, cached.get(vp.cache_identity(v, frozen['judges'][model], frozen)[0])) for v in items]
            valid_requests += sum(o is not None for v, o in scored)
            # All six metrics share the same complete-panorama denominator.
            if any(o is None for v, o in scored):
                continue
            metrics = {}
            for metric, criterion, _ in METRICS:
                selected = [(v, o) for v, o in scored if
                    (metric != 'polar_distortion' or v['is_polar']) and
                    (metric != 'wrap_continuity' or v['is_seam'])]
                assert len(selected) == {'polar_distortion': 2, 'wrap_continuity': 3}.get(metric, 14)
                metrics[metric] = statistics.mean(o['scores'][criterion] for v, o in selected)
            panoramas.append(dict(case_id=case, prompt_id=items[0]['prompt_id'], method_id=items[0]['method_id'],
                judge=model, cohort='confirmation' if case in confirm_cases else 'development', metrics=metrics))
    common = set(confirmation['prompt_ids'])
    for model in vp.MODELS:
        for method in ('sphere20', 'spherope', 'select'):
            common &= {p['prompt_id'] for p in panoramas if p['cohort']=='confirmation' and p['judge']==model and p['method_id']==method}
    rows = []
    for cohort, methods in [('development', ORDER), ('confirmation', ['sphere20','spherope','select'])]:
        for method in methods:
            planned = len({v['case_id'] for v in views if v['method_id']==method and (v['case_id'] in confirm_cases)==(cohort=='confirmation')})
            for model in vp.MODELS:
                subset = [p for p in panoramas if p['cohort']==cohort and p['method_id']==method and p['judge']==model
                          and (cohort!='confirmation' or p['prompt_id'] in common)]
                prompts = sorted({p['prompt_id'] for p in subset})
                assert len(prompts) == len(subset), 'Expected one selected seed per prompt'
                rows.append(dict(cohort=cohort, method_id=method, judge=model, completed=len(subset), planned=planned,
                    prompt_ids=prompts, **{metric:statistics.mean(p['metrics'][metric] for p in subset) if subset else None for metric,_,_ in METRICS}))
    write(EXT / 'scores.json', dict(updated_utc=now(), rows=rows, per_panorama=panoramas))
    csv_write(EXT / 'scores.csv', rows, ['cohort','method_id','judge','completed','planned']+[m[0] for m in METRICS]+['prompt_ids'])
    for filename, models in [('gpt4o',[vp.MODELS[0]]), ('gpt6sol',[vp.MODELS[1]]), ('gpt_combined',list(vp.MODELS))]:
        columns = 1 + 7*len(models)
        lines = [r'\begin{table*}[t]',r'\centering',
            r'\caption{Frozen-protocol GPT judge scores (1--5; higher is better). Each judge reports $n/N$: completely judged panoramas / available panoramas. Means use complete 14-view panoramas only. Development rows use available prompt subsets; partial rows are provisional and are not ranked. Confirmation rows use scenes complete for all three methods and both judges. Q: quality; A: aesthetics; G: geometry; PG: polar geometry; WC: wrap continuity; AC: all-view continuity. CC: coarse-color constraint.}',
            r'\label{tab:variants_'+filename+r'}',r'\resizebox{\textwidth}{!}{',r'\begin{tabular}{l'+'c'*(columns-1)+'}',r'\toprule',
            'Method & '+' & '.join(('GPT-4o' if model==vp.MODELS[0] else 'GPT-6 Sol')+' '+h for model in models for h in ['$n/N$']+[m[2]+r' $\uparrow$' for m in METRICS])+r' \\',r'\midrule']
        for cohort, methods in [('development',ORDER),('confirmation',['sphere20','spherope','select'])]:
            lines.append(r'\multicolumn{'+str(columns)+r'}{l}{\textit{'+cohort.title()+r'}} \\')
            for method in methods:
                cells=[]
                for model in models:
                    row=next(r for r in rows if r['cohort']==cohort and r['method_id']==method and r['judge']==model)
                    cells.append(str(row['completed'])+'/'+str(row['planned']))
                    cells.extend('--' if row[m] is None else '%.3f'%row[m] for m,_,_ in METRICS)
                lines.append(NAMES[method]+' & '+' & '.join(cells)+r' \\')
            lines.append(r'\midrule')
        lines[-1]=r'\bottomrule'
        lines += [r'\end{tabular}',r'}',r'\end{table*}','']
        atomic_text(EXT / ('flux_methods_'+filename+'.tex'), '\n'.join(lines))
    return rows


def run(max_hours):
    frozen=verify()
    manifest=read(EXT / 'vlm/production_views.json')
    preparation=read(EXT / 'preparation.json')
    assert digest(manifest)==preparation['manifest_sha256']
    assert frozen==read(EXT / 'vlm/pilot_protocol.json')
    assert (EXT/'api').resolve()==(RUN/'api').resolve()
    repeated={(v['image_sha256'],v['view_id']) for v in manifest['repeat_view_identities']}
    grouped=defaultdict(list)
    for v in manifest['views']: grouped[v['prompt_id']].append(v)
    started=time.monotonic()
    state='running'
    with vlm.worker_lock(RUN):
        for prompt, views in sorted(grouped.items()):
            requests=vlm.request_records(views,frozen)
            requests+=vlm.request_records([v for v in views if (v['image_sha256'],v['view_id']) in repeated],frozen,'repeat-1')
            cached=observations()
            if all(r['request_id'] in cached for r in requests): continue
            if time.monotonic()-started > max_hours*3600:
                state='checkpointed'; break
            block=digest(dict(stage='variant_extension_v1',prompt_id=prompt,protocol=frozen['protocol_sha256'],manifest_sha256=preparation['manifest_sha256']))
            write(EXT/'progress.json',dict(state='running',updated_utc=now(),prompt_id=prompt,
                valid_requests=sum(r in cached for r in preparation['request_ids']),planned_requests=preparation['planned_requests']))
            print(json.dumps(dict(event='prompt_start',prompt_id=prompt,requests=len(requests))),flush=True)
            # Keep the frozen complete-prompt reservation and maximum concurrency of two.
            results=vlm.execute_requests(RUN,block,requests,'variant_extension_v1')
            export()
            if any(not r or r.get('state')!='valid' for r in results) or len(results)!=len(requests):
                state='blocked'; break
        else:
            state='complete'
    cached=observations()
    valid=sum(r in cached for r in preparation['request_ids'])
    if state=='complete' and valid!=preparation['planned_requests']: state='blocked'
    result=dict(state=state,updated_utc=now(),valid_requests=valid,planned_requests=preparation['planned_requests'])
    write(EXT/'progress.json',result)
    export()
    if state=='complete': vlm.aggregate(EXT)
    print(json.dumps(result),flush=True)
    if state=='blocked': raise RuntimeError('Judge extension stopped; inspect the shared ledger before resuming')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','run','export'])
    parser.add_argument('--max-hours',type=float,default=9)
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='export': export()
    else: run(args.max_hours)
