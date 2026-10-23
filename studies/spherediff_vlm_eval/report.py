"""CSV/LaTeX reports and input comparison figures; never invent missing scores."""
from collections import defaultdict
from pathlib import Path

from .aggregate import CRITERIA, METHOD_LABELS, PRIMARY_METHODS
from .io_utils import atomic_text, write_csv, write_json

SCORE_COLUMNS = ['distortion', 'connectivity', 'image_quality', 'aesthetic_appeal']


def _number(value):
    return '--' if value is None else '%.3f' % value


def _tex(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def latex_table(rows, caption):
    lines = [r'\begin{table}[t]', r'\centering', r'\small', r'\begin{tabular}{lrcccc}', r'\hline',
             r'Method & N & Distortion $\uparrow$ & End Continuity $\uparrow$ & Image Quality $\uparrow$ & Aesthetic Appearance $\uparrow$ \\', r'\hline']
    for row in rows:
        lines.append(' & '.join([_tex(row['method']), str(row['n_prompts'])] +
                                [_number(row[c]) for c in SCORE_COLUMNS]) + r' \\')
    lines += [r'\hline', r'\end{tabular}', r'\caption{' + _tex(caption) + '}', r'\end{table}']
    return '\n'.join(lines)


def write_reports(output, inventory, view_records, aggregation, *, protocol, evaluation=None, preflight=None):
    output = Path(output)
    flat_views = []
    for row in view_records:
        flat = dict(row)
        scores, explanations = row.get('scores') or {}, row.get('explanations') or {}
        for criterion in CRITERIA:
            flat[criterion] = scores.get(criterion) if row.get('status') == 'success' else None
            flat[criterion + '_explanation'] = explanations.get(criterion)
        flat_views.append(flat)
    view_fields = ['panorama_id', 'method_id', 'prompt_id', 'prompt_name', 'view_id', 'yaw', 'pitch',
                   'elevation_group', 'is_seam', 'status', 'cache_key', 'cache_status', 'requested_model',
                   'resolved_model', 'image_sha256', 'view_sha256', 'view_path'] + list(CRITERIA) + [c + '_explanation' for c in CRITERIA]
    write_csv(output / 'view_scores.csv', flat_views, view_fields)
    write_csv(output / 'panorama_scores.csv', aggregation['panoramas'],
              ['panorama_id', 'method_id', 'prompt_id', 'prompt_name', 'image_sha256', 'status', 'expected_views',
               'successful_views', 'missing_views', 'failed_views', 'blocked_views', 'resolved_models'] + list(CRITERIA))
    summary_fields = ['cohort', 'method_id', 'method', 'n_prompts', 'available_prompts', 'status'] + list(CRITERIA)
    write_csv(output / 'matched_summary.csv', aggregation['summaries'], summary_fields)
    write_csv(output / 'alpha_ablation.csv', aggregation['alpha_ablation'], summary_fields)
    write_csv(output / 'paired_comparisons.csv', aggregation['comparisons'],
              ['cohort', 'method_id', 'baseline', 'criterion', 'n_prompts', 'available_prompts', 'status',
               'mean_difference', 'ci95_low', 'ci95_high', 'bootstrap_unit', 'bootstrap_draws', 'bootstrap_seed', 'uncertainty_note'])
    write_csv(output / 'diagnostics.csv', aggregation['diagnostic_summaries'],
              ['cohort', 'method_id', 'group', 'n_prompts', 'available_prompts', 'status'] + list(CRITERIA))
    write_csv(output / 'panorama_diagnostics.csv', aggregation['diagnostics'],
              ['panorama_id', 'method_id', 'prompt_id', 'group', 'expected_views', 'successful_views', 'status'] + list(CRITERIA))
    write_csv(output / 'per_prompt_ranking.csv', aggregation['rankings'],
              ['cohort', 'prompt_id', 'method_id', 'criterion', 'score', 'rank', 'status', 'ties'])
    write_json(output / 'aggregation.json', aggregation)
    primary = [row for row in aggregation['summaries'] if row['cohort'] == 'primary']
    primary.sort(key=lambda row: PRIMARY_METHODS.index(row['method_id']))
    state = aggregation['cohorts'].get('primary', {})
    caption = ('Primary four-method paired comparison. %d matched input prompts; %d fully evaluated paired prompts. '
               'Status: %s. Missing values are unscored, never zero. End Continuity is the judge Connectivity criterion.' %
               (state.get('available_prompts', 0), state.get('evaluated_prompts', 0), state.get('status', 'not_evaluated')))
    tables = [latex_table(primary, caption)]
    for cohort in ('alpha025', 'alpha075'):
        rows = [row for row in aggregation['alpha_ablation'] if row['cohort'] == cohort]
        if rows:
            tables.append(latex_table(rows, 'Predeclared %s ablation on its own paired prompt intersection: %d matched input prompts; %d fully evaluated paired prompts; status %s. Alpha is the maximum-gradient coefficient.' % (cohort, rows[0]['available_prompts'], rows[0]['n_prompts'], rows[0]['status'])))
    atomic_text(output / 'results.tex', '\n\n'.join(tables) + '\n')

    lines = ['# SphereDiff-style GPT-4o panorama evaluation', '',
             '**Status: %s. No scores are inferred from the input images.**' % state.get('status', 'not_evaluated'), '',
             'Primary input cohort: **%d prompts, %d panoramas, %d one-image requests** before cache reuse. '
             'Fully evaluated paired prompts: **%d**.' %
             (state.get('available_prompts', 0), 4 * state.get('available_prompts', 0), 56 * state.get('available_prompts', 0), state.get('evaluated_prompts', 0)), '',
             '| Method | Evaluated N | Matched input N | Distortion ↑ | End Continuity ↑ | Image Quality ↑ | Aesthetic Appearance ↑ |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in primary:
        lines.append('| %s | %d | %d | %s |' % (row['method'], row['n_prompts'], row['available_prompts'],
                                               ' | '.join(_number(row[c]) for c in SCORE_COLUMNS)))
    lines += ['', '## Protocol and reproducibility', '',
              'The prompt is versioned separately with its verification evidence. See [protocol.json](protocol.json) for the exact prompt hash, paper version, camera convention and renderer hash. '
              'The published source is [SphereDiff arXiv v2, Table 5 and Appendix F.2](https://arxiv.org/pdf/2504.14396v2).', '',
              'Each panorama has fourteen 512×512 views at 90° horizontal and vertical FOV: four azimuths at each of −45°, 0°, +45°, plus the two poles. '
              'Appendix F.2 specifies the fourteen directions and FOV; 512×512 resolution and pole yaw 0° are fixed implementation choices. '
              'Yaw 180° crosses the ERP wrap seam at the view center. These evaluation cameras are independent of generation cameras. '
              'The judge receives one image and the fixed evaluation prompt only; method labels in the human comparison figures are never transmitted.', '',
              'All four criteria use integer ratings from 1 to 5, with larger values better. A panorama mean requires all fourteen valid responses. '
              'Method means weight each fully evaluated prompt equally. Every displayed comparison uses the identical evaluated prompt subset for all its methods; partial subsets are explicitly preliminary. '
              'Confidence intervals resample paired prompts, not the fourteen correlated camera views; no interval is reported with fewer than two prompts. '
              'A mixture of resolved judge snapshots is blocked from aggregation.', '',
              'Alpha 0.50 was fixed for the main table before scoring. Alpha 0.25 and 0.75 use separate predeclared intersections; '
              'their paired comparisons include alpha 0.50 where available. The official 28-step SphereDiff baseline stays in its own cohort.', '',
              '## Source audit and missing inputs', '',
              'Accepted selected files and their hashes: [inventory.csv](inventory.csv). Exact pairing: [matched_cases.csv](matched_cases.csv). '
              'Missing cases: [missing_cases.csv](missing_cases.csv). Duplicate/ambiguity and configuration evidence: [provenance_audit.json](provenance_audit.json).', '',
              '| Cohort | Methods | Matched input prompts | Complete paired prompts | State |', '|---|---:|---:|---:|---|']
    for name, cohort in inventory['cohorts'].items():
        scores = aggregation['cohorts'][name]
        lines.append('| %s | %d | %d | %d | %s |' % (name, len(cohort['method_ids']), scores['available_prompts'], scores['evaluated_prompts'], scores['status']))
    missing = inventory.get('missing', [])
    lines += ['', 'Missing input cases: **%d**. Missing or unscored view ratings: **%d**; failed: **%d**; blocked: **%d**.' %
              (len(missing), aggregation['missing_views'], aggregation['failed_views'], aggregation['blocked_views']), '',
              '## API execution plan', '']
    if evaluation:
        for key in ('pending_unique_requests', 'estimated_cost_usd', 'conservative_max_cost_usd', 'requests_attempted', 'credential_available', 'network_access'):
            if key in evaluation:
                lines.append('- %s: `%s`' % (key.replace('_', ' '), evaluation[key]))
        lines += ['', 'The latest cost estimate and requested limits are recorded in [execution_plan.json](execution_plan.json); earlier plans remain in [execution_plans/](execution_plans/). '
                  'Rates are documented in the [GPT-4o model page](https://developers.openai.com/api/docs/models/gpt-4o) and '
                  '[vision token rules](https://developers.openai.com/api/docs/guides/images-vision). '
                  'Actual access and the resolved snapshot are only established by a later authorized API execution; GPT-4o is never replaced automatically.']
    if preflight:
        lines += ['', 'Credential/network preflight (separate from the zero-network API dry-run): `%s`.' % preflight]
    lines += ['', '## Diagnostics', '',
              '[diagnostics.csv](diagnostics.csv) contains elevation, polar and seam-centered results on the same complete paired prompts. '
              '[per_prompt_ranking.csv](per_prompt_ranking.csv) ranks methods separately for each criterion, with ties retained. '
              '[paired_comparisons.csv](paired_comparisons.csv) reports differences against SphereDiff and RGB. '
              'All diagnostics reuse the original fourteen calls. Input comparisons are in [figures/](figures/).', '',
              'No historical panorama pixels or generation files are modified by this study. No paid API calls are made by preparation, reports, tests, or dry-run.', '']
    atomic_text(output / 'report.md', '\n'.join(lines))


def make_figures(output, inventory, views, *, cameras):
    """Human-facing fixed-camera panels, strictly separate from judge payloads."""
    from PIL import Image, ImageDraw, ImageFont
    output = Path(output)
    target = output / 'figures'
    target.mkdir(parents=True, exist_ok=True)
    lookup = {(v['method_id'], v['prompt_id'], v['view_id']): v for v in views}
    prompt_names = {r['prompt_id']: r['prompt_name'] for r in inventory['records']}
    cohort = inventory['cohorts']['primary']
    methods = cohort['method_ids']
    if not methods:
        return []
    labels = {'sphere20': 'SphereDiff FLUX20', 'rgb': 'RGB Old89', 'select': 'Select + screened', 'meanmax050': 'Mean-Max 0.50 + screened'}
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 18)
        title_font = ImageFont.truetype('DejaVuSans.ttf', 26)
    except OSError:
        font = ImageFont.load_default(size=18)
        title_font = ImageFont.load_default(size=26)
    records = []
    for prompt in cohort['prompt_ids']:
        for kind, selected, tile in [('perspectives', cameras, 256), ('seam', [c for c in cameras if c['is_seam']], 512)]:
            pad, row_label, header = 16, 26, 92
            width = len(methods) * (tile + pad) + pad
            height = header + len(selected) * (tile + row_label + pad)
            canvas = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((pad, 12), prompt_names[prompt] + ' | ' + kind + ' | evaluation inputs', fill='black', font=title_font)
            for j, method in enumerate(methods):
                draw.text((pad + j * (tile + pad), 54), labels.get(method, method), fill='black', font=font)
            for i, camera in enumerate(selected):
                y = header + i * (tile + row_label + pad)
                for j, method in enumerate(methods):
                    x = pad + j * (tile + pad)
                    row = lookup[method, prompt, camera['view_id']]
                    draw.text((x, y), 'yaw %g°, pitch %+g°' % (camera['yaw'], camera['pitch']), fill='black', font=font)
                    with Image.open(row['view_path']) as image:
                        canvas.paste(image.convert('RGB').resize((tile, tile), Image.Resampling.LANCZOS), (x, y + row_label))
            # Hash suffix disambiguates same-basename prompt identities.
            import hashlib
            tag = hashlib.sha256(prompt.encode()).hexdigest()[:8]
            path = target / (prompt_names[prompt] + '-' + tag + '-' + kind + '.png')
            temporary = path.with_suffix('.tmp.png')
            canvas.save(temporary)
            temporary.replace(path)
            records.append(dict(prompt_id=prompt, kind=kind, path=str(path), width=width, height=height))
    write_json(target / 'index.json', records)
    return records
