"""Score-independent full-path prompt identities and matched evaluation cohorts."""
import hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL20 = ('air_balloons', 'aurora', 'blossom', 'city', 'clouds_over_grass',
              'desert', 'desert_sandstorm', 'firefly', 'firework', 'forest',
              'grand_canyon', 'lavender', 'moonlit_snowy_village', 'ocean',
              'rock_mountain', 'rock_mountain_with_star', 'ruins', 'snow_mountain',
              'storm', 'underwater')
NEW4 = ('bamboo_grove', 'lantern_market', 'sailboat_marina', 'tiled_courtyard')
ORIGINAL11 = ('air_balloons', 'aurora', 'blossom', 'clouds_over_grass', 'firefly',
              'firework', 'forest', 'rock_mountain', 'ruins', 'underwater', 'storm')
PRIMARY_METHODS = ('sphere20', 'rgb', 'select', 'meanmax050')


def canonical_path(value):
    """Keep the entire prompts/... path; basename-only matches are forbidden."""
    parts = Path(value).parts
    if 'prompts' not in parts or '..' in parts:
        raise ValueError('Prompt is not a canonical full prompts/ path: ' + str(value))
    return str(Path(*parts[parts.index('prompts'):]))


def prompt_records(root=ROOT):
    records = []
    for name in ORIGINAL20 + NEW4:
        relative = 'prompts/' + ('ournew/' if name in NEW4 else '') + name + '.txt'
        path = Path(root) / relative
        raw = path.read_bytes()
        lines = raw.decode('utf-8').splitlines()
        if len(lines) != 5 or not all(line.strip() for line in lines):
            raise ValueError('Exactly five nonempty directional lines required: ' + relative)
        value = hashlib.sha256(raw).hexdigest()
        records.append(dict(prompt_id=relative + '@' + value, prompt_name=name,
            prompt_path=relative, full_path=str(path.absolute()), filename=path.name,
            original_sha256=value, effective_sha256=value, effective_lines=lines,
            line_sha256=[hashlib.sha256(line.encode()).hexdigest() for line in lines],
            line_count=5, adapted=False,
            memberships=['development24'] + (['development15'] if name in ORIGINAL11 + NEW4 else []),
            source_group='new4' if name in NEW4 else 'original20'))
    return records


def validate_prompt(record, expected):
    path = record.get('full_path', record.get('path', record.get('prompt_path', '')))
    if canonical_path(path) != expected['prompt_path']:
        raise ValueError('Full prompt path differs')
    if record.get('effective_sha256', record.get('effective_prompt_sha256')) != expected['effective_sha256']:
        raise ValueError('Effective prompt bytes differ')
    if record.get('original_sha256', expected['original_sha256']) != expected['original_sha256'] or record.get('adapted', False):
        raise ValueError('Original prompt/adaptation differs')
    if record.get('effective_lines', expected['effective_lines']) != expected['effective_lines']:
        raise ValueError('Directional prompt lines differ')


def assemble(records, prompts):
    """Intersect complete, unambiguous method cells; never consult any scores."""
    by_cell = defaultdict(list)
    for row in records:
        if row['state'] == 'complete' and row.get('eligible', True):
            by_cell[(row['prompt_id'], row['method_id'], row.get('seed', row.get('parameters', {}).get('seed', 0)))].append(row)
    cells, ambiguous = {}, []
    for key, rows in by_cell.items():
        hashes = {row['image_sha256'] for row in rows}
        if len(hashes) != 1:
            ambiguous.append(dict(prompt_id=key[0], method_id=key[1], seed=key[2],
                case_ids=[r['case_id'] for r in rows], reason='Different eligible PNGs for the same frozen scientific cell'))
        else:
            # Retained canonical layout wrappers take precedence over aliases.
            selected = sorted(rows, key=lambda r: (r.get('selection_priority', 10), r['case_id']))[0]
            cells[key] = selected
    method_groups = dict(primary=list(PRIMARY_METHODS),
        alpha025=list(PRIMARY_METHODS) + ['meanmax025'],
        alpha075=list(PRIMARY_METHODS) + ['meanmax075'],
        camera=['sphere20', 'rgb', 'old70_rgb', 'old50_rgb', 'uniform89_rgb', 'uniform70_rgb', 'uniform50_rgb'],
        uniform=['sphere20', 'rgb', 'select', 'uniform89_rgb', 'uniform70_rgb', 'uniform50_rgb', 'uniform89_select'],
        refinement28=['select_refine05of28', 'select_refine08of28'])
    result = dict(schema_version=2,
        prompt_identity='Full repository-relative prompts/...txt path + @ + SHA256 of exact effective bytes; no basename matching',
        alpha_definition='Max-gradient coefficient: (1-alpha)*mean_gradient + alpha*max_gradient',
        selection='Fixed prompt memberships; matched method intersections without inspecting scores',
        prompts=prompts, ambiguities=ambiguous, cohorts={})
    for name in ('development15', 'development24'):
        members = [p for p in prompts if name in p['memberships']]
        matched = {}
        for comparison, methods in method_groups.items():
            included = [p['prompt_id'] for p in members if all((p['prompt_id'], m, 0) in cells for m in methods)]
            matched[comparison] = dict(method_ids=methods, prompt_ids=included, prompt_count=len(included),
                panorama_ids=[cells[(pid, method, 0)]['panorama_id'] for pid in included for method in methods],
                excluded=[dict(prompt_id=p['prompt_id'], missing_methods=[m for m in methods if (p['prompt_id'], m, 0) not in cells])
                          for p in members if p['prompt_id'] not in included])
        result['cohorts'][name] = dict(prompt_ids=[p['prompt_id'] for p in members],
            prompt_count=len(members), matched=matched)
    result['cohorts']['confirmation24'] = dict(prompt_ids=[], prompt_count=0, state='not-created',
        reason='Fresh real-world confirmation scenes are selected separately before generation and scores; development prompts cannot substitute.')
    result['selected_cells'] = [dict(prompt_id=pid, method_id=method, seed=seed, panorama_id=row['panorama_id'], case_id=row['case_id'])
                                for (pid, method, seed), row in sorted(cells.items())]
    return result
