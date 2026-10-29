"""Read-only audit of saved experimental outputs; no generation or model loading.

Historical source identities are evidence of the code that actually generated an
image. They are not required to equal today's source tree. Retained metadata,
image bytes, frozen semantics, execution counts, and independent completion
proofs are verified instead. Raw generation tensors are deliberately not read.
"""
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from . import cohorts

ROOT = Path(__file__).resolve().parents[2]
LAYOUT_IDS = {'spherediff_official_20': 'sphere20', 'diffpano_old89_rgb': 'rgb',
    'diffpano_old70_rgb': 'old70_rgb', 'diffpano_old50_rgb': 'old50_rgb',
    'diffpano_uniform89_rgb': 'uniform89_rgb', 'diffpano_uniform70_rgb': 'uniform70_rgb',
    'diffpano_uniform50_rgb': 'uniform50_rgb', 'diffpano_old89_screened_select': 'select',
    'diffpano_uniform89_screened_select': 'uniform89_select'}
FAMILIES = (
    ('layout24', '10.15layout24-flux216'),
    ('meanmax', '10.13mean-max-old89-flux90'),
    ('meanmax_controls', '10.14mean-max-old89-controls'),
    ('meanmax_tolerance_retry', '10.15mean-max-firefly-retry'),
    ('refinement20', '10.8gradient-refinement-flux'),
    ('refinement28', 'refinement28-flux48'),
    ('ournew', '10.9ournew-flux28'),
)
STATES = ('complete', 'missing', 'invalid', 'ambiguous', 'configuration-incompatible')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Incompatible(ValueError):
    pass


def compatible(condition, message):
    if not condition:
        raise Incompatible(message)


def dictionaries(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from dictionaries(item)
    elif isinstance(value, list):
        for item in value:
            yield from dictionaries(item)


def method_id(row):
    if row.get('variant') in LAYOUT_IDS:
        return LAYOUT_IDS[row['variant']]
    if row.get('method') == 'spherediff':
        return 'sphere' + str(row.get('steps', 20))
    mode = row.get('gradient_mode', row.get('mode'))
    layout = row.get('layout', 'old89')
    reference = row.get('reference_mode') or ('screened' if mode and mode.startswith('poisson') else None)
    alpha = row.get('gradient_mix_alpha')
    if mode == 'poisson_mean_max':
        base = 'meanmax%03d' % round(alpha * 100)
    else:
        base = {'rgb': 'rgb', 'poisson_select': 'select', 'poisson_mean': 'mean', 'poisson_max': 'max'}.get(mode, str(mode))
    if layout not in ('old89', 'official'):
        base = layout + '_' + base
    if reference not in (None, 'screened'):
        base += '_' + reference
    steps = row.get('steps', 20)
    tail = row.get('refinement_steps', round(row.get('last_fraction', 0) * steps))
    if tail:
        base += '_refine%02dof%02d' % (tail, steps)
    elif steps != 20:
        base += '_steps' + str(steps)
    return base


def seed_value(row):
    return row.get('seed', 0)


def configuration_from(meta):
    return meta.get('configuration') or meta.get('source', {}).get('config') or meta.get('source', {}).get('camera_schema') or {}


def runtime_from(meta):
    source = meta.get('source', {})
    old = source.get('original_provenance', {}).get('efficiency', {})
    result = {}
    aliases = {'generation_seconds': ('generation_seconds',), 'setup_seconds': ('setup_seconds',),
        'solver_seconds': ('solver_seconds',), 'gpu': ('gpu',),
        'peak_allocated_gib': ('generation_peak_allocated_gib', 'peak_allocated_gib', 'peak_gpu_allocated_gib'),
        'stage_seconds': ('stage_seconds',)}
    for label, keys in aliases.items():
        result[label] = next((container[key] for container in (meta, source, old)
                              for key in keys if container.get(key) is not None), None)
    result['missing_fields'] = [key for key, value in result.items() if value is None]
    result['policy'] = 'Observed saved values only; absent setup, solver, memory or GPU fields remain null.'
    return result


class Auditor:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.prompts = cohorts.prompt_records(self.root)
        self.by_name = {p['prompt_name']: p for p in self.prompts}
        self.evidence = {}
        self.cache = {}
        self.proofs = defaultdict(list)
        self.family_audit = []

    def hashed(self, path):
        path = Path(path).resolve()
        if path not in self.cache:
            self.cache[path] = sha(path)
        self.evidence[str(path)] = self.cache[path]
        return self.cache[path]

    def add_proofs(self):
        paths = [self.root / 'outputs/10.16report_images/provenance/figures.json',
                 self.root / 'outputs/10.16report_images/provenance/control-inventory.json',
                 self.root / 'outputs/10.16spherediff-vlm-eval/seed0-v1/inventory.json',
                 self.root / 'outputs/10.23report/refinement/provenance/sources.json']
        metric_root = self.root / 'outputs/10.9gradient-metrics/seed0-v1'
        for suite in ('blending20', 'refinement64'):
            paths.extend(sorted((metric_root / suite / 'cases').rglob('*.json')))
        for path in paths:
            if not path.exists():
                continue
            self.hashed(path)
            for item in dictionaries(read(path)):
                mp = item.get('metadata_path', item.get('metadata'))
                mh = item.get('metadata_sha256')
                ip = item.get('image_path', item.get('image', item.get('path')))
                ih = item.get('image_sha256')
                if mp is None and isinstance(ip, str) and isinstance(mh, str):
                    mp = str(Path(ip).parent / 'metadata.json')
                if all(isinstance(v, str) for v in (mp, mh, ip, ih)):
                    self.proofs[str(Path(mp).resolve())].append(dict(path=str(path),
                        metadata_sha256=mh, image_path=str(Path(ip).resolve()), image_sha256=ih))

    def image(self, path, expected):
        from PIL import Image
        require(isinstance(expected, str) and len(expected) == 64, 'No recorded PNG hash')
        require(self.hashed(path) == expected, 'PNG bytes differ from saved artifact hash')
        with Image.open(path) as image:
            require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (4096, 2048),
                    'Expected native 4096x2048 RGB PNG')
            image.verify()

    def base_record(self, family, row, directory, manifest):
        name = row['prompt']
        prompt = self.by_name[name]
        identity = row.get('case_id', row.get('key', name + '/' + str(row.get('mode'))))
        return dict(case_id=family + '/' + identity, historical_case_id=identity, family=family,
            manifest_index=row.get('index'), method_id=method_id(row), prompt_id=prompt['prompt_id'],
            prompt_name=name, prompt_path=prompt['prompt_path'], prompt=prompt,
            effective_prompt_sha256=prompt['effective_sha256'], memberships=prompt['memberships'], seed=row.get('seed', 0),
            image_path=str((directory / 'final.png').absolute()), image_sha256=None,
            metadata_path=str((directory / 'metadata.json').absolute()), metadata_sha256=None,
            manifest_path=str(manifest), manifest_sha256=self.hashed(manifest),
            state='missing', reasons=[], eligible=False, selection_priority=0 if family == 'layout24' else 10,
            parameters={}, runtime={}, provenance={})

    def scientific(self, row, meta, expected_prompt):
        semantic = row.get('semantic', {})
        config = configuration_from(meta)
        source = meta.get('source', {})
        prompt = meta.get('prompt')
        if not isinstance(prompt, dict):
            prompt = source.get('prompt_record', semantic.get('prompt'))
        cohorts.validate_prompt(prompt or {}, expected_prompt)
        steps = row.get('steps', semantic.get('steps', config.get('generation', {}).get('num_inference_steps', 20)))
        seed = row.get('seed', 0)
        official = config.get('pipeline') == 'SphericalFluxPipeline'
        legacy_camera = bool(source.get('camera_schema')) and not source.get('config')
        compatible(bool(config), 'No retained executed configuration')
        compatible(seed == 0, 'Historical development suite requires seed0')
        if official:
            call = config.get('call', {})
            compatible(config.get('model_source') == 'black-forest-labs/FLUX.1-dev', 'Official checkpoint differs')
            compatible(call.get('num_inference_steps') == steps and call.get('guidance_scale') == 3.5
                       and call.get('true_cfg_scale') == 1., 'Official step/guidance settings differ')
            compatible(call.get('n_spherical_points') == 26500 and call.get('erp_width') == 4096
                       and call.get('erp_height') == 2048, 'Official native geometry differs')
            model = dict(id=config.get('model_source'), revision=config.get('revision'), precision=config.get('mixed_precision'))
        elif legacy_camera:
            backend, generation, camera = config['backend'], config['generation'], config['camera']
            model = dict(id=backend['model_id'], revision=backend['model_revision'], precision=backend['precision'])
            compatible(backend['name'] == 'flux' and model['id'] == 'ModelsLab/flux.1-dev'
                and model['revision'] == 'fa45a9eb6808ba8fdfc7cc2756f7f1a16e0921f4', 'Legacy camera checkpoint differs')
            compatible(generation['num_inference_steps'] == steps and generation['seed'] == seed
                and generation['guidance_scale'] == 3.5 and generation['true_cfg_scale'] == 1., 'Legacy camera generation differs')
            compatible(backend['local_rgb_width'] == backend['local_rgb_height'] == 1024
                and camera['fov_x_deg'] == camera['fov_y_deg'] == 80, 'Legacy camera FOV/resolution differs')
            compatible(config['projection']['width'] == 4096 and config['projection']['height'] == 2048
                and config['projection']['consensus_canvas'] == 'erp', 'Legacy camera ERP differs')
            compatible(config['bridge']['mode'] == 'local_identity_preserving'
                and config['trajectory']['transition'] == 'preserve_current_state', 'Legacy camera bridge differs')
            compatible(config['fusion']['mode'] == 'weighted_average' and config['fusion']['weight_mode'] == 'spherediff_center'
                and config['fusion']['temperature'] == .1 and config['fusion']['warp_mode'] == 'standard'
                and config['fusion']['canvas_to_perspective_interpolation'] == 'nearest'
                and config['fusion']['perspective_to_canvas_interpolation'] == 'bilinear', 'Legacy camera projection/fusion differs')
        else:
            model = config.get('model', {})
            compatible(model.get('pipeline') == 'flux' and model.get('id') == 'ModelsLab/flux.1-dev'
                       and model.get('revision') == 'fa45a9eb6808ba8fdfc7cc2756f7f1a16e0921f4', 'FLUX checkpoint/revision differs')
            generation = config.get('generation', {})
            compatible(generation.get('num_inference_steps') == steps and generation.get('guidance_scale') == 3.5
                       and generation.get('true_cfg_scale') == 1., 'Executed steps/guidance differ')
            compatible(config.get('experiment', {}).get('seed') == seed, 'Executed seed differs')
            compatible(config.get('erp') == dict(width=4096, height=2048), 'Executed ERP dimensions differ')
            view = config.get('view', {})
            compatible(view.get('width') == view.get('height') == 1024 and view.get('fov_x') == view.get('fov_y') == 80,
                       'Executed perspective FOV/resolution differs')
            compatible(config.get('consensus_transition') == dict(mode='preserve_current_state', vae_residual_correction=True),
                       'Executed consensus transition differs')
            compatible(config.get('global_pipeline', {}).get('mode') == 'erp_bridge_factorial', 'Executed pipeline differs')
            warp = config.get('warp', {})
            compatible(warp.get('mode') == 'standard' and warp.get('erp_to_perspective', {}).get('interpolation') == 'nearest'
                       and warp.get('perspective_to_erp', {}).get('interpolation') == 'bilinear', 'Executed warp differs')
        if semantic and not meta.get('reused'):
            require(row.get('semantic_sha256') == digest(semantic), 'Frozen semantic digest differs')
            require(meta.get('semantic_sha256') == row['semantic_sha256'], 'Metadata semantic digest differs')
            expected_config = semantic.get('configuration')
            if expected_config is not None:
                compatible(config == expected_config, 'Executed configuration differs from frozen case')
            if row.get('source_sha256'):
                require(meta.get('source_sha256') == row['source_sha256'], 'Saved generation source identity differs')
        fusion = meta.get('fusion', source.get('fusion', {}))
        if not fusion and source.get('mode') == 'rgb':
            compatible(config.get('fusion', {}).get('mode') == 'weighted_average', 'Legacy RGB configuration missing averaging operator')
            fusion = dict(mode='rgb', base_rgb=config['fusion'])
        declared_fusion = semantic.get('solver', semantic.get('fusion'))
        if declared_fusion and not meta.get('reused'):
            compatible(fusion == declared_fusion, 'Executed fusion/solver differs from frozen case')
        mode = row.get('gradient_mode', row.get('mode'))
        reference = row.get('reference_mode')
        alpha = row.get('gradient_mix_alpha')
        if not official:
            compatible(fusion.get('mode') == mode, 'Executed fusion mode differs')
            if reference:
                compatible(fusion.get('reference_mode', 'screened') == reference, 'Executed reference mode differs')
            compatible(fusion.get('gradient_mix_alpha') == alpha, 'Executed max-gradient coefficient differs')
        geometry = semantic.get('geometry', meta.get('geometry', {}))
        expected_geometry = geometry.get('angular_sha256') if isinstance(geometry, dict) else None
        expected_geometry = expected_geometry or semantic.get('camera_geometry_sha256')
        actual_geometry = meta.get('camera_geometry_sha256', source.get('geometry'))
        if expected_geometry:
            compatible(actual_geometry == expected_geometry, 'Executed camera geometry differs')
        cameras = row.get('camera_count', meta.get('camera_count', 89))
        tail = row.get('refinement_steps', round(row.get('last_fraction', 0.) * steps))
        counts = meta.get('counts', source.get('counts'))
        expected_counts = dict(denoiser=cameras * steps, encode=0, decode=cameras) if official else dict(
            denoiser=cameras * steps, encode=2 * cameras * (steps-tail), decode=cameras * (steps-tail+1), initialize=0)
        require(counts == expected_counts, 'Executed denoiser/VAE operation counts differ')
        if not official:
            fusions = meta.get('fusions', source.get('fusions', []))
            # Legacy RGB wrappers retain counts/config but not per-interval fusions.
            if fusions:
                require(len(fusions) == steps-tail+1 and fusions[-1].get('terminal') is True,
                        'Coupled/terminal fusion sequence incomplete')
                for event in fusions:
                    require(event.get('mode') == mode, 'Fusion event mode differs')
                    if mode != 'rgb':
                        require(event.get('converged') is True, 'Poisson solve not converged')
                        compatible(event.get('reference_mode', fusion.get('reference_mode', 'screened')) == (reference or 'screened'), 'Per-solve reference differs')
                        compatible(event.get('gradient_mix_alpha') == alpha, 'Per-solve alpha differs')
            else:
                require(meta.get('reused') and mode == 'rgb', 'Missing executed fusion sequence')
        for container in (row, meta.get('details', {}), source.get('details', {})):
            for key in ('time_travel', 'time_travel_enabled', 'camera_rotation'):
                compatible(not container.get(key, False), 'Unexpected trajectory: ' + key)
            for key in ('backward_calls', 'backward_count', 'replay_count', 'original_noise_reinjections', 'original_noise_reinjection_count', 'additional_denoising_cycles'):
                compatible(container.get(key, 0) == 0, 'Unexpected trajectory count: ' + key)
        refinement = meta.get('refinement', source.get('refinement', {}))
        if isinstance(refinement, dict) and 'independent_intervals' in refinement:
            require(refinement['independent_intervals'] == tail and refinement['coupled_intervals'] == steps-tail,
                    'Executed refinement split differs')
        return dict(backend='flux', model=model, seed=seed, steps=steps, camera_count=cameras,
            layout=row.get('layout', 'old89'), camera_geometry_sha256=actual_geometry,
            generation_fov=[80, 80], erp_resolution=[4096, 2048], local_resolution=[1024, 1024],
            consensus_projection='spherical' if official else 'erp', fusion=fusion,
            gradient_mode=mode, reference_mode=reference, gradient_mix_alpha=alpha,
            alpha_is='max-gradient coefficient', refinement_steps=tail, coupled_steps=steps-tail,
            refinement_last_fraction=row.get('last_fraction', 0.), scheduler=meta.get('provenance', {}).get('prepared_schedule', source.get('schedule', semantic.get('schedule'))),
            schedule_sha256=meta.get('schedule_sha256', source.get('schedule_sha256')),
            initialization=meta.get('initialization', source.get('initialization', semantic.get('initializer'))),
            counts=counts, configuration_sha256=digest(config), configuration=config,
            noise_semantics=config.get('global_pipeline', {}).get('clean_consensus'),
            time_travel=False, camera_rotation=False)

    def case(self, family, row, directory, manifest_path, verifier=None):
        result = self.base_record(family, row, directory, manifest_path)
        image, metadata_path = directory / 'final.png', directory / 'metadata.json'
        if not image.exists() or not metadata_path.exists():
            result['reasons'] = ['Final PNG or retained metadata is absent']
            return result
        try:
            meta = read(metadata_path)
            mh = self.hashed(metadata_path)
            ih = meta.get('artifacts', {}).get('final.png')
            self.image(image, ih)
            result.update(image_sha256=ih, metadata_sha256=mh)
            proof = []
            if verifier is not None:
                require(verifier.checked_complete(row, verify=False), 'Pinned historical/fresh completion verifier rejected case')
                proof.append(dict(kind='pinned_layout_completion', path=str(manifest_path)))
            status = directory / 'status.json'
            if status.exists():
                value = read(status)
                require(value.get('state') in ('complete', 'reused'), 'Existing status is not complete')
                require(value.get('metadata_sha256') == mh, 'Status metadata hash differs')
                require(value.get('image_sha256', ih) == ih, 'Status image hash differs')
                proof.append(dict(kind='complete_status', path=str(status), sha256=self.hashed(status)))
            for candidate in self.proofs.get(str(metadata_path.resolve()), []):
                if candidate['metadata_sha256'] == mh and candidate['image_sha256'] == ih and candidate['image_path'] == str(image.resolve()):
                    proof.append(dict(kind='preserved_independent_audit', **candidate))
            result['parameters'] = self.scientific(row, meta, self.by_name[row['prompt']])
            if family == 'refinement28':
                from studies.refinement28.common import validate_metadata
                require(validate_metadata(row, meta), 'Strict refinement28 scientific completion rejected')
            # The old 20-step suite intentionally pruned statuses and tensors.
            # Its semantic hash, actual full execution and retained image still
            # provide completion evidence, explicitly weaker than a pinned audit.
            if not proof:
                require(bool(row.get('semantic_sha256')) and meta.get('semantic_sha256') == row['semantic_sha256'],
                        'No independently bound completion proof or retained complete semantic execution')
                proof.append(dict(kind='frozen_semantics_and_retained_execution', path=str(manifest_path),
                    limitation='No surviving independent status hash; original per-case execution metadata and image hashes are retained.'))
            config_path = directory / 'config.json'
            if 'config.json' in meta.get('artifacts', {}):
                require(config_path.exists() and self.hashed(config_path) == meta['artifacts']['config.json'], 'Configuration artifact hash differs/missing')
            source = meta.get('source', {})
            original = meta.get('original_metadata_path')
            gaps = []
            if original and not Path(original).exists():
                gaps.append('Pruned original provenance: ' + original + '; retained audited wrapper supplies scientific settings.')
            if not proof or all(p['kind'] == 'frozen_semantics_and_retained_execution' for p in proof):
                gaps.append('Independent historical completion status was pruned; retained semantic execution evidence is explicitly identified.')
            result.update(state='complete', eligible=True, reasons=[], runtime=runtime_from(meta),
                provenance=dict(completion_evidence=proof, metadata_gaps=gaps,
                    historical_source_sha256=meta.get('source_sha256', row.get('source_sha256', row.get('implementation_sha256'))),
                    historical_source_hashes=meta.get('source_hashes'),
                    source_revision=meta.get('implementation_revision', meta.get('historical_source_revision',
                        source.get('original_provenance', {}).get('source_revision'))),
                    semantic_sha256=row.get('semantic_sha256'), reused=bool(meta.get('reused')),
                    pruned_artifacts=[name for name in meta.get('artifacts', {}) if not (directory / name).exists()],
                    raw_tensor_hashes_rechecked=False, scientific_configuration_source='retained executed metadata',
                    current_generation_source_required_to_match=False))
            tolerance = result['parameters']['fusion'].get('relative_tolerance')
            if tolerance not in (None, 1e-5):
                result['method_id'] += '_rtol' + format(tolerance, '.6g')
                result['provenance']['tolerance_note'] = 'Explicit separate numerical-tolerance ablation; never substituted into a nominal-tolerance primary cell.'
            result['panorama_id'] = digest(dict(method=result['method_id'], prompt_id=result['prompt_id'], seed=seed_value(row), image_sha256=ih))
        except Incompatible as error:
            result.update(state='configuration-incompatible', reasons=[str(error)])
        except (OSError, ValueError, KeyError, AssertionError, TypeError, IndexError) as error:
            result.update(state='invalid', reasons=[type(error).__name__ + ': ' + str(error)])
        return result

    def families(self):
        from studies.faster_uniform_resume.completion import load_verifier
        from studies.layout24.common import complete as layout_complete
        verifier = load_verifier(strict_complete=layout_complete)
        self.add_proofs()
        records = []
        for family, output in FAMILIES:
            suite = self.root / 'outputs' / output / 'seed0-v1'
            path = suite / 'manifest.json'
            if not path.exists():
                self.family_audit.append(dict(family=family, manifest_path=str(path), state='missing'))
                continue
            manifest = read(path)
            self.hashed(path)
            rows = manifest['rows']
            self.family_audit.append(dict(family=family, manifest_path=str(path), manifest_sha256=self.hashed(path),
                logical_slots=len(rows), historical_source_sha256=digest(manifest['source_hashes']) if manifest.get('source_hashes') else None))
            for row in rows:
                if row['prompt'] not in self.by_name:
                    continue
                directory = suite / 'cases' / row.get('case_id', row.get('key', ''))
                records.append(self.case(family, row, directory, path, verifier if family == 'layout24' else None))
            print('INVENTORY_FAMILY', family, dict(Counter(r['state'] for r in records if r['family'] == family)), flush=True)
        # The original mean endpoint is an explicitly requested fusion control.
        # Its20 prompts are fixed by the canonical scope, not by metric values.
        suite = self.root / 'outputs/10.7gradient-blending-flux/seed0-v1'
        manifest_path = suite / 'manifest.json'
        self.hashed(manifest_path)
        for name in cohorts.ORIGINAL20:
            row = dict(prompt=name, case_id=name + '/poisson_mean', gradient_mode='poisson_mean',
                       reference_mode='screened', layout='old89', camera_count=89, steps=20, seed=0, last_fraction=0.)
            records.append(self.case('mean_endpoint20', row, suite/'cases'/row['case_id'], manifest_path))
        self.family_audit.append(dict(family='mean_endpoint20', manifest_path=str(manifest_path),
            manifest_sha256=self.hashed(manifest_path), logical_slots=20,
            scope='Canonical original20 screened Poisson-mean endpoints; independent saved metric-input hashes only, scores ignored'))
        print('INVENTORY_FAMILY', 'mean_endpoint20', dict(Counter(r['state'] for r in records if r['family'] == 'mean_endpoint20')), flush=True)
        return records

    def missing_primary(self, records):
        available = {(r['prompt_id'], r['method_id']) for r in records if r['state'] == 'complete' and r.get('eligible')}
        known = {(r['prompt_id'], r['method_id']) for r in records}
        for prompt in self.prompts:
            for method in cohorts.PRIMARY_METHODS:
                key = prompt['prompt_id'], method
                if key not in available and key not in known:
                    records.append(dict(case_id='required-development24/' + prompt['prompt_path'] + '/' + method,
                        family='required-development24', method_id=method, prompt_id=prompt['prompt_id'], prompt_name=prompt['prompt_name'],
                        prompt_path=prompt['prompt_path'], prompt=prompt, effective_prompt_sha256=prompt['effective_sha256'],
                        memberships=prompt['memberships'], state='missing', eligible=False,
                        reasons=['No verified saved result for this exact declared primary method/prompt configuration'],
                        image_path=None, image_sha256=None, metadata_path=None, metadata_sha256=None,
                        parameters={}, runtime={}, provenance={}))


def audit_standalone(auditor):
    """Recheck existing 120 aliases without requiring today's generation code."""
    from studies.standalone_fid import reference as old
    for path, expected in old.PINNED.items():
        require(auditor.hashed(path) == expected, 'Pinned standalone evidence changed: ' + str(path))
    suite = old.STANDALONE
    manifest, index = read(old.MANIFEST), read(old.INDEX)
    saved_path = auditor.root / 'outputs/10.23fid-standalone85/seed0-v1/reference-manifest.json'
    plan_path = saved_path.parent / 'plan.json'
    saved, plan = read(saved_path), read(plan_path)
    require(plan['input_hashes'][str(saved_path)] == auditor.hashed(saved_path), 'Saved standalone manifest is not bound to original FID plan')
    require(saved['logical_count'] == 120 and saved['unique_count'] == 85, 'Saved reference counts differ')
    require(digest(manifest['source_hashes']) == old.SOURCE_SHA256, 'Recorded standalone source identity differs')
    slots = {row['index']: row for row in manifest['rows']}
    indexed = {row['index']: row for row in index['rows']}
    aliases = saved['all_aliases']
    require(len(aliases) == 120 and {a['index'] for a in aliases} == set(range(120)), 'Standalone slot coverage differs')
    groups = defaultdict(list)
    for alias in aliases:
        row, idx = slots[alias['index']], indexed[alias['index']]
        require(idx['valid'] is True and alias['line_sha256'] == row['line_sha256'] == idx['line_sha256'], 'Standalone directional identity differs')
        prompt = auditor.by_name[alias['prompt']]
        require(cohorts.canonical_path(alias['prompt_path']) == prompt['prompt_path'] and alias['prompt_sha256'] == prompt['original_sha256'], 'Standalone scene identity differs')
        require(alias['line_text'] == prompt['effective_lines'][alias['line_index'] - 1], 'Standalone directional line differs')
        require(hashlib.sha256(alias['line_text'].encode()).hexdigest() == alias['line_sha256'], 'Standalone line hash differs')
        image, mp, cp = Path(alias['image']), Path(alias['metadata']), Path(alias['config'])
        require(auditor.hashed(image) == alias['image_sha256'] == idx['image_sha256'], 'Standalone PNG differs')
        require(auditor.hashed(mp) == alias['metadata_sha256'] == idx['metadata_sha256'], 'Standalone metadata differs')
        require(auditor.hashed(cp) == alias['config_sha256'], 'Standalone config differs')
        meta, config = read(mp), read(cp)
        require(meta['semantic_sha256'] == row['semantic_sha256'] == digest(row['semantic']), 'Standalone semantic execution differs')
        require(meta['counts'] == old.COUNTS and meta['guided_steps'] == 20 and meta['native_intervals'] == 20, 'Standalone execution counts differ')
        require(meta['source_sha256'] == old.SOURCE_SHA256 and meta['initialization']['seed'] == 0
                and meta['initialization']['fresh_generator_per_image'] is True, 'Standalone source/seed semantics differ')
        require(meta['artifacts'] == {'final.png': alias['image_sha256'], 'config.json': alias['config_sha256']}, 'Standalone artifact binding differs')
        require({key: config[key] for key in row['semantic']} == row['semantic'], 'Standalone executed configuration differs')
        old.png_header(image)
        groups[alias['image_sha256']].append(alias)
    require(len(groups) == len({a['line_sha256'] for a in aliases}) == 85, 'Standalone unique PNG/line count differs')
    for values in groups.values():
        require(len({a['line_sha256'] for a in values}) == 1, 'Identical image spans different directional texts')
    return dict(state='complete', logical_count=120, unique_count=85, aliases=aliases,
        unique_rows=saved['rows'], alias_multiplicities=dict(Counter(str(len(values)) for values in groups.values())),
        line_multiplicities=dict(Counter(a['line_sha256'] for a in aliases)),
        source_manifest=str(old.MANIFEST), source_manifest_sha256=auditor.hashed(old.MANIFEST),
        saved_reference_manifest=str(saved_path), saved_reference_manifest_sha256=auditor.hashed(saved_path),
        validation='Pinned generation index and independent migration audit; actual PNG, metadata, config, prompt-line and execution checks',
        limitations=['Seed0 was reset for each image; 85 byte-distinct images do not imply independent draws.',
                     'Standalone images have no physically enforced optical FOV.',
                     'Historical source identity retained; current code does not need to equal the old source freeze.'],
        old_protocols=[dict(cohort='development24', generated_views_per_scene=20, generated_views_per_method=480,
                            root=str(saved_path.parent), reference_unique_images=85),
                       dict(cohort='development24', generated_views_per_scene=5, generated_views_per_method=120,
                            root=str(auditor.root/'outputs/10.23fid-standalone85/five-views-yaw0'), reference_unique_images=85),
                       dict(cohort='development15', generated_views_per_scene=20, generated_views_per_method=300,
                            root=str(auditor.root/'outputs/10.23fid-standalone85/max-blending-20views'), reference_unique_images=85)],
        new_protocol_note='native_five_v2 banks A/B: 24 scenes x5=120 directional images each; development15 derives75 slots per bank, never all24.')


def audit_sun(auditor):
    import zipfile
    path = auditor.root / 'outputs/metrics-evaluation/seed0-v1/reference_manifest.json'
    if not path.exists():
        return dict(state='missing', manifest_path=str(path), reason='No saved SUN manifest')
    manifest = read(path)
    archive = Path(manifest['archive'])
    require(manifest['panorama_count'] == len(manifest['records']) == 4260, 'SUN manifest count differs')
    require(manifest['split'] == 'SUN360/test' and manifest['dataset'] == 'zhengli1013/Pano_data', 'Saved real reference release/split differs')
    require(auditor.hashed(archive) == manifest['archive_sha256'], 'SUN archive hash differs')
    with zipfile.ZipFile(archive) as source:
        members = sorted(n for n in source.namelist() if n.startswith('SUN360/test/') and n.lower().endswith('.jpg'))
        require(members == [r['member'] for r in manifest['records']], 'SUN saved member list differs from archive')
        for row in manifest['records']:
            require(row['width'] == 2 * row['height'] and row['width'] > 0, 'SUN manifest ERP dimensions invalid')
            require(hashlib.sha256(source.read(row['member'])).hexdigest() == row['sha256'], 'SUN image hash differs: ' + row['member'])
    return dict(state='complete', manifest_path=str(path), manifest_sha256=auditor.hashed(path),
        dataset=manifest['dataset'], revision=manifest['revision'], split=manifest['split'], image_count=4260,
        archive=str(archive), archive_sha256=manifest['archive_sha256'], all_member_hashes_rechecked=True,
        existing_feature_cache=str(path.parent/'cache/reference_features'),
        old_feature_completion=str(path.parent/'reference_completion.json'),
        reuse_policy='Raw reference manifest verified; old features are not reusable under a new feature/preprocessing protocol without exact identity match.',
        limitations=manifest.get('limitations'))


def _csv(path, rows):
    fields = ['case_id','family','state','method_id','prompt_id','prompt_name','prompt_path','image_path',
              'image_sha256','metadata_path','metadata_sha256','panorama_id','eligible','reasons','parameters','runtime','provenance']
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    os.replace(temporary, path)


def collect(run_dir, root=ROOT):
    """Run on a CPU allocation; write only run_dir/inventory, return all records."""
    auditor = Auditor(root)
    records = auditor.families()
    auditor.missing_primary(records)
    grouped = cohorts.assemble(records, auditor.prompts)
    ambiguities = grouped['ambiguities']
    for ambiguity in ambiguities:
        for row in records:
            if row['case_id'] in ambiguity['case_ids']:
                row.update(state='ambiguous', eligible=False, reasons=[ambiguity['reason']])
    grouped = cohorts.assemble(records, auditor.prompts)
    grouped['ambiguities'] = ambiguities
    try:
        standalone = audit_standalone(auditor)
    except (OSError, ValueError, KeyError, AssertionError, TypeError, IndexError) as error:
        standalone = dict(state='invalid', reason=type(error).__name__ + ': ' + str(error))
    try:
        sun = audit_sun(auditor)
    except (OSError, ValueError, KeyError, AssertionError, TypeError, IndexError) as error:
        sun = dict(state='invalid', reason=type(error).__name__ + ': ' + str(error))
    aliases = defaultdict(list)
    for row in records:
        if row.get('image_sha256'):
            aliases[row['image_sha256']].append(row['case_id'])
    audit = dict(schema_version=2, state_counts=dict(Counter(r['state'] for r in records)),
        method_counts=dict(Counter(r['method_id'] for r in records if r['state'] == 'complete')),
        family_counts={f: dict(Counter(r['state'] for r in records if r['family'] == f)) for f in sorted({r['family'] for r in records})},
        evidence_hashes=auditor.evidence, family_manifests=auditor.family_audit,
        identical_image_aliases={key: value for key, value in aliases.items() if len(value) > 1},
        source_images_modified=False, historical_sources_modified=False, generation_performed=False,
        raw_tensors_rehashed=False, historical_source_policy='Preserve actual historical source hashes; validate saved execution and scientific compatibility, not current source-tree equality.',
        standalone_reference=standalone, real_reference=sun)
    result = dict(schema_version=2, records=records, cohorts=grouped, audit=audit)
    directory = Path(run_dir) / 'inventory'
    write(directory/'records.json', records)
    write(directory/'cohorts.json', grouped)
    write(directory/'provenance_audit.json', audit)
    _csv(directory/'inventory.csv', records)
    _csv(directory/'missing_cases.csv', [r for r in records if r['state'] != 'complete'])
    print('INVENTORY_COMPLETE', audit['state_counts'], flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    require(bool(os.environ.get('SLURM_JOB_ID')), 'Run the full image/archive audit on a CPU allocation')
    collect(args.run_dir)
