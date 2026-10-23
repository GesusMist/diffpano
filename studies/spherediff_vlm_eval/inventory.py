"""Read-only verification of historical panoramas and predeclared paired cohorts.

No generation module is imported. Deleted tensor/status artifacts are not needed
for judging immutable PNGs when a preserved completed audit independently binds
both the PNG and its metadata hashes. Such evidence is explicitly recorded.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ('air_balloons', 'aurora', 'blossom', 'clouds_over_grass', 'firefly',
            'firework', 'forest', 'rock_mountain', 'ruins', 'underwater', 'storm')
NEW = ('bamboo_grove', 'lantern_market', 'sailboat_marina', 'tiled_courtyard')
PROMPTS = ORIGINAL + NEW
METHOD_IDS = ('sphere20', 'rgb', 'select', 'meanmax050', 'meanmax025', 'meanmax075', 'sphere28')
PRIMARY_METHODS = ('sphere20', 'rgb', 'select', 'meanmax050')
OLD89 = 'b55b4eb5051cdfb8c579e81d4f51ece1f15aeb0e8d7e1fde6b7b74a49154435c'
OFFICIAL_COMMIT = '2c8c68ba088f2803b3dce4b52b7b0d68bc996139'
OFFICIAL_PIPELINE_SHA = '641e5c839188887939e8e2346e971a001f1444bfe1dce760b04487ae4e03fa7e'
COUNTS = dict(denoiser=1780, encode=3560, decode=1869, initialize=0)
DIFFPANO_CHECKPOINT = ('ModelsLab/flux.1-dev', 'fa45a9eb6808ba8fdfc7cc2756f7f1a16e0921f4')
INITIAL_LOCAL_SHA = '0841cf95c51858d6641f8b4780a4c0ef205f6d1f068570feef529675a6df78a7'
SCHEDULE_SHA = 'b2caee458b8279600e77e5dfd78048b0544ce178981c397a3189edc2d430b960'
AUDIT_NAMES = {'inventory.json', 'control-inventory.json', 'figures.json', 'index.json',
               'generation_plan.json', 'reuse_audit.json', 'provenance.json',
               'final_audit.json', 'completion.json', 'manifest.json'}
SKIP_DIRS = {'cache', 'logs', 'history', '__pycache__', '.git', 'intermediates',
             'diagnostics', 'runtime', 'validation', 'orchestration', 'snapshots'}


class Rejected(ValueError):
    """An artifact cannot satisfy the declared comparison semantics."""


def require(condition, reason):
    if not condition:
        raise Rejected(reason)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def write_csv(path, rows, columns):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})
    os.replace(tmp, path)


def walk_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


def canonical_prompt_path(path):
    """Preserve the whole prompts/... path; never match by basename alone."""
    parts = Path(str(path)).parts
    require('prompts' in parts, 'Prompt path has no canonical prompts/ prefix')
    offset = parts.index('prompts')
    relative = Path(*parts[offset:])
    require('..' not in relative.parts, 'Noncanonical prompt path')
    return relative.as_posix()


def prompt_scope(root):
    """Read the experiment's literals without importing its frozen generation code."""
    source = root / 'studies/mean_max_old89/common.py'
    original, new = ORIGINAL, NEW
    if source.exists():
        values = {}
        for node in ast.parse(source.read_text()).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ('ORIGINAL', 'NEW'):
                        values[target.id] = tuple(ast.literal_eval(node.value))
        require(set(values) == {'ORIGINAL', 'NEW'}, 'Cannot parse exact mean-max prompt scope')
        original, new = values['ORIGINAL'], values['NEW']
        require(len(original) == 11 and len(new) == 4, 'Expected exact 11+4 mean-max prompt scope')
    result = []
    for name in original + new:
        relative = 'prompts/' + ('ournew/' if name in new else '') + name + '.txt'
        path = root / relative
        if not path.exists():
            result.append(dict(prompt_name=name, prompt_path=relative, prompt_id=relative + '@missing',
                               effective_prompt_sha256=None, missing_prompt=True))
            continue
        raw = path.read_bytes()
        require(len(raw.decode().splitlines()) == 5, 'Expected five-line prompt: ' + relative)
        h = hashlib.sha256(raw).hexdigest()
        result.append(dict(prompt_name=name, prompt_path=relative, prompt_id=relative + '@' + h,
                           effective_prompt_sha256=h, missing_prompt=False))
    return result


class Resolver:
    """Explicit relocation first, then current checkout and configured output roots."""
    def __init__(self, root, historical_roots=(), remappings=None):
        self.root = Path(root).resolve()
        self.roots = list(dict.fromkeys([self.root / 'outputs'] + [Path(p).resolve() for p in historical_roots]))
        self.remappings = sorted((remappings or {}).items(), key=lambda item: -len(item[0]))
        self.cache = {}

    def resolve(self, value, base=None):
        cache_key = (str(value), str(base))
        if cache_key in self.cache:
            return self.cache[cache_key]
        path = Path(str(value))
        candidates = []
        for old, new in self.remappings:
            old = old.rstrip('/')
            if str(path) == old or str(path).startswith(old + '/'):
                candidates.append(Path(new) / str(path)[len(old):].lstrip('/'))
        has_explicit_remap = bool(candidates)
        if not path.is_absolute():
            candidates += [self.root / path]
            if base is not None:
                candidates.append(Path(base) / path)
        else:
            candidates.append(path)
        # Preserve relative path below outputs, including date-qualified names.
        parts = path.parts
        for i, part in enumerate(parts):
            if part == 'outputs' or part.startswith('outputs-from-'):
                suffix = Path(*parts[i + 1:])
                candidates += [out / suffix for out in self.roots]
        existing = []
        for candidate in candidates:
            if candidate.exists():
                real = candidate.resolve()
                if real not in existing:
                    existing.append(real)
        require(existing, 'Missing referenced path: ' + str(value))
        # Explicit remapping is an intentional precedence rule.
        if has_explicit_remap and candidates and candidates[0].exists():
            self.cache[cache_key] = candidates[0].resolve()
            return self.cache[cache_key]
        if len(existing) > 1:
            require(all(p.is_file() for p in existing) and len({sha(p) for p in existing}) == 1,
                    'Ambiguous relocated path: ' + str(value))
        self.cache[cache_key] = existing[0]
        return existing[0]


def discover(resolver, output):
    """Discover surviving case metadata, including additional historical roots."""
    metadata, audit = set(), set()
    visited = set()
    output = Path(output).resolve()
    for root in resolver.roots:
        if not root.exists():
            continue
        for directory, subdirs, files in os.walk(root, followlinks=True):
            folder = Path(directory)
            real = folder.resolve()
            if real in visited or real == output or output in real.parents:
                subdirs[:] = []
                continue
            visited.add(real)
            subdirs[:] = [name for name in subdirs if name not in SKIP_DIRS and
                          'perspective24' not in name and 'spherediff-vlm-eval' not in name]
            for name in files:
                if name in AUDIT_NAMES and not ('cases' in folder.parts and name == 'manifest.json'):
                    audit.add((folder / name).resolve())
                if name == 'metadata.json' or (name == 'config.json' and
                        'metadata.json' not in files and ('final.png' in files or 'final_result.png' in files)):
                    metadata.add((folder / name).resolve())
    return sorted(metadata), sorted(audit)


def assemble_cohorts(records, scope=None):
    """Predeclared intersections; no scores are consulted in cohort construction."""
    by_method = {method: {r['prompt_id']: r for r in records if r['method_id'] == method}
                 for method in METHOD_IDS}
    definitions = dict(primary=PRIMARY_METHODS,
                       alpha025=PRIMARY_METHODS + ('meanmax025',),
                       alpha075=PRIMARY_METHODS + ('meanmax075',),
                       sphere28=('sphere28', 'sphere20', 'rgb', 'select', 'meanmax050'))
    order = {r['prompt_id']: i for i, r in enumerate(scope or [])}
    result = {}
    for name, methods in definitions.items():
        common = set.intersection(*(set(by_method[m]) for m in methods))
        prompts = sorted(common, key=lambda p: (order.get(p, 10**9), p))
        result[name] = dict(method_ids=list(methods), prompt_ids=prompts,
                            panorama_ids=[by_method[m][p]['panorama_id'] for p in prompts for m in methods],
                            matched_prompt_count=len(prompts), panorama_count=len(prompts) * len(methods),
                            expected_requests=len(prompts) * len(methods) * 14)
    return result


class InventoryBuilder:
    def __init__(self, root, output, historical_roots=(), remappings=None):
        self.root, self.output = Path(root).resolve(), Path(output).resolve()
        self.resolver = Resolver(root, historical_roots, remappings)
        self.scope = prompt_scope(self.root)
        self.prompts = {p['prompt_path']: p for p in self.scope}
        self.proofs = defaultdict(list)
        self.archived = []
        self.completed_images = defaultdict(list)
        self.excluded = []
        self.source_files = {}
        self.hashes = {}
        self.image_checks = {}
        self.manifests = {}

    def hashed(self, path):
        path = Path(path).resolve()
        if path not in self.hashes:
            self.hashes[path] = sha(path)
        self.source_files[str(path)] = self.hashes[path]
        return self.hashes[path]

    def resolve(self, path, base=None):
        return self.resolver.resolve(path, base)

    def proof(self, metadata_path, metadata_hash, image_path, image_hash, evidence_path, kind):
        if not all(isinstance(x, str) and x for x in (metadata_path, metadata_hash, image_path, image_hash)):
            return
        try:
            meta, image = self.resolve(metadata_path), self.resolve(image_path)
        except (Rejected, OSError):
            return
        self.proofs[meta].append(dict(metadata_sha256=metadata_hash, image_path=str(image),
                                    image_sha256=image_hash, evidence_path=str(evidence_path), kind=kind))

    def collect_audits(self, paths):
        for path in paths:
            try:
                document = read(path)
            except (ValueError, OSError):
                continue
            if path.name == 'manifest.json':
                self.manifests[path.parent] = document
            if path.name == 'completion.json' and isinstance(document, dict) and document.get('complete') is True:
                for item in document.get('rows', []):
                    if item.get('status') == 'complete' and isinstance(item.get('output'), str) and item.get('image_sha256'):
                        try:
                            self.completed_images[self.resolve(item['output'])].append((item, path))
                        except Rejected:
                            pass
            if isinstance(document, dict) and isinstance(document.get('rows'), list):
                for row in document['rows']:
                    if isinstance(row, dict) and row.get('provenance_status') == 'verified' and row.get('method') == 'spherediff':
                        self.archived.append((row, path))
            for item in walk_dicts(document):
                if item.get('valid') is False or item.get('completion_status') in ('failed', 'missing', 'incomplete'):
                    continue
                mp = item.get('metadata_path') or item.get('metadata')
                ip = item.get('image_path') or item.get('image') or item.get('path') or item.get('output_path')
                ih = item.get('image_sha256') or item.get('sha256')
                self.proof(mp, item.get('metadata_sha256'), ip, ih, path, 'preserved_audit')
                self.proof(item.get('original_metadata_path'), item.get('original_metadata_sha256'),
                           item.get('image_path'), item.get('image_sha256'), path, 'preserved_reuse_audit')
                # The official original-reference manifest records audited reused images.
                if isinstance(item.get('reused_image'), dict):
                    image = item['reused_image']
                    self.proof(item.get('metadata'), item.get('reused_metadata_sha256'),
                               image.get('path'), image.get('sha256'), path, 'official_reuse_manifest')

    def prompt(self, record):
        require(isinstance(record, dict), 'Missing complete prompt record')
        relative = canonical_prompt_path(record.get('path', record.get('relative_path', '')))
        require(relative in self.prompts, 'Outside the exact 15-prompt scope: ' + relative)
        desired = self.prompts[relative]
        require(not desired['missing_prompt'], 'Current prompt file missing')
        effective = record.get('effective_sha256', record.get('effective_prompt_sha256',
                               record.get('original_sha256', record.get('sha256'))))
        require(effective == desired['effective_prompt_sha256'], 'Effective prompt hash differs: ' + relative)
        original = record.get('original_sha256', record.get('sha256', effective))
        require(original == effective and not record.get('adapted', False), 'Adapted/original prompt identity differs')
        lines = record.get('effective_lines', record.get('lines'))
        if lines is not None:
            require(lines == (self.root / relative).read_text().splitlines(), 'Effective prompt lines differ')
        self.hashed(self.root / relative)
        return desired

    def image(self, path, expected):
        path = Path(path).resolve()
        require(isinstance(expected, str) and len(expected) == 64, 'No recorded image SHA256')
        require(self.hashed(path) == expected, 'PNG SHA256 differs from provenance')
        if path not in self.image_checks:
            with Image.open(path) as image:
                require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (4096, 2048),
                        'Expected an RGB PNG panorama at 4096x2048')
                image.load()
            self.image_checks[path] = True

    def completion(self, path, image, expected, meta):
        evidence = []
        status_path = path.parent / 'status.json'
        if status_path.exists():
            status = read(status_path)
            require(status.get('state') == 'complete', 'Present case status is not complete')
            require(status.get('metadata_sha256', status.get('config_sha256')) == self.hashed(path),
                    'Completion status metadata hash differs')
            require(status.get('image_sha256', expected) == expected, 'Completion status image hash differs')
            self.hashed(status_path)
            evidence.append(dict(path=str(status_path), kind='live_complete_status'))
        for proof in self.proofs.get(path.resolve(), []):
            if proof['metadata_sha256'] == self.hashed(path) and proof['image_sha256'] == expected and Path(proof['image_path']).resolve() == image.resolve():
                ep = Path(proof['evidence_path'])
                self.hashed(ep)
                evidence.append(dict(path=str(ep), kind=proof['kind']))
        # Some original step-control configs are self-contained completed records;
        # demand their immutable manifest too, rather than relying on a filename.
        if not evidence and meta.get('state') == 'complete' and meta.get('manifest_sha256'):
            for parent in path.parents:
                manifest = parent / 'manifest.json'
                if manifest.exists() and self.hashed(manifest) == meta['manifest_sha256']:
                    evidence.append(dict(path=str(manifest), kind='complete_config_and_manifest'))
                    break
        require(evidence, 'No preserved independent completion proof binding metadata and PNG')
        return evidence

    def check_manifest(self, path, meta):
        """Compare surviving per-case semantics with its frozen manifest when present."""
        if not meta.get('semantic_sha256'):
            return []
        case = meta.get('case', {})
        case_id = case.get('case_id', meta.get('case_id'))
        for parent in path.parents:
            manifest = self.manifests.get(parent)
            if not manifest:
                continue
            rows = [row for row in manifest.get('rows', []) if row.get('case_id') == case_id]
            if not rows:
                continue
            require(len(rows) == 1, 'Ambiguous frozen manifest case')
            row = rows[0]
            require(meta['semantic_sha256'] == row.get('semantic_sha256'), 'Frozen semantic hash mismatch')
            require(meta.get('source_sha256') == row.get('source_sha256'), 'Frozen source hash mismatch')
            self.hashed(parent / 'manifest.json')
            return [dict(path=str(parent / 'manifest.json'), kind='frozen_case_semantics')]
        return []

    @staticmethod
    def reject_trajectory(meta, config):
        for container in (meta.get('case', {}), meta.get('trajectory', {}), meta.get('details', {}),
                          meta.get('refinement', {}), config.get('global_pipeline', {}).get('refinement', {})):
            for key in ('time_travel', 'time_travel_enabled', 'camera_rotation', 'dynamic_cameras'):
                require(not container.get(key, False), 'Forbidden trajectory option: ' + key)
            for key in ('replay_count', 'backward_count', 'backward_calls', 'replays', 'original_noise_reinjections', 'original_noise_reinjection_count'):
                require(container.get(key, 0) in (0, None), 'Forbidden trajectory counter: ' + key)
        require(config.get('sampling', {}).get('rotation', {}).get('enabled') is False,
                'Camera rotation is not explicitly disabled')

    def standard(self, path, meta):
        """Gradient, mean-max, control, RGB wrapper and matched official schemas."""
        config = meta.get('configuration', {})
        require(isinstance(meta.get('prompt'), dict) and config, 'Unsupported metadata schema')
        prompt = self.prompt(meta['prompt'])
        official = meta.get('official', meta)
        is_sphere = config.get('pipeline') == 'SphericalFluxPipeline'
        artifacts = meta.get('artifacts', {})
        image = path.parent / ('final.png' if (path.parent / 'final.png').exists() else 'final_result.png')
        expected = artifacts.get(image.name, meta.get('output', {}).get('sha256'))
        require(image.exists(), 'Missing final panorama')
        if is_sphere:
            call = config.get('call', {})
            steps = call.get('num_inference_steps')
            require(steps in (20, 28), 'Official steps outside 20/28')
            seed = meta.get('case', meta.get('row', {})).get('seed')
            if seed is None:
                seed = meta.get('seed')
            require(seed == 0, 'Official seed is not 0')
            require(config.get('model_source') == 'black-forest-labs/FLUX.1-dev', 'Wrong official FLUX checkpoint')
            require(official.get('official_source_commit') == OFFICIAL_COMMIT, 'Wrong/missing official source commit')
            require(official.get('official_source_hashes', {}).get('pipelines_ours/pipeline_spherical_flux.py') == OFFICIAL_PIPELINE_SHA,
                    'Wrong/missing official pipeline hash')
            require(call.get('n_spherical_points') == 26500 and call.get('erp_width') == 4096 and call.get('erp_height') == 2048,
                    'Wrong official native geometry')
            require(call.get('guidance_scale') == 3.5 and call.get('true_cfg_scale') == 1.0 and call.get('weighted_average_temperature') == .1,
                    'Wrong official call settings')
            counts = meta.get('counts', official.get('execution', {}).get('counts'))
            require(counts == dict(denoiser=89 * steps, encode=0, decode=89), 'Official execution counts mismatch')
            schedule = official.get('schedule', {})
            require(len(schedule.get('timesteps', [])) == steps and len(schedule.get('sigmas', [])) == steps + 1,
                    'Official schedule length mismatch')
            method, fusion, layout = 'sphere' + str(steps), {}, 'official_native'
            geometry = 'official_native_spherical_26500'
            native = 'spherical'
            notes = ['Official SphereDiff retains its native spherical representation, initialization and generation cameras.']
        else:
            require(config.get('model', {}).get('pipeline') == 'flux' and meta.get('backend', 'flux') == 'flux', 'Not FLUX')
            require(config.get('generation', {}).get('num_inference_steps') == 20, 'Not 20 denoising steps')
            model = config.get('model', {})
            require((model.get('id'), model.get('revision')) == DIFFPANO_CHECKPOINT, 'Different FLUX checkpoint/revision')
            require(model.get('precision') == 'bf16', 'Different model precision')
            generation = config.get('generation', {})
            require(generation.get('guidance_scale') == 3.5 and generation.get('true_cfg_scale') == 1.0
                    and generation.get('batch_size') == 1, 'Different FLUX guidance or batch configuration')
            require(config.get('consensus_transition') == dict(mode='preserve_current_state', vae_residual_correction=True),
                    'Different consensus bridge transition')
            require(config.get('global_pipeline', {}).get('mode') == 'erp_bridge_factorial', 'Different global bridge pipeline')
            warp = config.get('warp', {})
            require(warp.get('mode') == 'standard' and
                    warp.get('erp_to_perspective', {}).get('interpolation') == 'nearest' and
                    warp.get('perspective_to_erp', {}).get('interpolation') == 'bilinear', 'Different projection warp/interpolation')
            base_fusion = config.get('fusion', {})
            require(base_fusion.get('mode') == 'weighted_average' and base_fusion.get('weight_mode') == 'spherediff_center'
                    and base_fusion.get('spherediff_temperature') == .1, 'Different base RGB confidence weights')
            require(meta.get('initialization', {}).get('initial_local_sha256') == INITIAL_LOCAL_SHA, 'Different GWTF initial local state')
            require(meta.get('schedule_sha256') == SCHEDULE_SHA, 'Different prepared diffusion schedule')
            require(config.get('experiment', {}).get('seed') == 0, 'Not seed 0')
            require(config.get('canvas', {}).get('mode') == 'erp', 'Not ERP consensus')
            require(config.get('erp') == dict(height=2048, width=4096), 'Wrong ERP configuration')
            require(meta.get('camera_geometry_sha256') == OLD89, 'Not saved Old89 geometry')
            require(config.get('view', {}).get('width') == config.get('view', {}).get('height') == 1024,
                    'Wrong local RGB resolution')
            require(config.get('view', {}).get('fov_x') == config.get('view', {}).get('fov_y') == 80, 'Wrong generation FOV')
            require(meta.get('counts') == COUNTS, 'Wrong fully coupled execution counts')
            refinement = meta.get('refinement', config.get('global_pipeline', {}).get('refinement', {}))
            if refinement:
                require(refinement.get('last_fraction') == 0, 'Nonzero refinement')
            self.reject_trajectory(meta, config)
            fusion = meta.get('fusion', {})
            mode = fusion.get('mode')
            notes = ['Historical source repository revisions may differ; checkpoint, guidance, bridge, warp, saved geometry, schedule and initial local state are matched.']
            if mode == 'rgb':
                method = 'rgb'
            elif mode in ('poisson_select', 'poisson_mean_max'):
                reference = fusion.get('reference_mode')
                if reference is None and mode == 'poisson_select' and 'gradient-blending' in str(path):
                    # This legacy implementation predates reference_mode: it is
                    # the screened operator, explicitly audited by later controls.
                    require(any(p['kind'].startswith('preserved') for p in self.proofs.get(path.resolve(), [])),
                            'Legacy screened operator lacks independent audited provenance')
                    reference = 'screened'
                    notes.append('Legacy gradient-blending Poisson operator is screened; independently audited in preserved control/reuse inventory.')
                require(reference == 'screened', 'Not lambda_blending (screened)')
                require(fusion.get('lambda_color') == .1, 'Wrong screened lambda_color')
                if mode == 'poisson_select':
                    require(fusion.get('gradient_mix_alpha') is None, 'Select must not carry mean-max alpha')
                    method = 'select'
                else:
                    alpha = fusion.get('gradient_mix_alpha')
                    require(alpha in (.25, .50, .75), 'Mean-max alpha outside predeclared ablations')
                    method = 'meanmax%03d' % round(alpha * 100)
                fusions = meta.get('fusions', [])
                require(len(fusions) == 21, 'Expected 20 coupled + terminal Poisson solves')
                for event in fusions:
                    require(event.get('mode') == mode and event.get('converged') is True, 'Poisson solve not converged or wrong mode')
                    require(event.get('reference_mode', reference) == reference, 'Per-solve reference mismatch')
                    if mode == 'poisson_mean_max':
                        require(event.get('gradient_mix_alpha') == fusion['gradient_mix_alpha'], 'Per-solve alpha mismatch')
                require(fusions[-1].get('terminal') is True, 'Terminal Poisson solve absent')
            else:
                raise Rejected('Unsupported fusion mode: ' + str(mode))
            steps, seed, layout, geometry, native = 20, 0, 'old89', OLD89, 'erp'
            if not refinement:
                notes.append('Legacy no-refinement execution verified by full coupled 1780/3560/1869 counts.')
        self.image(image, expected)
        evidence = self.completion(path, image, expected, meta) + self.check_manifest(path, meta)
        if 'config.json' in artifacts:
            config_path = path.parent / 'config.json'
            require(config_path.exists() and self.hashed(config_path) == artifacts['config.json'], 'Config artifact hash mismatch/missing')
            evidence.append(dict(path=str(config_path), kind='configuration_artifact'))
        return self.record(prompt, method, image, expected, path, evidence, config, fusion,
                           steps, seed, layout, geometry, native, notes, meta)

    def record(self, prompt, method, image, image_hash, metadata, evidence, config, fusion,
               steps, seed, layout, geometry, native, notes, meta):
        identity = dict(method_id=method, prompt_id=prompt['prompt_id'], image_sha256=image_hash)
        return dict(prompt, panorama_id=digest(identity), method_id=method,
                    image_path=str(image.resolve()), image_sha256=image_hash,
                    metadata_path=str(metadata), metadata_sha256=self.hashed(metadata),
                    metadata_paths=[str(metadata)], checks=dict(image_hash=True, png_rgb_4096x2048=True,
                    effective_prompt_identity=True, actual_configuration=True, completed_provenance=True),
                    evidence=evidence, backend='flux', steps=steps, seed=seed, layout=layout,
                    consensus_projection=native, erp_width=4096, erp_height=2048,
                    camera_geometry_sha256=geometry, refinement_last_fraction=0,
                    time_travel=False, camera_rotation=False, fusion=fusion,
                    gradient_mix_alpha=fusion.get('gradient_mix_alpha'),
                    reference_mode=fusion.get('reference_mode', 'screened' if method == 'select' else None),
                    configuration_sha256=digest(config), configuration=config,
                    source_revision=meta.get('implementation_revision', meta.get('source_revision')),
                    notes=notes, completion_evidence_kind=sorted(set(e['kind'] for e in evidence)))

    def native_sphere(self, path, meta):
        """Legacy original_spherediff native metadata plus its final completion audit."""
        require(meta.get('backend') == 'flux' and meta.get('source_commit') == OFFICIAL_COMMIT,
                'Not an official native FLUX reference')
        hashes = meta.get('official_source_hashes', meta.get('source_hashes', {}))
        require(hashes.get('pipelines_ours/pipeline_spherical_flux.py') == OFFICIAL_PIPELINE_SHA,
                'Native official pipeline hash differs')
        prompt = self.prompt(meta.get('prompt'))
        config = meta.get('spec') or dict(pipeline=meta.get('pipeline_class'),
                    model_source=meta.get('checkpoint'), revision=meta.get('revision'), variant=meta.get('variant'),
                    mixed_precision={'torch.bfloat16': 'bf16'}.get(meta.get('precision'), meta.get('precision')),
                    enable_model_cpu_offload=meta.get('model_cpu_offload'),
                    enable_vae_tiling=meta.get('vae_tiling'), call=meta.get('explicit_call_arguments'))
        require(config.get('pipeline') == 'SphericalFluxPipeline' and
                config.get('model_source') == 'black-forest-labs/FLUX.1-dev', 'Wrong native official pipeline/checkpoint')
        call = config.get('call') or {}
        require(call == dict(num_inference_steps=28, guidance_scale=3.5, true_cfg_scale=1.0,
                            n_spherical_points=26500, weighted_average_temperature=.1,
                            erp_height=2048, erp_width=4096), 'Native original invocation differs')
        require(meta.get('seed') == 0 and meta.get('transformer_forwards') == 2492,
                'Native original seed/execution count differs')
        schedule = meta.get('scheduler', meta.get('prepared_schedule', {}))
        require(len(schedule.get('timesteps', [])) == 28 and len(schedule.get('sigmas', [])) == 29,
                'Native original schedule differs')
        image = (path.parent / 'final_result.png').resolve()
        completions = self.completed_images.get(image, [])
        evidence = []
        image_hash = meta.get('output', {}).get('sha256')
        for completion, audit_path in completions:
            require(completion.get('backend') == 'flux' and completion.get('seed') == 0 and
                    completion.get('steps') == 28 and completion.get('forwards') == 2492 and
                    completion.get('prompt') == prompt['prompt_name'], 'Native completion configuration differs')
            if image_hash is not None:
                require(completion['image_sha256'] == image_hash, 'Native completed image hashes disagree')
            image_hash = completion['image_sha256']
            self.hashed(audit_path)
            final_audit_path = audit_path.parent / 'final_audit.json'
            require(final_audit_path.exists(), 'Native completion lacks its final audit')
            final_audit = read(final_audit_path)
            require(final_audit.get('passed') is True and final_audit.get('all_four_complete') is True and
                    final_audit.get('artifact_hashes', {}).get(audit_path.name) == self.hashed(audit_path),
                    'Native final completion audit hash differs')
            self.hashed(final_audit_path)
            manifest_path = audit_path.parent / 'manifest.json'
            manifest = read(manifest_path)
            require(final_audit.get('artifact_hashes', {}).get('manifest.json') == self.hashed(manifest_path),
                    'Native original manifest hash differs')
            planned = [r for r in manifest.get('rows', []) if r.get('backend') == 'flux' and r.get('prompt') == prompt['prompt_name']]
            require(len(planned) == 1 and planned[0].get('seed') == 0 and planned[0].get('spec') == config,
                    'Native original configuration differs from audited manifest')
            if meta.get('manifest_sha256'):
                require(meta['manifest_sha256'] == self.hashed(manifest_path), 'Native generation manifest hash differs')
            evidence += [dict(path=str(audit_path), kind='native_completed_image_audit'),
                         dict(path=str(final_audit_path), kind='native_final_audit'),
                         dict(path=str(manifest_path), kind='native_audited_generation_manifest')]
        require(evidence, 'No independently audited native completion record')
        self.image(image, image_hash)
        return self.record(prompt, 'sphere28', image, image_hash, path, evidence, config, {},
                    28, 0, 'official_native', 'official_native_spherical_26500', 'spherical',
                    ['Legacy official FLUX28 uses its native spherical representation and is a separate diagnostic.',
                     'Surviving native metadata/call, pinned official source, generation manifest and completed image are checked against the preserved final audit.'], meta)

    def archived_sphere(self, row, path):
        """Original benchmark rows retain the validated native 28-step invocation.

        They are a separately labeled diagnostic, never substituted for sphere20.
        This adapter keeps original missing-log limitations visible.
        """
        require(row.get('backend') == 'flux' and row.get('steps') == 28, 'Not original FLUX28 diagnostic')
        require(row.get('completion_status') == 'complete' and row.get('provenance_status') == 'verified', 'Archived result not validated complete')
        require(row.get('accounting') == dict(exit_code='0:0', state='COMPLETED'), 'No successful archived accounting')
        require(row.get('camera_strategy') == 'official_spherediff' and row.get('consensus_projection') == 'spherical', 'Not official native SphereDiff')
        require(row.get('seed') == 0 and row.get('model_id') == 'black-forest-labs/FLUX.1-dev', 'Wrong official seed/model')
        prompt = self.prompt(row.get('prompt'))
        config = row.get('configuration') or {}
        native_metadata = None
        native_path = None
        if not config:
            for original in row.get('source_provenance', []):
                try:
                    possible = self.resolve(original)
                    if possible.suffix == '.json':
                        candidate = read(possible)
                        if candidate.get('source_commit') == OFFICIAL_COMMIT:
                            native_metadata, native_path = candidate, possible
                            break
                except (Rejected, OSError, ValueError):
                    continue
            require(native_metadata is not None, 'Missing archived invocation and surviving native metadata')
            source_hashes = native_metadata.get('official_source_hashes', native_metadata.get('source_hashes', {}))
            require(source_hashes.get('pipelines_ours/pipeline_spherical_flux.py') == OFFICIAL_PIPELINE_SHA,
                    'Legacy native official pipeline hash differs')
            self.prompt(native_metadata['prompt'])
            require(native_metadata.get('seed') == 0, 'Legacy native seed differs')
            config = native_metadata.get('spec') or dict(
                pipeline=native_metadata.get('pipeline_class'),
                model_source=native_metadata.get('checkpoint'),
                revision=native_metadata.get('revision'),
                explicit=native_metadata.get('explicit_call_arguments'))
        call = config.get('explicit', config.get('call'))
        if call is None:
            call = config.get('effective_call', config.get('explicit_call_arguments', config.get('spec', {}).get('call')))
        require(isinstance(call, dict) and call.get('num_inference_steps') == 28, 'Archived original invocation missing')
        require(call.get('n_spherical_points') == 26500 and call.get('erp_height') == 2048 and call.get('erp_width') == 4096,
                'Archived original geometry mismatch')
        require(call.get('guidance_scale') == 3.5 and call.get('true_cfg_scale') == 1.0 and call.get('weighted_average_temperature') == .1, 'Archived original guidance/fusion differs')
        schedule = row.get('schedule', {})
        require(len(schedule.get('timesteps', [])) == 28 and len(schedule.get('sigmas', [])) == 29, 'Archived schedule mismatch')
        efficiency = row.get('efficiency') or {}
        forwards = efficiency.get('transformer_calls', efficiency.get('transformer_forwards',
                   (native_metadata or config).get('transformer_forwards')))
        require(forwards == 89 * 28, 'Archived execution count mismatch')
        image = self.resolve(row['output_path'])
        self.image(image, row['image_sha256'])
        evidence = [dict(path=str(path), kind='preserved_verified_benchmark_inventory')]
        missing = []
        for original in row.get('source_provenance', []):
            try:
                source = self.resolve(original)
                self.hashed(source)
                evidence.append(dict(path=str(source), kind='surviving_original_provenance'))
            except Rejected:
                missing.append(original)
        notes = ['Official FLUX28 is a separate diagnostic baseline; it is never mixed into the matched-20 primary comparison.']
        if missing:
            notes.append('Original provenance files removed/moved: ' + '; '.join(missing) + '. Surviving previously verified inventory preserves invocation, completed accounting, schedule and image hash.')
        return self.record(prompt, 'sphere28', image, row['image_sha256'], path, evidence,
                           config, {}, 28, 0, 'official_native', 'official_native_spherical_26500',
                           'spherical', notes, row)

    def run(self):
        candidates, audit_paths = discover(self.resolver, self.output)
        print('INVENTORY_DISCOVERED', len(candidates), 'metadata files;', len(audit_paths), 'audit documents', flush=True)
        self.collect_audits(audit_paths)
        print('INVENTORY_AUDIT_INDEXED', len(self.proofs), 'metadata proofs', flush=True)
        accepted = []
        for path in candidates:
            try:
                meta = read(path)
                if meta.get('source_commit') == OFFICIAL_COMMIT and 'configuration' not in meta:
                    accepted.append(self.native_sphere(path, meta))
                else:
                    accepted.append(self.standard(path, meta))
            except (Rejected, OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                self.excluded.append(dict(metadata_path=str(path), reason=str(error)))
        print('INVENTORY_PRIMARY_CANDIDATES_VERIFIED', len(accepted), flush=True)
        for row, path in self.archived:
            if row.get('backend') != 'flux' or row.get('steps') != 28:
                continue
            try:
                accepted.append(self.archived_sphere(row, path))
            except (Rejected, OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                self.excluded.append(dict(metadata_path=str(path), historical_id=row.get('id'), reason=str(error)))
        selected, duplicates, ambiguities = select_duplicates(accepted)
        cohorts = assemble_cohorts(selected, self.scope)
        by_key = {(r['prompt_id'], r['method_id']): r for r in selected}
        missing = [dict(prompt_id=p['prompt_id'], prompt_name=p['prompt_name'], prompt_path=p['prompt_path'],
                        method_id=method, reason='No unambiguous completed artifact satisfying the declared configuration')
                   for p in self.scope for method in METHOD_IDS if (p['prompt_id'], method) not in by_key]
        result = dict(schema_version=1, root=str(self.root), output=str(self.output),
                      historical_roots=[str(p) for p in self.resolver.roots],
                      path_remappings=dict(self.resolver.remappings), prompt_scope=self.scope,
                      method_ids=list(METHOD_IDS), records=selected, cohorts=cohorts,
                      missing=missing, ambiguities=ambiguities,
                      counts=dict(selected_panoramas=len(selected), by_method={m: sum(r['method_id'] == m for r in selected) for m in METHOD_IDS},
                                  candidate_metadata_files=len(candidates), verified_candidates=len(accepted), missing_cases=len(missing)),
                      duplicate_policy='Collapse independently verified identical image hashes; exclude different image hashes within one method/prompt identity. No score-dependent selection.',
                      provenance_notes=['Historical sources are read-only. Missing original statuses/tensors do not invalidate PNGs independently bound to preserved completed audit hashes.',
                                        'The 15-prompt priority scope is read from mean_max_old89/common.py without importing generation code.'])
        self.output.mkdir(parents=True, exist_ok=True)
        atomic_json(self.output / 'inventory.json', result)
        columns = ['panorama_id', 'method_id', 'prompt_id', 'prompt_name', 'prompt_path', 'effective_prompt_sha256',
                   'image_path', 'image_sha256', 'metadata_path', 'metadata_sha256', 'backend', 'seed', 'steps', 'layout',
                   'consensus_projection', 'reference_mode', 'gradient_mix_alpha', 'configuration_sha256', 'checks', 'notes']
        write_csv(self.output / 'inventory.csv', selected, columns)
        matched = []
        for cohort, values in cohorts.items():
            for prompt in values['prompt_ids']:
                for method in values['method_ids']:
                    matched.append(dict(by_key[prompt, method], cohort=cohort))
        write_csv(self.output / 'matched_cases.csv', matched, ['cohort'] + columns)
        write_csv(self.output / 'missing_cases.csv', missing, ['prompt_id', 'prompt_name', 'prompt_path', 'method_id', 'reason'])
        atomic_json(self.output / 'provenance_audit.json', dict(schema_version=1, source_files=self.source_files,
                    audit_documents_discovered=[str(p) for p in audit_paths], selected_records=selected,
                    verified_candidate_count=len(accepted), excluded=self.excluded, duplicates=duplicates, ambiguities=ambiguities,
                    no_historical_writes=True, cleanup_policy=result['provenance_notes'][0]))
        return result


def select_duplicates(records):
    groups = defaultdict(list)
    for record in records:
        groups[record['prompt_id'], record['method_id']].append(record)
    selected, duplicates, ambiguities = [], [], []
    for (prompt, method), choices in sorted(groups.items()):
        hashes = {choice['image_sha256'] for choice in choices}
        if len(hashes) != 1:
            ambiguities.append(dict(prompt_id=prompt, method_id=method, image_sha256=sorted(hashes),
                                    candidate_paths=sorted({r['image_path'] for r in choices}),
                                    disposition='excluded_different_verified_images'))
            continue
        # Prefer the strongest independently hashed completion evidence; tie-break
        # by path is deterministic and independent of image content or VLM scores.
        choices.sort(key=lambda row: (-len(row.get('evidence', [])), row['metadata_path'], row['image_path']))
        chosen = dict(choices[0])
        chosen['identical_aliases'] = sorted({row['image_path'] for row in choices})
        chosen['metadata_paths'] = sorted({row['metadata_path'] for row in choices})
        chosen['independent_verification_count'] = len(chosen['metadata_paths'])
        selected.append(chosen)
        if len(choices) > 1:
            duplicates.append(dict(prompt_id=prompt, method_id=method, image_sha256=chosen['image_sha256'],
                                   selected_metadata=chosen['metadata_path'], identical_metadata=chosen['metadata_paths']))
    return selected, duplicates, ambiguities


def build_inventory(root: Path = ROOT, output: Path | None = None,
                    historical_roots: list[Path] | None = None,
                    path_remappings: dict[str, str] | None = None) -> dict[str, Any]:
    """Verify sources and atomically write inventory/cohort/provenance artifacts."""
    root = Path(root)
    output = Path(output) if output is not None else root / 'outputs/10.16spherediff-vlm-eval/seed0-v1'
    return InventoryBuilder(root, output, historical_roots or [], path_remappings).run()
