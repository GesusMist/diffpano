"""Authenticate the existing 15-scene max-blending comparison without regeneration.

October's artifact cleanup intentionally removed status JSON and raw tensors.
The pinned, completed figure provenance authenticates the surviving PNGs and
metadata; no absent status or tensor is reconstructed or treated as a failure.
"""
import copy
import hashlib
import json
from pathlib import Path
import struct


ROOT = Path('/home/shig/diffpano')
FIGURES = ROOT / 'outputs/10.16report_images/provenance/figures.json'
MAX_MANIFEST = ROOT / 'outputs/mean-max-old89-flux90/seed0-v1/manifest.json'
CONTROL_MANIFEST = ROOT / 'outputs/10.14mean-max-old89-controls/seed0-v1/manifest.json'
RETRY_MANIFEST = ROOT / 'outputs/10.15mean-max-firefly-retry/seed0-v1/manifest.json'
PRIOR_MANIFEST = ROOT / 'outputs/10.23fid-standalone85/seed0-v1/panorama-manifest.json'
PINNED = {
    FIGURES: '6ac2163d307a171baa0cc2fb66e4ed6fb34a16f6b3ec7ee8a08816dc52137e69',
    MAX_MANIFEST: '0ec0b4e2edec1dd75648b06e2e021f20ce40175053a217d582ad4792239d7935',
    CONTROL_MANIFEST: '7cd9014ae8f0cb4e35b1b00ad8cce5856b9fec7a613164f6d58eddcdf90d6dc9',
    RETRY_MANIFEST: '76ca5c984b765ced2a270dfc282dbb4673e6a8c17651fdee33dd81055d9de98e',
    PRIOR_MANIFEST: '7e860e69bf34feecdb28a1f564cbaa9b289d0bcbd1de910055c58b3c409dcb4c',
}
PROMPTS = (
    'air_balloons', 'aurora', 'blossom', 'clouds_over_grass', 'firefly',
    'firework', 'forest', 'rock_mountain', 'ruins', 'underwater', 'storm',
    'bamboo_grove', 'lantern_market', 'sailboat_marina', 'tiled_courtyard',
)
BASELINES = (
    ('spherediff_official_20', 'SphereDiff', None, None),
    ('diffpano_old89_rgb', 'RGB (ours)', 'rgb', None),
    ('diffpano_old89_coarse_select', 'coarse_color + select_closest (ours)',
     'poisson_select', 'coarse_color'),
    ('diffpano_old89_screened_select', 'lambda_blending + select_closest (ours)',
     'poisson_select', 'screened'),
)
COUNTS = {'denoiser': 1780, 'encode': 3560, 'decode': 1869, 'initialize': 0}
GEOMETRY = 'b55b4eb5051cdfb8c579e81d4f51ece1f15aeb0e8d7e1fde6b7b74a49154435c'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _png_shape(path):
    with Path(path).open('rb') as stream:
        header = stream.read(24)
    _require(header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR',
             'Expected PNG image: ' + str(path))
    return struct.unpack('>II', header[16:24])


def _verify_generated(panel, metadata, row, mode, reference, alpha):
    identity = row['case_id']
    for key in ('semantic_sha256', 'source_sha256'):
        _require(panel[key] == row[key] == metadata[key], identity + ': ' + key)
    _require(_digest(row['semantic']) == row['semantic_sha256'], identity + ': semantic digest')
    _require(metadata['case']['case_id'] == identity, identity + ': metadata case')
    _require(metadata['fusion'] == row['semantic']['solver'], identity + ': solver configuration')
    _require(metadata['poisson_solves'] == 21, identity + ': solve count')
    _require(metadata['refinement']['last_fraction'] == 0, identity + ': refinement')
    _require(metadata['FOV_x'] == metadata['FOV_y'] == 80, identity + ': FOV')
    _require(metadata['local_RGB_resolution'] == [1024, 1024], identity + ': RGB raster')
    _require(metadata['local_native_resolution'] == [128, 128], identity + ': native raster')
    _require(metadata['camera_count'] == 89, identity + ': camera count')
    _require(panel['gradient_mode'] == mode, identity + ': gradient mode')
    _require(panel['reference_mode'] == reference, identity + ': reference')
    _require(panel.get('gradient_mix_alpha') == alpha, identity + ': alpha')
    _require(metadata.get('tolerance_retry') == panel.get('tolerance_retry'), identity + ': retry record')


def build_inputs():
    """Return authenticated rows, prompt order, and immutable provenance hashes."""
    for path, expected in PINNED.items():
        _require(_sha(path) == expected, 'Pinned provenance changed: ' + str(path))
    figures = _read(FIGURES)
    manifest = _read(MAX_MANIFEST)
    control = _read(CONTROL_MANIFEST)
    retry = _read(RETRY_MANIFEST)
    prior = _read(PRIOR_MANIFEST)
    _require(figures['complete'] and figures['figure_count'] == 15, 'Incomplete figure inventory')
    _require(tuple(row['prompt'] for row in figures['figures']) == PROMPTS, 'Prompt set changed')
    _require(len(manifest['rows']) == 90 and len(retry['rows']) == 1, 'Generation manifest size changed')
    generated = {row['case_id']: row for row in manifest['rows'] + control['rows']}
    retry_row = retry['rows'][0]
    base_row = generated[retry_row['case_id']]
    restored = copy.deepcopy(retry_row['semantic'])
    restored['solver']['relative_tolerance'] = 1e-4
    _require(restored == base_row['semantic'], 'Retry changed more than authorized tolerance')
    _require(retry_row['semantic']['solver']['relative_tolerance'] == 2e-4, 'Retry tolerance changed')
    _require((retry_row['prompt'], retry_row['reference_mode'], retry_row['gradient_mix_alpha']) ==
             ('firefly', 'coarse_color', .25), 'Retry case changed')
    generated[retry_row['case_id']] = retry_row
    for generation in (manifest, control, retry):
        _require(all(row['source_sha256'] == _digest(generation['source_hashes'])
                     for row in generation['rows']), 'Frozen generation source digest mismatch')
    prior_by_key = {(row['prompt'], row['variant']): row for row in prior['rows']}
    original_prompt = {row['prompt']: row['semantic']['prompt'] for row in manifest['rows']}
    variants = list(BASELINES)
    for reference in ('coarse_color', 'screened'):
        label = 'lambda_blending' if reference == 'screened' else reference
        for alpha in (.25, .5, .75):
            tag = '%03d' % round(alpha * 100)
            variants.append((reference + '_max_' + tag,
                             label + ' + max_blending (ours)',
                             'poisson_mean_max', reference))
    rows, reused, generated_count = [], [], 0
    for figure in figures['figures']:
        prompt = figure['prompt']
        _require(len(figure['panels']) == 10, prompt + ': panel count')
        for index, (variant, label, mode, reference) in enumerate(variants):
            panel = figure['panels'][index]
            image = Path(panel['path'])
            metadata_path = Path(panel['metadata_path'])
            _require(_sha(image) == panel['image_sha256'], 'Image changed: ' + str(image))
            _require(_sha(metadata_path) == panel['metadata_sha256'], 'Metadata changed: ' + str(metadata_path))
            _require(_png_shape(image) == (4096, 2048), 'ERP dimensions changed: ' + str(image))
            metadata = _read(metadata_path)
            for key in ('original_sha256', 'effective_sha256'):
                _require(metadata['prompt'][key] == original_prompt[prompt][key], prompt + ': prompt ' + key)
            alpha = panel.get('gradient_mix_alpha')
            _require(alpha == ((.25, .5, .75)[(index - 4) % 3] if index >= 4 else None), prompt + ': alpha order')
            if 'artifacts' in metadata:
                _require(metadata['artifacts']['final.png'] == panel['image_sha256'], str(image) + ': artifact hash')
            if index == 0:
                _require(panel['method'] == 'spherediff' and panel['steps'] == 20 and panel['seed'] == 0,
                         prompt + ': original SphereDiff identity')
                if panel['source_kind'] == 'official_config':
                    _require(metadata['state'] == 'complete', prompt + ': SphereDiff state')
                    counts = metadata['execution']['counts']
                else:
                    counts = metadata['counts']
                _require(counts == {'denoiser': 1780, 'encode': 0, 'decode': 89}, prompt + ': SphereDiff counts')
            else:
                _require(metadata['counts'] == COUNTS, str(image) + ': call counts')
                _require(metadata['camera_geometry_sha256'] == GEOMETRY, str(image) + ': camera geometry')
                _require(metadata['fusion']['mode'] == mode, str(image) + ': fusion mode')
                if mode != 'rgb':
                    fusions = metadata['fusions']
                    _require(len(fusions) == 21 and fusions[-1]['terminal'], str(image) + ': final fusion')
                    for fusion in fusions:
                        _require(fusion['converged'] and fusion['mode'] == mode, str(image) + ': fusion convergence')
                        _require(fusion.get('reference_mode', 'screened') == reference, str(image) + ': fusion reference')
                        _require(fusion.get('gradient_mix_alpha') == alpha, str(image) + ': fusion alpha')
                if 'case_id' in panel:
                    generation_row = generated[panel['case_id']]
                    _require(generation_row['prompt'] == prompt, str(image) + ': case prompt')
                    _verify_generated(panel, metadata, generation_row, mode, reference, alpha)
                    generated_count += 1
                else:
                    _require(all(panel['checks'].values()), str(image) + ': historical validation')
            row = dict(variant=variant, prompt=prompt, image_path=str(image),
                       image_sha256=panel['image_sha256'], metadata_path=str(metadata_path),
                       metadata_sha256=panel['metadata_sha256'],
                       case_id=panel.get('case_id', prompt + '/' + variant),
                       reference_mode=reference, alpha=alpha, gradient_mix_alpha=alpha,
                       gradient_mode=mode, label=label, tolerance_retry=panel.get('tolerance_retry'),
                       provenance_title=panel['title'], generation_solver=metadata.get('fusion'))
            if index in (0, 1, 3):
                previous = prior_by_key[(prompt, variant)]
                _require(previous['image_sha256'] == row['image_sha256'], prompt + ': baseline FID input mismatch')
                row['prior_fid_case_id'] = previous['case_id']
                reused.append(dict(prompt=prompt, variant=variant, image_sha256=row['image_sha256']))
            rows.append(row)
    _require(len(rows) == 150 and len(reused) == 45, 'Unexpected result matrix')
    _require(len({(row['variant'], row['prompt']) for row in rows}) == 150, 'Duplicate comparison slots')
    return dict(prompts=list(PROMPTS), variants=[item[0] for item in variants], rows=rows,
                source_hashes={str(path): value for path, value in PINNED.items()},
                tolerance_override=figures['authorized_tolerance_override'],
                checks=dict(authenticated_images=150, authenticated_metadata=150,
                            full_generated_metadata_checks=generated_count,
                            prior_fid_overlap_images=45, prior_fid_overlap=reused,
                            historical_cleanup_policy='Pinned completed PNG and metadata evidence; no missing raw tensor or status required',
                            shared_prompts=15, max_blending_variants=6))
