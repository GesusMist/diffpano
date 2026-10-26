"""Authenticate the 120 standalone FLUX results and retain 85 unique PNGs.

This module never depends on status.json: those files were intentionally pruned.
The frozen generation index and the independent FASTER migration audit pin the
metadata and image hashes, while metadata pins config.json. No images are copied,
regenerated, weighted, or scored here. All duplicate aliases remain in the output
so the evaluation plan can state its panorama-view sampling policy explicitly.
"""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import struct
import zlib

ROOT = Path('/home/shig/diffpano')
STANDALONE = ROOT / 'outputs/10.15perspective24-flux/seed0-v1'
MANIFEST = STANDALONE / 'manifest.json'
INDEX = STANDALONE / 'index.json'
COMPLETION = STANDALONE / 'completion.json'
MIGRATION_AUDIT = ROOT / 'outputs/10.16faster-reconstruction-20261016/artifacts/audit.json'
PINNED = {
    MANIFEST: '6f3bf41cde03240288c86d0ccf3b7903e5163f4429006a2a7d97ccf1659ad5fc',
    INDEX: '381e515433ed6725ddcccce39e3591b0d3746197cb6f94d4edb1a52b6134f903',
    COMPLETION: 'dbe1d4d2fc8b01b2849dcefbd294cbeeebbba747be58521afeecc0262686df59',
    MIGRATION_AUDIT: '71af5739905474bcf05717104a27a712168b7919c5897c74d83f054146865d0e',
}
SOURCE_SHA256 = 'eefbda67f3c41bd234be0b81365224b47e87a4f741701d3a7790286664495261'
COUNTS = {'denoiser': 20, 'encode': 0, 'decode': 1, 'initialize': 1}
LABELS = ('01_up', '02_slightly_up', '03_eye_level', '04_slightly_down', '05_down')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def png_header(path):
    """Read the authenticated PNG's dimensions/mode without importing imaging ML."""
    with Path(path).open('rb') as stream:
        header = stream.read(33)
    require(len(header) == 33 and header[:8] == b'\x89PNG\r\n\x1a\n',
            'Invalid PNG signature: ' + str(path))
    require(header[8:16] == b'\0\0\0\rIHDR', 'Invalid PNG IHDR: ' + str(path))
    require(zlib.crc32(header[12:29]) & 0xffffffff == struct.unpack('>I', header[29:33])[0],
            'Invalid PNG IHDR CRC: ' + str(path))
    width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', header[16:29])
    require((width, height, depth, color, compression, filtering, interlace)
            == (1024, 1024, 8, 2, 0, 0, 0), 'Unexpected reference PNG format: ' + str(path))
    return {'width': width, 'height': height, 'mode': 'RGB', 'bit_depth': depth}


def build_reference_manifest():
    """Return 85 equally counted unique references plus all 120 audited aliases.

    ``rows`` is ordered by each group's lowest original generation index, with
    ``image``/``image_sha256`` pointing to that representative. ``aliases`` records
    every original scene, directional line, path, and artifact hash. Deduplication
    is exact PNG-byte SHA256; all groups also have exactly one prompt-line SHA256.
    """
    for path, expected in PINNED.items():
        require(sha256(path) == expected, 'Pinned reference evidence changed: ' + str(path))
    manifest, index, completion, audit = (read_json(path) for path in
                                        (MANIFEST, INDEX, COMPLETION, MIGRATION_AUDIT))
    require(len(manifest['rows']) == 120 and len(index['rows']) == 120 and index['valid'] == 120,
            'Standalone generation/index must contain 120 valid references')
    require(completion['expected'] == 120 and completion['counts']['complete'] == 120
            and not completion['unfinished'], 'Original standalone suite was not complete')
    require(os.readlink(ROOT / 'outputs') == manifest['outputs_symlink'], 'Outputs symlink changed')
    require(digest(manifest['source_hashes']) == SOURCE_SHA256, 'Standalone source identity changed')
    for relative, expected in manifest['source_hashes'].items():
        require(sha256(ROOT / relative) == expected, 'Frozen standalone source changed: ' + relative)
    for path, expected in manifest['input_artifacts'].items():
        require(sha256(path) == expected, 'Frozen standalone input changed: ' + path)
    for name, prompt in manifest['prompts'].items():
        require(sha256(prompt['path']) == prompt['original_sha256'] == prompt['effective_sha256'],
                'Frozen standalone prompt changed: ' + name)
    require(not audit['validation_failures'] and not audit['historical_outputs_modified'],
            'Migration audit did not validate original results')
    require(audit['perspective_inventory']['exact_index_set']
            and audit['perspective_inventory']['expected'] == 120,
            'Migration audit has an incomplete perspective inventory')
    migrated = {}
    for record in audit['file_hash_results']:
        if record['label'] in ('A_image', 'A_metadata'):
            key = (record['label'], str(Path(record['path']).resolve()))
            require(key not in migrated, 'Duplicate migration hash record: ' + str(key))
            migrated[key] = record
    require(len(migrated) == 240, 'Migration audit must pin 120 images and 120 metadata files')
    indexed = {row['index']: row for row in index['rows']}
    require(set(indexed) == set(range(120)), 'Invalid standalone index set')
    aliases, grouped = [], defaultdict(list)
    for row in sorted(manifest['rows'], key=lambda value: value['index']):
        i = row['index']
        record = indexed[i]
        directory = STANDALONE / 'cases' / row['case_id']
        image, meta_path, config_path = (directory / name for name in ('final.png', 'metadata.json', 'config.json'))
        require(record['valid'] is True and record['scene'] == row['prompt']
                and record['direction_label'] == row['direction_label']
                and record['line_sha256'] == row['line_sha256']
                and record['line_text'] == row['line_text']
                and record['source_group'] == row['source_group']
                and record['line_index'] == row['line_index']
                and record['image'] == str(image) and record['metadata'] == str(meta_path),
                'Frozen generation/index identity mismatch: ' + row['case_id'])
        require(row['source_sha256'] == SOURCE_SHA256
                and digest(row['semantic']) == row['semantic_sha256'],
                'Frozen semantic identity mismatch: ' + row['case_id'])
        require(hashlib.sha256(row['line_text'].encode()).hexdigest() == row['line_sha256'],
                'Prompt line hash mismatch: ' + row['case_id'])
        for label, path, expected in (('A_image', image, record['image_sha256']),
                                      ('A_metadata', meta_path, record['metadata_sha256'])):
            require(sha256(path) == expected, 'Reference artifact changed: ' + str(path))
            migration = migrated[(label, str(path.resolve()))]
            require(migration['hash_matches'] is True
                    and migration['actual_sha256'] == migration['expected_sha256'] == expected
                    and migration['bytes'] == path.stat().st_size,
                    'Migration/artifact mismatch: ' + str(path))
        metadata, config = read_json(meta_path), read_json(config_path)
        require(metadata['semantic_sha256'] == row['semantic_sha256']
                and metadata['source_sha256'] == row['source_sha256']
                and metadata['source_hashes'] == manifest['source_hashes']
                and metadata['case'] == {k: v for k, v in row.items() if k != 'semantic'},
                'Reference metadata identity mismatch: ' + row['case_id'])
        require(set(metadata['artifacts']) == {'config.json', 'final.png'}
                and metadata['artifacts']['final.png'] == record['image_sha256']
                and sha256(config_path) == metadata['artifacts']['config.json'],
                'Reference config/image hash mismatch: ' + row['case_id'])
        require(metadata['counts'] == COUNTS and metadata['guided_steps'] == 20
                and metadata['native_intervals'] == 20
                and metadata['initialization']['seed'] == 0
                and metadata['initialization']['fresh_generator_per_image'] is True
                and metadata['line_sha256'] == row['line_sha256']
                and metadata['line_text'] == row['line_text']
                and metadata['panorama_trajectory'] is False
                and metadata['fusion_stages'] == 0 and metadata['poisson_solves'] == 0,
                'Reference generation protocol mismatch: ' + row['case_id'])
        require({key: config[key] for key in row['semantic']} == row['semantic']
                and config['conditioning_sha256'] == metadata['conditioning_sha256']
                and config['line_sha256'] == row['line_sha256'],
                'Reference config semantics mismatch: ' + row['case_id'])
        header = png_header(image)
        alias = {
            'index': i, 'case_id': row['case_id'], 'scene': row['prompt'], 'prompt': row['prompt'],
            'source_group': row['source_group'], 'direction_label': row['direction_label'],
            'line_index': row['line_index'], 'line_text': row['line_text'],
            'line_sha256': row['line_sha256'], 'prompt_path': row['prompt_path'],
            'prompt_sha256': row['original_prompt_sha256'], 'image': str(image), 'image_path': str(image),
            'image_sha256': record['image_sha256'], 'image_bytes': image.stat().st_size,
            'metadata': str(meta_path), 'metadata_sha256': record['metadata_sha256'],
            'config': str(config_path), 'config_sha256': metadata['artifacts']['config.json'],
            'semantic_sha256': row['semantic_sha256'], 'source_sha256': row['source_sha256'],
            **header,
        }
        aliases.append(alias)
        grouped[alias['image_sha256']].append(alias)
    require(len(grouped) == 85, 'Expected exactly 85 unique reference PNG hashes')
    require(len({row['line_sha256'] for row in aliases}) == 85,
            'Expected exactly 85 distinct standalone prompt lines')
    rows = []
    for group in sorted(grouped.values(), key=lambda members: members[0]['index']):
        require(len({row['line_sha256'] for row in group}) == 1,
                'An identical image was found under different prompt lines')
        canonical = group[0]
        rows.append({**canonical, 'reference_index': len(rows),
                     'canonical_source_index': canonical['index'], 'canonical_index': canonical['index'],
                     'alias_count': len(group), 'aliases': group})
    require(set(row['index'] for row in aliases) == set(range(120)), 'Alias coverage is incomplete')
    return {
        'schema_version': 1, 'backend': 'flux', 'reference_kind': 'standalone_synthetic',
        'original_count': 120, 'logical_count': 120, 'unique_count': 85, 'duplicate_files_removed': 35,
        'deduplication': 'exact original PNG bytes grouped by SHA256; each unique group counted once',
        'representative_rule': 'lowest original frozen standalone manifest index',
        'source_sha256': SOURCE_SHA256, 'source_hashes': manifest['source_hashes'],
        'manifest_path': str(MANIFEST), 'manifest_sha256': PINNED[MANIFEST],
        'evidence': [{'path': str(path), 'sha256': value} for path, value in PINNED.items()],
        'validation': {'source_files': len(manifest['source_hashes']), 'png_files': 120,
                       'metadata_files': 120, 'config_files': 120, 'prompt_files': 24,
                       'png_format': '1024x1024 uint8 RGB PNG', 'migration_hashes_rechecked': 240,
                       'status_files_required': False, 'historical_outputs_modified': False},
        'alias_multiplicities': {str(size): count for size, count in sorted(Counter(map(len, grouped.values())).items())},
        'canonical_direction_counts': dict(sorted(Counter(row['direction_label'] for row in rows).items())),
        'canonical_scene_counts': dict(sorted(Counter(row['scene'] for row in rows).items())),
        'original_direction_counts': dict(sorted(Counter(row['direction_label'] for row in aliases).items())),
        'rows': rows, 'all_aliases': aliases,
    }


if __name__ == '__main__':
    result = build_reference_manifest()
    print(json.dumps({key: value for key, value in result.items() if key not in ('rows', 'all_aliases', 'source_hashes')},
                     indent=2, sort_keys=True))
