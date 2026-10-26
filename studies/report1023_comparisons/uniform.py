"""Collect authenticated, available Uniform comparisons without writing outputs."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = (
    ('spherediff_official_20', 'SphereDiff (original)'),
    ('diffpano_old89_rgb', 'Old89 RGB (ours)'),
    ('diffpano_old70_rgb', 'Old70 RGB (ours)'),
    ('diffpano_old50_rgb', 'Old50 RGB (ours)'),
    ('diffpano_uniform89_rgb', 'Uniform89 RGB (ours)'),
    ('diffpano_uniform70_rgb', 'Uniform70 RGB (ours)'),
    ('diffpano_uniform50_rgb', 'Uniform50 RGB (ours)'),
    ('diffpano_old89_screened_select', 'Old89 lambda_blending + select_closest (ours)'),
    ('diffpano_uniform89_screened_select', 'Uniform89 lambda_blending + select_closest (ours)'),
)
OPTIONAL = frozenset(('diffpano_old70_rgb', 'diffpano_old50_rgb'))
FULL_CONTROL_PROMPTS = frozenset(('air_balloons', 'firework', 'ruins', 'underwater'))


def collect():
    """Return 24 figures with seven or nine verified source panels per prompt.

    The pinned migration inventory authenticates retained historical PNGs and
    metadata after deliberate cleanup. Fresh Uniform results must satisfy the
    original strict completion check, including their recorded artifacts.
    """
    from collections import Counter
    from PIL import Image
    from studies.faster_uniform_resume.completion import (
        INVENTORY, INVENTORY_SHA256, MANIFEST_SHA256, SUITE, load_verifier,
    )
    from studies.followup24.common import PROMPTS24, prompt
    from studies.layout24 import common as original

    verifier = load_verifier(strict_complete=original.complete)
    manifest_path = SUITE / 'manifest.json'
    manifest = original.read(manifest_path)
    assert original.sha(manifest_path) == MANIFEST_SHA256
    rows = {(row['prompt'], row['variant']): row for row in manifest['rows']}
    assert len(rows) == 216
    assert set(rows) == {(name, variant) for name in PROMPTS24 for variant, _ in VARIANTS}
    figures, counts = [], Counter()
    protected_count = fresh_count = 0
    for name in PROMPTS24:
        info, raw = prompt(name)
        assert info['original_sha256'] == info['effective_sha256'] and info['line_count'] == 5
        panels, missing = [], []
        for variant, label in VARIANTS:
            row = rows[(name, variant)]
            assert row['semantic']['prompt'] == info, 'Prompt changed: ' + name
            assert row['prompt_path'] == info['full_path']
            assert row['backend'] == 'flux' and row['steps'] == 20 and row['seed'] == 0
            assert row['last_fraction'] == 0.0 and not row['time_travel'] and not row['camera_rotation']
            directory = SUITE / 'cases' / row['case_id']
            image_path, metadata_path = directory / 'final.png', directory / 'metadata.json'
            valid = verifier.checked_complete(row, verify=True)
            if not valid:
                if variant not in OPTIONAL:
                    raise ValueError('Required comparison source is incomplete: ' + row['case_id'])
                if image_path.exists() or image_path.is_symlink():
                    raise ValueError('Existing optional image failed authentication: ' + str(image_path))
                missing.append(dict(key=variant, label=label, reason='No verified completed result'))
                continue
            metadata = original.read(metadata_path)
            image_hash, metadata_hash = original.sha(image_path), original.sha(metadata_path)
            assert metadata['artifacts']['final.png'] == image_hash
            if row['index'] in verifier.protected_records:
                proof = verifier.protected_records[row['index']]
                assert proof['image_sha256_recorded'] == image_hash
                assert proof['metadata_sha256_recorded'] == metadata_hash
                authentication = 'pinned retained historical image and metadata'
                protected_count += 1
            else:
                status = original.read(directory / 'status.json')
                assert status['metadata_sha256'] == metadata_hash
                assert status['image_sha256'] == image_hash
                assert metadata['configuration'] == row['semantic']['configuration']
                authentication = 'original strict completion and all artifact hashes'
                fresh_count += 1
            with Image.open(image_path) as image:
                assert image.format == 'PNG' and image.mode == 'RGB' and image.size == (4096, 2048)
                image.verify()
            panels.append(dict(
                key=variant, label=label, image_path=str(image_path.absolute()),
                image_sha256=image_hash, metadata_path=str(metadata_path.absolute()),
                metadata_sha256=metadata_hash,
                scientific=dict(index=row['index'], case_id=row['case_id'],
                    semantic_sha256=row['semantic_sha256'], source_sha256=row['source_sha256'],
                    backend='flux', steps=20, seed=0, camera_count=row['camera_count'],
                    layout=row['layout'], gradient_mode=row['gradient_mode'],
                    reference_mode=row['reference_mode'], refinement_fraction=0.0,
                    authentication=authentication)))
            counts[variant] += 1
        expected_count = 9 if name in FULL_CONTROL_PROMPTS else 7
        assert len(panels) == expected_count and len(missing) == 9 - expected_count
        figures.append(dict(
            kind='uniform', prompt=name, title=name.replace('_', ' ').capitalize(),
            subtitle='FLUX | 20 steps | seed 0', prompt_path=info['full_path'],
            prompt_sha256=info['original_sha256'], panels=panels, missing=missing))
    assert len(figures) == 24 and sum(len(figure['panels']) for figure in figures) == 176
    assert protected_count == 101 and fresh_count == 75
    previous_builder = ROOT / 'outputs/10.16report_images/uniform/provenance/build.py'
    evidence = dict(
        manifest_path=str(manifest_path), manifest_sha256=MANIFEST_SHA256,
        migration_inventory_path=str(INVENTORY), migration_inventory_sha256=INVENTORY_SHA256,
        verifier_path=str(ROOT / 'studies/faster_uniform_resume/completion.py'),
        verifier_sha256=original.sha(ROOT / 'studies/faster_uniform_resume/completion.py'),
        previous_builder_path=str(previous_builder), previous_builder_sha256=original.sha(previous_builder),
        prompt_count=24, verified_panel_count=176, protected_panel_count=protected_count,
        fresh_panel_count=fresh_count, per_variant=dict(counts),
        full_control_prompts=sorted(FULL_CONTROL_PROMPTS), omitted_control_panels=40,
        missing_panel_policy='Omit missing Old70/Old50 panels; retain every verified available control',
        historical_cleanup_policy='Pinned retained image/metadata hashes authorize historical results; pruned source logs and tensors are not required',
        image_protocol='Original RGB PNG, 4096 by 2048, seed 0, 20 denoising steps',
        source_images_modified=False, generation_performed=False)
    return dict(figures=figures, evidence=evidence)
