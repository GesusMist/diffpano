"""Render and publish the requested percentage-labeled refinement comparisons."""
import argparse
import os
from pathlib import Path
import shutil

from . import build

TARGET = build.OUT / 'refinement'


def render(workers):
    assert os.environ.get('SLURM_JOB_ID'), 'Render in a CPU Slurm allocation'
    import cv2
    from studies.mean_max_figures.pole_panels import validate_geometry
    from .refinement_percent import collect
    cv2.setNumThreads(1)
    stage = build.OUT / ('.refinement-percent-' + os.environ['SLURM_JOB_ID'])
    assert not stage.exists(), str(stage)
    source = collect(allow_mixed_steps=True)
    assert len(source['figures']) == 24
    stage.mkdir()
    inputs = stage / 'provenance' / 'previous-report'
    inputs.mkdir(parents=True)
    snapshots = {
        'comparison-index.json': build.OUT / 'comparison-index.json',
        'README.txt': build.OUT / 'README.txt',
        'refinement-manifest.json': TARGET / 'manifest.json',
        'refinement-sources.json': TARGET / 'provenance/sources.json',
        'uniform-manifest.json': build.OUT / 'uniform/manifest.json',
        'uniform-sources.json': build.OUT / 'uniform/provenance/sources.json',
        'history20-manifest.json': TARGET / 'history20/manifest.json',
    }
    hashes = {}
    for name, path in snapshots.items():
        shutil.copyfile(path, inputs / name)
        hashes[str(path)] = build.sha(path)
        assert build.sha(inputs / name) == hashes[str(path)]
    manifest = build.group_report('refinement_percent', source['figures'], stage,
                                  source['evidence'], validate_geometry(), workers)
    manifest['revision_builder_sha256'] = build.sha(__file__)
    manifest['revision_source_collector_sha256'] = build.sha(Path(__file__).with_name('refinement_percent.py'))
    manifest['previous_report_hashes'] = hashes
    manifest['display_percentages'] = [0, 20, 30]
    manifest['total_steps_by_column'] = [20, 28, 28]
    build.write(stage / 'manifest.json', manifest)
    print('REVISION_READY', stage, flush=True)


def rebase(value, before, after):
    if isinstance(value, str):
        return after + value[len(before):] if value == before or value.startswith(before + '/') else value
    if isinstance(value, list):
        return [rebase(item, before, after) for item in value]
    if isinstance(value, dict):
        return {key: rebase(item, before, after) for key, item in value.items()}
    return value


def publish(stage):
    assert stage.parent == build.OUT and stage.name.startswith('.refinement-percent-')
    manifest = build.read(stage / 'manifest.json')
    assert manifest['complete'] and manifest['figure_count'] == 24 and manifest['source_panel_count'] == 72
    for path, expected in manifest['previous_report_hashes'].items():
        assert build.sha(path) == expected, 'Prior report changed: ' + path
    assert {path.stem for path in stage.glob('*.png')} == {fig['prompt'] for fig in manifest['figures']}
    for figure in manifest['figures']:
        assert build.sha(stage / (figure['prompt'] + '.png')) == figure['sha256']
        assert figure['figure_size'] == [5828, 1012]
        assert [panel['display_percent'] for panel in figure['panels']] == [0, 20, 30]
        assert [panel['label'] for panel in figure['panels']] == [
            '0% (no refinement)', '20%', '30%']
        for panel in figure['panels']:
            assert build.sha(panel['image_path']) == panel['image_sha256']
            assert build.sha(panel['metadata_path']) == panel['metadata_sha256']
    archive = build.OUT / 'archive' / ('refinement-before-percent-' + manifest['job'])
    assert not archive.exists()
    # Rebase generated report paths before the directory is published. Source paths stay intact.
    for path in sorted((stage / 'provenance').glob('*.json')):
        build.write(path, rebase(build.read(path), str(stage), str(TARGET)))
    manifest = rebase(manifest, str(stage), str(TARGET))
    manifest['source_inventory_sha256'] = build.sha(stage / 'provenance/sources.json')
    manifest['previous_refinement_archive'] = str(archive)
    build.write(stage / 'manifest.json', manifest)
    previous_index = (build.OUT / 'comparison-index.json').read_text()
    index = build.read(build.OUT / 'comparison-index.json')
    matching = [group for group in index['groups'] if group['directory'] == str(TARGET)]
    assert len(matching) == 1
    matching[0].update(figure_count=24, source_panel_count=72,
                       manifest_sha256=build.sha(stage / 'manifest.json'))
    index['total_figures'] = sum(group['figure_count'] for group in index['groups'])
    index['total_verified_panels'] = sum(group['source_panel_count'] for group in index['groups'])
    index['refinement_revision'] = dict(job=manifest['job'], display_percentages=[0, 20, 30],
        total_steps_by_column=[20, 28, 28], previous_refinement_archive=str(archive),
        source_files={str(path.relative_to(build.ROOT)): build.sha(path) for path in
            [Path(__file__), Path(build.__file__), Path(__file__).with_name('refinement_percent.py')]})
    old_readme = (build.OUT / 'README.txt').read_text()
    new_readme = old_readme.replace(
        '24 matched 28-step comparisons: refinement 5/28 versus 8/28.',
        '24 comparisons: 0% (no refinement), 20%, 30%.').replace(
        'All 232 exported panels', 'All 256 exported panels')
    new_readme += ('\nRefinement revision: 20% and 30% are the requested display labels for the existing\n'
                   '5- and 8-step refinement tails. The 0% panel uses the existing 20-step baseline;\n'
                   'the refined panels use 28 total steps. Step counts are omitted from figure labels as requested.\n'
                   'Exact settings are retained in provenance. Previous figures are archived at\n'
                   + str(archive.relative_to(build.OUT)) + '/.\n'
                   'Revision builder: studies/report1023_comparisons/redo_refinement.py\n'
                   'Revision CPU log: logs/refinement-percent-' + manifest['job'] + '.log\n')
    assert new_readme != old_readme and index['total_figures'] == 52 and index['total_verified_panels'] == 256
    archive.parent.mkdir(exist_ok=True)
    TARGET.rename(archive)
    try:
        (archive / 'history20').rename(stage / 'history20')
        stage.rename(TARGET)
        build.write(build.OUT / 'comparison-index.json', index)
        build.atomic_text(build.OUT / 'README.txt', new_readme)
        for group in index['groups']:
            assert build.sha(group['manifest']) == group['manifest_sha256']
    except BaseException:
        if TARGET.exists():
            TARGET.rename(stage)
        if (stage / 'history20').exists():
            (stage / 'history20').rename(archive / 'history20')
        archive.rename(TARGET)
        build.atomic_text(build.OUT / 'comparison-index.json', previous_index)
        build.atomic_text(build.OUT / 'README.txt', old_readme)
        raise
    print('REVISION_PUBLISHED', TARGET, '24 figures / 72 panels', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--publish', type=Path)
    args = parser.parse_args()
    assert 1 <= args.workers <= 4
    if args.publish:
        publish(args.publish)
    else:
        render(args.workers)


if __name__ == '__main__':
    main()
