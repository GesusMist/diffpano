"""Pole metadata capture and certification of this exact metadata-only revision."""
import ast
import math
from studies.tt_cea.common import REPO, ROOT, read, sha, study_hashes

ARCHIVE = ROOT / 'provenance/pre-pole-metadata'
PRIOR_VALIDATION_SHA256 = 'e67d31333c5b2b77b5fb2d96d13b0f7aede14c32655d2c116eefb2eb9867e141'
SOURCE_SNAPSHOT_SHA256 = '4ef6d3eed257e25b985fc1261a25ec727ed0d2cd0846b824321c1d063173b404'


def validate_pole_record(value):
    assert isinstance(value, dict) and set(value) == {'north', 'south'}
    for color in value.values():
        assert isinstance(color, list) and len(color) == 3
        assert all(type(x) in (int, float) and math.isfinite(x) for x in color)
    return value


def terminal_pole_rgb(native):
    """Copy actual floating pole colors, in raw, unclipped RGB [-1,1] convention."""
    import torch
    assert native.pole_rgb is not None
    poles = native.pole_rgb.detach().cpu()
    assert tuple(poles.shape) == (1, 3, 2, 1) and poles.dtype == torch.float32
    assert bool(torch.isfinite(poles).all())
    return validate_pole_record(dict(north=poles[0, :, 0, 0].tolist(),
                                    south=poles[0, :, 1, 0].tolist()))


def assert_capture_only_run_change(before, after):
    tree = ast.parse(after)
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)
               and n.module == 'studies.tt_cea.metadata']
    assert len(imports) == 1
    assert [(n.name, n.asname) for n in imports[0].names] == [('terminal_pole_rgb', None)]
    tree.body.remove(imports[0])
    hits = []
    expected = ast.parse("terminal_pole_rgb(native) if row['projection']=='cea' else None", mode='eval').body
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in list(node.keywords):
                if keyword.arg == 'terminal_pole_rgb':
                    assert isinstance(node.func, ast.Name) and node.func.id == 'dict'
                    assert ast.dump(keyword.value) == ast.dump(expected)
                    node.keywords.remove(keyword)
                    hits.append(keyword)
    assert len(hits) == 1
    assert ast.dump(tree) == ast.dump(ast.parse(before)), 'Generation AST changed beyond pole capture'


def revised_common_source(before):
    header = 'def require_gate(backend=None,case=None):\n'
    assert before.count(header) == 1
    result = before.replace(header, header + '    from studies.tt_cea.metadata import compatible_source_hashes\n')
    for name in ('p', 'g'):
        source = name + "['source_hashes']==study_hashes()"
        assert result.count(source) == 1
        result = result.replace(source, name + "['source_hashes'] in compatible_source_hashes(v)")
    return result


def certify_metadata_revision():
    import hashlib
    prior_path = ARCHIVE / 'records/validation.json'
    snapshot_path = ARCHIVE / 'source_snapshot.json'
    assert sha(prior_path) == PRIOR_VALIDATION_SHA256
    assert sha(snapshot_path) == SOURCE_SNAPSHOT_SHA256
    prior = read(prior_path)
    snapshot = read(snapshot_path)
    assert prior['passed'] and prior['source_hashes'] == snapshot['source_hashes']
    assert prior['manifest_sha256'] == sha(ROOT / 'manifest.json')
    for path, source in snapshot['source_text'].items():
        assert hashlib.sha256(source.encode()).hexdigest() == prior['source_hashes'][path]
    current = study_hashes()
    added = {'studies/tt_cea/metadata.py', 'studies/tt_cea/tests/test_metadata.py',
             'studies/tt_cea/recover_poles.py', 'studies/tt_cea/recover_poles.slurm'}
    changed = {'studies/tt_cea/common.py', 'studies/tt_cea/run.py',
               'studies/tt_cea/report.py', 'studies/tt_cea/validate.py'}
    assert set(current) == set(prior['source_hashes']) | added
    protected = sorted(set(prior['source_hashes']) - changed)
    for path in protected:
        assert current[path] == prior['source_hashes'][path], 'Protected source changed: ' + path
    assert_capture_only_run_change(snapshot['source_text']['studies/tt_cea/run.py'],
                                   (REPO / 'studies/tt_cea/run.py').read_text())
    assert (REPO / 'studies/tt_cea/common.py').read_text() == revised_common_source(
        snapshot['source_text']['studies/tt_cea/common.py'])
    def render_functions(source):
        return {n.name: ast.dump(n) for n in ast.parse(source).body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name in ('panel', 'baseline_views', 'viewport', 'comparisons')}
    assert render_functions(snapshot['source_text']['studies/tt_cea/report.py']) == render_functions(
        (REPO / 'studies/tt_cea/report.py').read_text())
    return dict(kind='terminal_CEA_pole_metadata_only', prior_validation_sha256=PRIOR_VALIDATION_SHA256,
                source_snapshot_sha256=SOURCE_SNAPSHOT_SHA256, prior_source_hashes=prior['source_hashes'],
                current_source_hashes=current, protected_sources_unchanged=protected,
                generation_AST_unchanged_except_metadata=True,
                common_changes_limited_to_provenance_checks=True, report_rendering_unchanged=True)


def compatible_source_hashes(validation):
    current = study_hashes()
    assert validation['passed'] and validation['source_hashes'] == current
    accepted = [current]
    certificate = validation.get('metadata_only_compatibility')
    if certificate is not None:
        assert certificate == certify_metadata_revision()
        accepted.append(certificate['prior_source_hashes'])
    return accepted



def assert_same_recovery_sample(original,candidate):
    for key in ('backend','prompt','case','generation_configuration','model_checkpoint','model_revision',
                'camera_geometry_sha256','local_native_resolution','local_RGB_resolution','native_channels',
                'native_scale','prepared_intervals','conditioning_sha256','counts','manifest_sha256'):
        assert original[key]==candidate[key], 'Recovery configuration mismatch: '+key
    assert original['initialization']['initial_local_sha256']==candidate['initialization']['initial_local_sha256']
    assert original['diagnostics']['terminal_state_sha256']==candidate['diagnostics']['terminal_state_sha256']
    for file in ('final.png','final_cea.png','viewports.png','schedule.csv'):
        assert original['artifacts'][file]==candidate['artifacts'][file], 'Recovery artifact mismatch: '+file
    validate_pole_record(candidate['terminal_pole_rgb'])


def recovered_pole_record(backend,prompt,case,original):
    if (prompt,case)!=('ruins','C0'):return None
    directory=ROOT/'pole-metadata-recovery'/backend
    certificate_path=directory/'certificate.json'
    if not certificate_path.exists():return None
    certificate=read(certificate_path)
    if not certificate['passed']:return None
    original_folder=ROOT/'samples'/prompt/backend/case
    candidate_folder=directory/'attempt-1'
    candidate=read(candidate_folder/'metadata.json');status=read(candidate_folder/'status.json')
    assert certificate['original_metadata_sha256']==sha(original_folder/'metadata.json')
    assert certificate['candidate_metadata_sha256']==sha(candidate_folder/'metadata.json')==status['metadata_sha256']
    assert status['state']=='complete'
    assert candidate['source_hashes'] in compatible_source_hashes(read(ROOT/'validation.json'))
    assert_same_recovery_sample(original,candidate)
    for file in ('final.png','final_cea.png','viewports.png','schedule.csv'):
        assert sha(candidate_folder/file)==candidate['artifacts'][file]
    assert certificate['terminal_pole_rgb']==candidate['terminal_pole_rgb']
    return validate_pole_record(candidate['terminal_pole_rgb'])

if __name__ == '__main__':
    import json
    print(json.dumps(certify_metadata_revision(), indent=2))
