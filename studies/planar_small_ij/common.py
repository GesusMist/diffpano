"""Aligned 256 RGB-pixel patch / 64 RGB-pixel stride follow-up to I/J."""
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from diffpano.config import load_experiment_config
from diffpano.planar import build_planar_patch_layout

ROOT = Path('outputs/planar-small-ij/20260918-p256-s64')
FACTORS = dict(sd2=8, sana=32, flux=8, sd35=8)
MANIFESTS = dict(I='configs/experiments/vae_residual/i-all-models.json',
                 J='configs/experiments/implied_endpoint_consensus/j-all-models.json')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def sources():
    paths = [p for folder in ('diffpano', 'scripts', 'studies/planar_small_ij')
             for p in Path(folder).rglob('*.py')]
    paths += [Path('prompts/native_control.txt')]
    return {str(p): sha(p) for p in sorted(paths)}


def spec_config(backend, method):
    spec = read(MANIFESTS[method])[backend]
    config = load_experiment_config(spec['config'])
    baseline = Path(spec['output'])/'generation/metadata.json'
    saved = read(baseline)
    if config.to_dict() != saved['config']:
        raise AssertionError('Historical config changed: '+str(baseline))
    old = copy.deepcopy(config.to_dict())
    factor = FACTORS[backend]
    config.native_multidiffusion.patch_size = 256//factor
    config.native_multidiffusion.stride = 64//factor
    config.validate()
    expected = copy.deepcopy(old)
    expected['native_multidiffusion'].update(patch_size=256//factor, stride=64//factor)
    assert config.to_dict() == expected
    return spec, config, baseline, saved


def make_manifest():
    rows = []
    for backend, factor in FACTORS.items():
        for method in ('I', 'J'):
            spec, c, baseline, saved = spec_config(backend, method)
            n = c.native_multidiffusion
            native = build_planar_patch_layout(n.canvas_height, n.canvas_width, n.patch_size, n.stride)
            rgb = build_planar_patch_layout(n.canvas_height*factor, n.canvas_width*factor, 256, 64)
            assert native.num_patches == rgb.num_patches == (65 if backend == 'sd2' else 377)
            assert all((a.y*factor,a.x*factor,a.size*factor)==(b.y,b.x,b.size)
                       for a,b in zip(native.patches,rgb.patches))
            rows.append(dict(backend=backend, method=method, config=c.to_dict(),
                baseline_metadata=str(baseline), baseline_metadata_sha256=sha(baseline),
                initial_native=spec['initial_native'], initial_file_sha256=sha(spec['initial_native']),
                conditioning_sha256=saved['consensus_audit']['conditioning_sha256'],
                native_geometry=asdict(native), rgb_geometry=asdict(rgb),
                expected_guided_predictions=native.num_patches*c.generation.num_inference_steps,
                output=str(ROOT/backend/method)))
    for b in FACTORS:
        a,j=[r for r in rows if r['backend']==b]
        assert a['initial_file_sha256']==j['initial_file_sha256']
        assert a['conditioning_sha256']==j['conditioning_sha256']
        ac,jc=copy.deepcopy(a['config']),copy.deepcopy(j['config'])
        ac.pop('consensus_transition'); jc.pop('consensus_transition')
        # Historical SD3.5 I/J retain different experiment names/output groups.
        ac['experiment'].pop('name'); jc['experiment'].pop('name')
        ac.pop('output'); jc.pop('output')
        assert ac==jc
    return dict(study='I/J planar small patches', rgb_patch_size=256, rgb_stride=64,
        schedule_policy='Existing backend prepare at actual local 256x256 resolution; identical within each I/J pair. FLUX dynamic shift is resolution-derived.',
        initial_policy='Reuse saved global native state, then exact aligned integer crops; no interpolation.',
        source_hashes=sources(), runs=rows)


def validated_row(backend, method):
    validation=read(ROOT/'validation.json')
    assert validation['passed'] and validation['source_hashes']==sources()
    assert validation['manifest_sha256']==sha(ROOT/'manifest.json')
    row=next(r for r in read(ROOT/'manifest.json')['runs'] if (r['backend'],r['method'])==(backend,method))
    assert sha(row['initial_native'])==row['initial_file_sha256']
    assert sha(row['baseline_metadata'])==row['baseline_metadata_sha256']
    spec,c,_,_=spec_config(backend,method)
    assert c.to_dict()==row['config']
    return row,c
