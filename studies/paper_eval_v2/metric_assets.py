"""Pinned, isolated metric assets; no mutation of the shared evaluation environment."""
import importlib
import json
import os
import sys
import types
import urllib.request
from pathlib import Path
from studies.panorama_metrics.common import sha, atomic, read, ASSETS

PYIQA_REVISION = 'a3eecfbf33bd9313ac613df9bc80ccbba94f28e5'
GOOGLE_REVISION = 'e6890f85757dd84e27ca6df2dd30651dafad28e0'
PYIQA_FILES = ('pyiqa/archs/musiq_arch.py', 'pyiqa/archs/arch_util.py',
    'pyiqa/archs/constants.py', 'pyiqa/matlab_utils/padding.py',
    'pyiqa/data/multiscale_trans_util.py', 'pyiqa/utils/registry.py',
    'pyiqa/utils/download_util.py', 'pyiqa/utils/misc.py', 'pyiqa/utils/dist_util.py',
    'scripts/convert_musiq_to_pth.py', 'LICENSE')
OFFICIAL_URL = 'https://storage.googleapis.com/gresearch/musiq/koniq_ckpt.npz'
PORT_URL = 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_koniq_ckpt-e95806b9.pth'


def _download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(path.name+'.partial')
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as output:
            if 'text/html' in response.headers.get('Content-Type', ''):
                raise ValueError('Unexpected HTML checkpoint response')
            while True:
                block = response.read(4*1024*1024)
                if not block:
                    break
                output.write(block)
        os.replace(temporary, path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, url=url)


def prepare(run):
    root = Path(run).resolve()/'group1/assets'
    root.mkdir(parents=True, exist_ok=True)
    original = read(ASSETS/'asset_status.json')
    result = {name: original[name] for name in ('inception', 'clip')}
    for value in result.values():
        assert value['status'] == 'available' and sha(value['path']) == value['sha256']
    try:
        sources = {}
        for relative in PYIQA_FILES:
            url = 'https://raw.githubusercontent.com/chaofengc/IQA-PyTorch/'+PYIQA_REVISION+'/'+relative
            sources[relative] = _download(url, root/'pyiqa_source'/relative)
        official = _download(OFFICIAL_URL, root/'koniq_ckpt.npz')
        converted = _download(PORT_URL, root/'musiq_koniq_ckpt-e95806b9.pth')
        assert converted['sha256'].startswith('e95806b9')
        wheel_meta = json.load(urllib.request.urlopen('https://pypi.org/pypi/huggingface-hub/0.25.2/json', timeout=30))
        wheel = next(value for value in wheel_meta['urls'] if value['filename'].endswith('.whl'))
        dependency = _download(wheel['url'], root/wheel['filename'])
        assert dependency['sha256'] == '1897caf88ce7f97fe0110603d8f66ac264e3ba6accdf30cd66cc0fed5282ad25'
        yaml_meta = json.load(urllib.request.urlopen('https://pypi.org/pypi/PyYAML/6.0.2/json', timeout=30))
        yaml_wheel = next(value for value in yaml_meta['urls'] if 'cp311-cp311-manylinux_2_17_x86_64' in value['filename'])
        yaml_dependency = _download(yaml_wheel['url'], root/yaml_wheel['filename'])
        assert yaml_dependency['sha256'] == '3ad2a3decf9aaba3d29c8f537ac4b243e36bef957511b4766cb0057d32b0be85'
        result['musiq'] = dict(dependency_wheels=[dependency,yaml_dependency], status='weights_available_validation_pending', official=official, converted=converted,
            code_revision=PYIQA_REVISION, official_code_revision=GOOGLE_REVISION,
            source_root=str(root/'pyiqa_source'), source_files=sources,
            implementation='PyIQA port of official Google MUSIQ, KonIQ-10k checkpoint; upstream files unmodified',
            preprocessing=dict(input='RGB float32 [0,1], 512x512 viewport', internal_range=[-1,1],
                patch_size=32, patch_stride=32, hse_grid_size=10, longer_side_lengths=[224,384],
                max_seq_len_from_original_res=-1), scale='original KonIQ MOS scale, no normalization')
    except Exception as error:
        result['musiq'] = dict(status='blocked_asset_download', error=str(error), official_url=OFFICIAL_URL, converted_url=PORT_URL)
    result['faed'] = dict(status='blocked_missing_verified_checkpoint',
        source='https://github.com/chengzhag/PanFusion',
        checkpoint_url=original['faed']['url'], latest_check='2026-10-26: official author endpoint HTTP403',
        preprocessing='full ERP1024x512; RGB[-1,1]; official RGB encoder; longitude average; cosine latitude weights; flatten')
    result['distort_fid'] = dict(status='blocked_upstream_release',
        source='https://github.com/iSEE-Laboratory/PanoDecouple', latest_check='2026-10-26: official repository contains LICENSE and README only')
    atomic(root/'status.json', result)
    return result


def load_port(asset):
    """Load only MUSIQ dependencies; skip unrelated IQA package auto-discovery."""
    root = Path(asset['source_root'])
    for wheel in asset.get('dependency_wheels', []):
        assert sha(wheel['path']) == wheel['sha256']
        if str(Path(wheel['path']).resolve()) not in sys.path:
            sys.path.insert(0, str(Path(wheel['path']).resolve()))
    for relative, evidence in asset['source_files'].items():
        assert sha(root/relative) == evidence['sha256'], relative
    # Namespace packages retain all upstream modules unchanged while avoiding
    # __init__ importing dozens of unrelated models and optional dependencies.
    for name in ('pyiqa', 'pyiqa.archs', 'pyiqa.data', 'pyiqa.utils', 'pyiqa.matlab_utils'):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(root/Path(*name.split('.')))]
            sys.modules[name] = package
    padding = importlib.import_module('pyiqa.matlab_utils.padding')
    for name in ('ExactPadding2d', 'exact_padding_2d', 'symm_pad'):
        setattr(sys.modules['pyiqa.matlab_utils'], name, getattr(padding, name))
    return importlib.import_module('pyiqa.archs.musiq_arch').MUSIQ


def verify_musiq(asset):
    """Compare every converted tensor exactly with Google's original checkpoint."""
    import numpy as np
    import torch
    for key in ('official', 'converted'):
        assert sha(asset[key]['path']) == asset[key]['sha256']
    converted = torch.load(asset['converted']['path'], map_location='cpu', weights_only=True)
    official = np.load(asset['official']['path'], allow_pickle=False)
    official_keys = [key for key in official.files if 'target' in key]
    groups = [('root',), ('block1',), ('target/embedding',), ('target/head',),
        ('posembed_input',), ('scaleembed_input',), ('encoder_norm',), ('cls',),
        ('encoderblock_', 'norm'), ('encoderblock_', 'mlp'), ('encoderblock_', 'attention')]
    checked = set()
    for terms in groups:
        target_terms = (terms[0].replace('target/', ''),) if len(terms) == 1 else terms
        source_keys = sorted(key for key in official_keys if all(t.lower() in key.lower() for t in terms))
        target_keys = sorted(key for key in converted if all(t.lower() in key.lower() for t in target_terms))
        assert len(source_keys) == len(target_keys)
        for source, target in zip(source_keys, target_keys):
            value = official[source]
            if 'conv' in source:
                value = value.transpose(3,2,0,1)
            elif any(t in source for t in ('key/', 'value/', 'query/')):
                value = value.transpose(1,2,0).reshape(384,384) if 'kernel' in source else value.reshape(-1)
            elif 'out/' in source:
                value = value.transpose(2,0,1).reshape(384,384) if 'kernel' in source else value.reshape(-1)
            elif 'bias' in source or 'scale' in source:
                if value.ndim > 1:
                    value = value.squeeze()
            elif value.ndim == 2:
                value = value.T
            assert np.array_equal(value, converted[target].numpy()), target
            checked.add(target)
    assert checked == set(converted) and len(checked) == len(official_keys)
    model = load_port(asset)(pretrained=False, pretrained_model_path=asset['converted']['path']).eval()
    return model, dict(checked_tensors=len(checked), all_tensors_exact=True,
        conversion_map='pinned upstream scripts/convert_musiq_to_pth.py', strict_architecture_load=True)


def status(run):
    path = Path(run)/'group1/assets/status.json'
    return read(path) if path.exists() else {'musiq':dict(status='pending_asset_preparation')}
