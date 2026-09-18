"""Shared provenance and model-free native geometry resolution for V."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from diffpano.pipelines.flux import FluxViewDenoiser
from diffpano.pipelines.pixeldit import PixelDiTViewDenoiser
from scripts.consensus_dense_controls import make_config

ROOT=Path('outputs/vae-residual-controls/20260917-noise-v')
S_ROOT=Path('outputs/vae-residual-controls/20260916-diagnostics-pt/S')
CACHE=Path('/scratch/user/shig/diffpano/hf_cache/hub')


def read(path):return json.loads(Path(path).read_text())


def source_hashes():
    paths=list(Path('diffpano').rglob('*.py'))+list(Path('scripts').glob('*.py'))+list(Path('tests').glob('*.py'))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def require_validation():
    value=read(ROOT/'execution.json')['validation']
    if not value['passed'] or value['source_hashes']!=source_hashes():
        raise AssertionError('V scientific source differs from the passed full validation gate')


def s_config(name):
    c=make_config('S',name);m=read(S_ROOT/name/'metadata.json')
    if c.to_dict()!=m['config']:raise AssertionError('Generation config differs from completed S')
    return c,m


def shape_backend(name):
    """Read configuration only; query the real adapter's native shape properties."""
    _,m=s_config(name)
    if name=='pixeldit':
        b=object.__new__(PixelDiTViewDenoiser)
        b.model=SimpleNamespace(parameters=lambda: iter([torch.zeros(1)])) # device/dtype only; no RNG draw.
        return b,dict(source=m['model_checkpoint'],revision=m['model_revision'],shape_api='PixelDiTViewDenoiser properties')
    revision=m['current_cached_revision'];folder=CACHE/'models--ModelsLab--flux.1-dev'/'snapshots'/revision
    t=read(folder/'transformer/config.json');v=read(folder/'vae/config.json')
    pipeline=SimpleNamespace(_execution_device='cpu',transformer=SimpleNamespace(config=SimpleNamespace(**t)),
                             vae_scale_factor=2**(len(v['block_out_channels'])-1))
    b=FluxViewDenoiser(pipeline,guidance_scale=m['config']['generation']['guidance_scale'])
    return b,dict(source=m['model_checkpoint'],revision=revision,historical_model_revision=m['model_revision'],
                  shape_api='FluxViewDenoiser properties from cached transformer/VAE configuration',
                  config_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/'transformer/config.json',folder/'vae/config.json']})
