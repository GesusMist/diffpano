"""Frozen settings, matched-control audit and source gates for GWTFlow study."""
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from diffpano.bridge_factorial import BACKENDS, make_config, saved_cameras, angular_geometry, digest
from diffpano.erp_noise_initialization import native_cameras, primary_noise_size
from diffpano.erp_local_consensus import camera_digest
from diffpano.pipelines.native_state import NativeStateMixin

ROOT=Path('outputs/gwtf-noise-comparison/20260922')
BASE=Path('outputs/bridge-factorial-ruins/20260918')
CELL='A1B0C0D1'
CACHE=Path('/scratch/user/shig/diffpano/hf_cache/hub')


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def source_hashes():
    paths=[]
    for folder in ('diffpano','scripts','tests','studies/gwtf_noise'):
        paths += [p for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.md','.txt','.json')]
    paths += list(Path('slurm').glob('gwtf_noise*.slurm'))
    paths += [Path('prompts/ruins.txt')]
    return {str(p):sha(p) for p in sorted(paths)}


class GeometryBackend(NativeStateMixin):
    device=torch.device('cpu')
    def __init__(self,channels,factor,scale=1.):
        self.native_channels=channels;self.native_spatial_factor=factor;self.native_initial_noise_sigma=scale
    def encode_clean(self,*a):raise AssertionError('No model/VAE in noise preflight')
    def decode_clean(self,*a):raise AssertionError('No model/VAE in noise preflight')
    def predict_clean_and_endpoint(self,*a):raise AssertionError('No denoiser in noise preflight')


def backend_geometry(name):
    m=read(BASE/'cells'/name/CELL/'metadata.json');c=make_config(name,CELL)
    cams=saved_cameras(c.view,(c.erp.height,c.erp.width))
    factor=c.view.height//m['local_native_resolution'][0]
    b=GeometryBackend(m['native_channels'],factor,m['native_scale'])
    assert list(b.native_spatial_shape_for_rgb(c.view.height,c.view.width))==m['local_native_resolution']
    assert list(primary_noise_size(native_cameras(b,cams)))==m['noise_grid']
    evidence=dict(source='actual completed factorial backend preflight + pinned cached model configuration',
                  historical_preflight=str(BASE/'preflight'/(name+'.json')))
    if name!='pixeldit':
        snapshot=CACHE/('models--'+c.model.id.replace('/','--'))/'snapshots'/c.model.revision
        tpath=snapshot/('unet' if name=='sd2' else 'transformer')/'config.json'
        t=read(tpath);vpath=snapshot/'vae/config.json';v=read(vpath)
        assert t['in_channels']//(4 if name=='flux' else 1)==b.native_channels
        if 'block_out_channels' in v:assert 2**(len(v['block_out_channels'])-1)==factor
        elif 'encoder_block_out_channels' in v:assert 2**(len(v['encoder_block_out_channels'])-1)==factor
        evidence['config_hashes']={str(p):sha(p) for p in (tpath,vpath)}
    else:
        assert (b.native_channels,factor)==(3,1)
        evidence['config_hashes']={c.pixeldit.config_path:sha(c.pixeldit.config_path)}
    return c,b,cams,m,evidence


def audit_controls():
    old=read(BASE/'manifest.json');gate=read(BASE/'validation.json')
    # New files do not invalidate the historical manifest; all old bytes remain.
    for p,h in gate['source_hashes'].items():
        if sha(p)!=h:raise AssertionError('Historical source changed: '+p)
    models={};preserved={}
    for name in BACKENDS:
        c,b,cams,m,evidence=backend_geometry(name)
        row=next(r for r in old['cells'] if (r['backend'],r['cell'])==(name,CELL))
        assert c.to_dict()==row['resolved_config']==m['config']
        assert m['prompt']==old['prompt'] and sha(c.prompt.path)==m['prompt']['sha256']
        assert camera_digest(cams)==m['camera_sha256']
        assert digest(angular_geometry(cams))==m['camera_geometry_sha256']==old['camera_geometry_sha256']
        assert m['camera_count']==89 and all(v.fov_x==v.fov_y==80 for v in cams)
        assert m['prompt_indices']==old['models'][name]['directional_prompt_indices']
        assert (m['warp_mode'],m['reducer_mode'],m['spatial_weight_mode'])==('standard','weighted_average','spherediff_center')
        assert m['audit']['transition']=='preserve_current_state' and not m['audit']['fixed_noise_renoising']
        assert m['initialization']['config']['variant']=='V-shared-erp'
        controls={}
        for cell in (CELL,'A0B0C0D1'):
            folder=BASE/'cells'/name/cell;v=read(folder/'metadata.json')
            expected=make_config(name,cell).to_dict()
            assert expected==v['config']
            for key in ('model_checkpoint','model_revision','prepared_schedule','conditioning_sha256','prompt_indices',
                        'camera_sha256','native_channels','local_native_resolution','local_RGB_resolution','noise_grid','native_scale','bridge_mode','vae_dtype'):
                assert v[key]==m[key],(name,cell,key)
            for p in (folder/'metadata.json',folder/'final_result.png'):preserved[str(p)]=sha(p)
            controls[cell]=dict(metadata=str(folder/'metadata.json'),image=str(folder/'final_result.png'),
                metadata_sha256=sha(folder/'metadata.json'),image_sha256=sha(folder/'final_result.png'),
                static_scientific_match=True,runtime_match_required=True)
        models[name]=dict(config=c.to_dict(),cameras=[asdict(v) for v in cams],camera_sha256=m['camera_sha256'],
            angular_camera_sha256=m['camera_geometry_sha256'],native_channels=b.native_channels,native_factor=b.native_spatial_factor,
            local_native_resolution=m['local_native_resolution'],noise_grid=m['noise_grid'],scale=b.native_initial_noise_sigma,
            expected_predictions=89*c.generation.num_inference_steps,controls=controls,shape_provenance=evidence,
            audit_fields=['full resolved config incl model revision/guidance/negative prompts/VAE/interpolation/center temperature',
                'exact prompt SHA/routing','exact ordered camera/raster/angular hashes','prepared scheduler/timesteps/sigmas',
                'native shape/scale','local identity bridge/current-state transition','terminal decoded-local-state assembly'],
            sd2_coefficients_policy='Reconstruct historical DDIM from saved full scheduler config and compare actual alpha/sigma intervals on GPU; original metadata lacks full alpha array' if name=='sd2' else None)
    write(ROOT/'manifest.json',dict(study='GWTFlow-style vs nearest shared ERP, B0C0D1',seed=0,models=models,
        historical_artifact_hashes=preserved,historical_source_hashes=gate['source_hashes'],
        baseline_manifest_sha256=sha(BASE/'manifest.json'),initialization_only=True,all_five_required=True,
        control_policy='Reuse only after static and actual-backend runtime audit; no historical regeneration when matched.'))


def require_validation(statistical=False):
    v=read(ROOT/'validation.json')
    assert v['passed'] and v['source_hashes']==source_hashes(),'Scientific sources changed after validation'
    assert sha(ROOT/'manifest.json')==v['manifest_sha256']
    if statistical:
        p=read(ROOT/'preflight.json');assert p['passed'] and p['source_hashes']==source_hashes()
    return read(ROOT/'manifest.json')


def verify_preservation():
    m=read(ROOT/'manifest.json')
    for p,h in m['historical_artifact_hashes'].items():assert sha(p)==h,'Historical output changed: '+p
    for p,h in m['historical_source_hashes'].items():assert sha(p)==h,'Historical code changed: '+p
