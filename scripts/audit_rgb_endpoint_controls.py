#!/usr/bin/env python3
"""Audit the saved four-backend RGB endpoint experiment against its controls."""
import hashlib,json
from pathlib import Path
import torch
import numpy as np
from PIL import Image


def run():
    root=Path(__file__).resolve().parents[1];rows=[]
    old_native=dict(sd2='20260907-202838-19699005',sana='20260907-210237-19699131',flux='20260907-213330-19699367',pixeldit='20260907-213819-19699380')
    old_endpoint=dict(sd2='20260909-043714-19716621',sana='20260909-041049-19716613',flux='20260909-041112-19716614')
    sources=list((root/'outputs/rgb-endpoint-controls').glob('*/*/comparison.json'))
    assert len(sources)==4, sources
    for source in sources:
        d=json.loads(source.read_text());b=d['config']['model']['pipeline'];folder=source.parent
        state=torch.load(folder/'initial_native.pt',map_location='cpu',weights_only=True)
        assert list(state.shape)==d['initial_native_shape']
        assert hashlib.sha256(state.contiguous().numpy().tobytes()).hexdigest()==d['initial_native_sha256']
        assert all(v for v in d['fairness'].values() if isinstance(v,bool))
        assert len(set(d['model_evaluations'].values()))==1
        a=json.loads((folder/'native_multidiffusion/metadata.json').read_text());c=json.loads((folder/'implied_endpoint_consensus/metadata.json').read_text())
        for key in ('scheduler_timesteps','scheduler_sigmas','pixeldit_flow_schedule','model','native_geometry'):
            assert a[key]==c[key], (b,key)
        for item in (a,c):
            assert len(item['steps'])==d['config']['generation']['num_inference_steps']
            assert all(s['coverage_percent']==100 for s in item['steps'])
            assert [s['scheduler_timestep'] for s in item['steps']]==item['scheduler_timesteps']
        cfg=d['config']['native_multidiffusion'];factor=d['geometry']['factor']
        assert d['geometry']['rgb']['canvas_height']==cfg['canvas_height']*factor
        assert d['geometry']['rgb']['canvas_width']==cfg['canvas_width']*factor
        for n,r in zip(d['geometry']['native']['patches'],d['geometry']['rgb']['patches']):
            for key in ('x','y','size'):assert r[key]==n[key]*factor
        old=root/'outputs/native-controls'/('{}-native_multidiffusion-global_native_canvas-average-uniform'.format(b))/old_native[b]/'result.png'
        row=dict(backend=b,source=str(source.relative_to(root)),initial_hash_matches=True,schedule_geometry_counts_coverage_pass=True,
            native_png_bit_identical_to_prior_baseline=old.read_bytes()==(folder/'native_multidiffusion/result.png').read_bytes())
        prior_rgb=np.asarray(Image.open(old).convert('RGB'),dtype=np.float32)/255
        current_rgb=np.asarray(Image.open(folder/'native_multidiffusion/result.png').convert('RGB'),dtype=np.float32)/255
        assert prior_rgb.shape==current_rgb.shape
        row['historical_a100_vs_fresh_a40_native_display_rgb_mae']=float(np.abs(prior_rgb-current_rgb).mean())
        if b!='pixeldit':
            control=json.loads((folder/'one_patch_roundtrip/control.json').read_text())
            epsilon=torch.load(folder/'one_patch_roundtrip/initial_epsilon.pt',map_location='cpu',weights_only=True)
            assert hashlib.sha256(epsilon.contiguous().numpy().tobytes()).hexdigest()==control['epsilon_sha256']
            assert torch.equal(epsilon,torch.randn(epsilon.shape,generator=torch.Generator().manual_seed(d['config']['experiment']['seed'])))
            assert all(x==len(a['steps']) for x in control['model_evaluations'].values())
            prior=root/'outputs/endpoint-controls'/('{}-trajectory-global_native_canvas-average-uniform-threeway'.format(b))/old_endpoint[b]/'model_implied_endpoint/model_implied_endpoint_final.png'
            row['single_patch_no_roundtrip_png_bit_identical_to_prior_implied']=prior.read_bytes()==(folder/'one_patch_roundtrip/no_roundtrip.png').read_bytes()
            old_rgb=np.asarray(Image.open(prior).convert('RGB'),dtype=np.float32)/255
            new_rgb=np.asarray(Image.open(folder/'one_patch_roundtrip/no_roundtrip.png').convert('RGB'),dtype=np.float32)/255
            row['historical_a100_vs_fresh_a40_no_roundtrip_display_rgb_mae']=float(np.abs(old_rgb-new_rgb).mean())
            row['control_epsilon_seed_and_counts_pass']=True
        rows.append(row)
    out=root/'outputs/rgb-endpoint-controls/report/artifact-audit.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__ == '__main__':
    run()
