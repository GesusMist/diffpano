"""Immutable ten-baseline audit and 60-row scientific plan."""
from dataclasses import asdict
import struct
from pathlib import Path
import torch
from diffpano.bridge_factorial import angular_geometry,digest as geometry_digest
from diffpano.erp_local_consensus import camera_digest
from studies.tt_cea.common import *
from studies.tt_cea.cameras import serialize_covers,routing
from studies.tt_cea.schedule import eligible_indices


def baseline_coefficients(m):
    if m['backend']=='sd2':
        rows=m['control_audit']['ddim_coefficients']['intervals']
        return [dict(k=i,model_timestep=r['t'],alpha_high=r['alpha'],sigma_high=r['sigma'],
            alpha_low=r['next_alpha'],sigma_low=r['next_sigma']) for i,r in enumerate(rows)]
    s=torch.tensor(m['prepared_schedule']['sigmas'],dtype=torch.float32)
    return [dict(k=i,model_timestep=t,alpha_high=float(1-s[i]),sigma_high=float(s[i]),
        alpha_low=float(1-s[i+1]),sigma_low=float(s[i+1])) for i,t in enumerate(m['prepared_schedule']['timesteps'])]

def build_audit():
    from studies.gwtf_erp4k import common as fourk
    from studies.gwtf_underwater import common as underwater
    fourk.require_validation();underwater.require_validation()
    historical={};baselines=[];rows=[];covers=None
    for prompt in PROMPTS:
        for name in BACKENDS:
            c,stub,cams,m=base_config(name,prompt)
            base=BASES[prompt]/name
            paths={f:base/f for f in ('gwtf-final.png','metadata.json','gpu-preflight.json')}
            for p in paths.values():
                if not p.is_file():raise FileNotFoundError(p)
                historical[str(p.relative_to(REPO))]=sha(p)
            p=paths['gwtf-final.png']
            with p.open('rb') as f:header=f.read(24)
            assert header[:8]==b'\x89PNG\r\n\x1a\n'
            wh=struct.unpack('>II',header[16:24]);assert wh==(4096,2048)
            assert m['config']['experiment']['seed']==0 and m['config']['generation']['batch_size']==1
            assert m['config']['erp']['height']==2048 and m['config']['erp']['width']==4096
            assert m['config']['generation']['num_inference_steps']==STEPS[name]
            assert m['method']=='gwtf' and m['initialization']['config']['seed']==0
            assert read(paths['gpu-preflight.json'])['passed']
            assert camera_digest(cams)==m['camera_sha256']
            assert geometry_digest(angular_geometry(cams))==m['camera_geometry_sha256']
            assert len(cams)==89 and all(v.fov_x==v.fov_y==80 for v in cams)
            assert m['actual_transformer_forward_invocations']==89*STEPS[name]
            assert m['vae_calls']==dict(encode=0 if name=='pixeldit' else 178*STEPS[name],decode=0 if name=='pixeldit' else 89*(STEPS[name]+1))
            prompt_path=REPO/c.prompt.path;historical[c.prompt.path]=sha(prompt_path)
            if prompt=='underwater':assert sha(prompt_path)==m['prompt_sha256']
            else:assert sha(prompt_path)==read(fourk.ROOT/'validation.json')['source_hashes'][c.prompt.path]
            for path,h in m['source_hashes'].items():
                if sha(REPO/path)!=h:raise AssertionError('Historical source mismatch: '+path)
                historical[path]=h
            if covers is None:
                # Serialize only during the audit; jobs only load these lists.
                covers=serialize_covers(cams)
            assert covers['old89']['cameras']==angular_geometry(cams)
            route=routing(cams,prompt_path.read_text().splitlines());assert route['indices']==m['prompt_indices']
            coeff=baseline_coefficients(m);assert len(coeff)==STEPS[name]
            baselines.append(dict(backend=name,prompt=prompt,image_relative=str(p.relative_to(REPO)),image_absolute=str(p),
                image_sha256=sha(p),metadata_sha256=sha(paths['metadata.json']),width=wh[0],height=wh[1],
                metadata_path=str(paths['metadata.json']),seed=0,prompt_text=prompt_path.read_text(),prompt_sha256=sha(prompt_path),
                config=c.to_dict(),checkpoint=m['model_checkpoint'],revision=m['model_revision'],
                prepared_schedule=m['prepared_schedule'],interval_coefficients=coeff,cameras=[asdict(v) for v in cams],
                camera_geometry_sha256=m['camera_geometry_sha256'],camera_sha256=m['camera_sha256'],
                local_RGB_resolution=m['local_RGB_resolution'],local_native_resolution=m['local_native_resolution'],
                native_channels=m['native_channels'],initialization=m['initialization'],routing=route,
                bridge=m['bridge_mode'],transition=m['audit']['transition'],terminal_semantics='terminal local-state decode/fusion'))
            for case in CASES:
                _,_,_,_,row=resolved(name,prompt,case)
                row.update(baseline_image=str(p),baseline_metadata=str(paths['metadata.json']),
                    baseline_image_sha256=sha(p),reuse_historical=case=='R0',
                    output=str(p) if case=='R0' else str(sample_dir(name,prompt,case)/'final.png'))
                rows.append(row)
    # Preserve every prior gate/manifest plus all tracked sources/configuration.
    import subprocess
    for file in subprocess.check_output(['git','ls-files'],text=True).splitlines():
        p=REPO/file
        if p.is_file() and not file.startswith('studies/tt_cea/'):historical[file]=sha(p)
    for root in BASES.values():
        for p in root.rglob('*.json'):historical[str(p.relative_to(REPO))]=sha(p)
    return dict(passed=True,baselines=baselines),dict(hashes=historical),rows,covers

def main(config):
    settings=load_settings(config)
    # Existing partially audited roots are never silently accepted as a study.
    if ROOT.exists() and not (ROOT/'manifest.json').exists() and any(ROOT.iterdir()):
        allowed={'geometry'}
        if {p.name for p in ROOT.iterdir()}-allowed:raise RuntimeError('Output root exists without manifest')
    audit,preserved,rows,covers=build_audit()
    plan=dict(study='tt-cea-seed0-v1',settings=settings,settings_sha256=sha(config),
        starting_checkout=read(REPO/'studies/tt_cea/start_inventory.json'),rows=rows,
        baseline_audit_sha256=digest(audit),preservation_sha256=digest(preserved),covers=covers,
        baseline_count=10,planned_rows=60,new_production_generations=50,
        comparisons={'time_travel':['R0:T1','T1:TB'],'projection':['R0:C0','R0:P0','P0:C1','R0:C1']})
    immutable(ROOT/'manifest.json',plan);immutable(ROOT/'baseline_audit.json',audit,allow_nan=True);immutable(ROOT/'preservation.json',preserved)
    print('AUDIT PASSED: ten 4096x2048 GWTFlow baselines; sixty rows, fifty new generations',flush=True)

if __name__=='__main__':
    p=parser(__doc__);main(p.parse_args().config)
