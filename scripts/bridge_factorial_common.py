"""Shared immutable settings, manifests and validation gate for ruins factorial."""
import hashlib
import json
import os
from pathlib import Path
from dataclasses import asdict
from collections import Counter

from diffpano.bridge_factorial import BACKENDS, PILOT, GEOMETRY, PROMPT_SHA, factor_cells, factors, make_config, common_settings, digest, saved_cameras, angular_geometry
from diffpano.conditioning import expand_directional_prompts, camera_prompt_indices

ROOT=Path('outputs/bridge-factorial-ruins/20260918')
SPHERE=Path('/home/shig/diffpano_reference_sources/spherediff-2c8c68b')
SPHERE_COMMIT='2c8c68ba088f2803b3dce4b52b7b0d68bc996139'


def read(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def source_hashes():
    paths=list(Path('diffpano').rglob('*.py'))+list(Path('scripts').glob('*.py'))+list(Path('tests').glob('*.py'))
    return {str(p):sha(p) for p in sorted(paths)}


def verify_prompt():
    p=Path('prompts/ruins.txt');q=SPHERE/'data/prompts/ruins.txt'
    if p.read_bytes()!=q.read_bytes() or sha(p)!=PROMPT_SHA or len(p.read_bytes().splitlines())!=5:
        raise AssertionError('Exact five-line SphereDiff ruins prompt required')
    return dict(path=str(p),sha256=sha(p),original_sha256=sha(q),physical_lines=5,lines=p.read_text().splitlines(),byte_identical=True)


def verify_sources():
    x=read(ROOT/'execution.json')
    for p,h in x['spherediff_source_hashes'].items():
        if sha(SPHERE/p)!=h:raise AssertionError('Original SphereDiff source changed: '+p)
    return x['spherediff_source_hashes']


def reference_specs():
    return {
      'flux':dict(pipeline='SphericalFluxPipeline',model_source='black-forest-labs/FLUX.1-dev',revision='3de623fc3c33e44ffbe2bad470d0f45bccf2eb21',variant=None,mixed_precision='bf16',enable_model_cpu_offload=False,enable_vae_tiling=False,
                  call=dict(num_inference_steps=28,guidance_scale=3.5,true_cfg_scale=1.,n_spherical_points=26500,weighted_average_temperature=.1,erp_height=2048,erp_width=4096)),
      'sana':dict(pipeline='SphericalSanaPipeline',model_source='Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers',revision='e2b3c0cbffebcd09d83805e88b9f5f106afc74ac',variant='bf16',mixed_precision='bf16',enable_model_cpu_offload=False,enable_vae_tiling=False,
                  call=dict(num_inference_steps=20,guidance_scale=4.5,height=1024,width=1024,n_spherical_points=2600,weighted_average_temperature=.1,erp_height=2048,erp_width=4096))}


def make_manifest():
    prompt=verify_prompt();verify_sources();rows=[];models={};angular_hashes=set()
    bank=expand_directional_prompts(prompt['lines'])
    for name in BACKENDS:
        c=make_config(name,'A0B0C0D0');cams=saved_cameras(c.view,(c.erp.height,c.erp.width));angles=angular_geometry(cams);ah=digest(angles);angular_hashes.add(ah)
        slots=camera_prompt_indices(cams,bank.directions).tolist()
        models[name]=dict(common_settings=common_settings(c),common_settings_sha256=digest(common_settings(c)),camera_geometry_sha256=ah,
                          camera_count=len(cams),cameras=[asdict(v) for v in cams],directional_prompt_indices=slots,semantic_band_counts=dict(Counter(i//4 for i in slots)),
                          local_RGB_resolution=[c.view.height,c.view.width],clean_ERP_resolution=[c.erp.height,c.erp.width],expected_guided_predictions=len(cams)*c.generation.num_inference_steps)
        for cell in factor_cells():
            conf=make_config(name,cell);f=factors(cell)
            if common_settings(conf)!=models[name]['common_settings']:raise AssertionError('Non-factor setting changed')
            rows.append(dict(backend=name,cell=cell,**f,pilot=cell in PILOT,prompt_sha256=prompt['sha256'],camera_geometry_sha256=ah,
                initialization_mode='independent_erp' if f['A']==0 else 'shared_erp',bridge_mode='not_applicable_identity' if name=='pixeldit' else 'local_identity_preserving',
                warp_mode=conf.warp.mode,reducer_mode=conf.fusion.mode,spatial_weight_mode=conf.fusion.weight_mode,transition_mode='preserve_current_state',
                resolved_config=conf.to_dict(),resolved_config_sha256=digest(conf.to_dict()),output=str(ROOT/'cells'/name/cell)))
    if len(rows)!=80 or len({(r['backend'],r['cell']) for r in rows})!=80 or len(angular_hashes)!=1:raise AssertionError('Invalid 80-cell geometry/matrix')
    x=dict(study='Bridge + Directional Prompting 2^4 Factorial Study — Ruins',seed=0,prompt=prompt,geometry_source=GEOMETRY,
           camera_geometry_sha256=next(iter(angular_hashes)),angular_cameras=angles,models=models,cells=rows,expected_cells=80,pilot_cells=20,
           references=reference_specs(),spherediff_source_commit=SPHERE_COMMIT,expected_scientific_generations=82,
           factor_contrast_convention='mean(metric | product of +/-1 factor signs = +1) minus mean(metric | product = -1); descriptive one-seed contrasts, no p-values')
    write_new(ROOT/'manifest.json',x)
    return x


def require_validation():
    gate=read(ROOT/'validation.json')
    if not gate['passed'] or gate['source_hashes']!=source_hashes():raise AssertionError('Scientific sources differ from passed full-suite gate')
    if gate['manifest_sha256']!=sha(ROOT/'manifest.json'):raise AssertionError('Manifest changed')
    verify_prompt();verify_sources()
    return read(ROOT/'manifest.json')


def cell_record(name,cell):
    manifest=require_validation();record=next(r for r in manifest['cells'] if (r['backend'],r['cell'])==(name,cell))
    c=make_config(name,cell)
    if c.to_dict()!=record['resolved_config']:raise AssertionError('Resolved scientific settings changed')
    return manifest,record,c


def make_provenance():
    """CPU allocation: preserve all historical output hashes; inspect cached model identities."""
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Compute allocation required for historical hashing')
    historical={}
    for p in sorted(Path('outputs').rglob('*')):
        if p.is_file() and ROOT not in p.parents:
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            historical[str(p)]=dict(sha256=h.hexdigest(),bytes=p.stat().st_size)
    caches={}
    locations={'diffpano_flux':(Path('/scratch/user/shig/diffpano/hf_cache/hub/models--ModelsLab--flux.1-dev/snapshots'),'fa45a9eb6808ba8fdfc7cc2756f7f1a16e0921f4'),
               'spherediff_flux':(Path('/scratch/user/shig/SphereDiff/hf_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots'),'3de623fc3c33e44ffbe2bad470d0f45bccf2eb21')}
    for name,(root,rev) in locations.items():
        folder=root/rev;configs={};weights={}
        for p in sorted(folder.rglob('*')):
            if not p.is_file():continue
            rel=str(p.relative_to(folder))
            if p.suffix=='.json' and (p.name=='config.json' or p.name=='model_index.json' or p.name=='scheduler_config.json'):
                configs[rel]=dict(sha256=sha(p),content=read(p))
            if p.suffix in ('.safetensors','.bin'):
                target=p.resolve().name
                weights[rel]=dict(bytes=p.stat().st_size,cached_content_address=target,
                    address_is_sha256=len(target)==64 and all(c in '0123456789abcdef' for c in target))
        caches[name]=dict(revision=rev,configs=configs,weights=weights)
    left=caches['diffpano_flux'];right=caches['spherediff_flux']
    keys=set(left['configs'])|set(right['configs'])
    equal_configs=[k for k in sorted(keys) if left['configs'].get(k)==right['configs'].get(k)]
    weight_ids=lambda x:sorted((v['bytes'],v['cached_content_address']) for v in x['weights'].values())
    record=dict(historical_artifacts=historical,flux_sources=caches,equal_component_configs=equal_configs,
        differing_component_configs=sorted(keys-set(equal_configs)),all_cached_weight_addresses_match=weight_ids(left)==weight_ids(right),
        independent_full_weight_rehash=False,
        conclusion='Cached content addresses and component configs are reported; full weight bytes were not independently rehashed, so do not claim checkpoint equivalence solely from different model IDs.')
    write_new(ROOT/'provenance.json',record)
    print('Historical artifact hashes recorded:',len(historical),flush=True)
