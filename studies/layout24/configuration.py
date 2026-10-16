"""Saved canonical layouts and the established audited FLUX20 configuration."""
from dataclasses import asdict
from studies.layout24.common import *


def resolved(row):
    from studies.all_prompts.audit import configuration
    from studies.camera_patching.cameras import fibonacci_camera_cover,stable_ids,angular_hash,routing
    from studies.camera_count_v2.cameras import restore,build,ids as count_ids
    from studies.camera_count_v2.common import layout_path
    from diffpano.refinement import RefinementConfig
    from diffpano.gradient_fusion import GradientSettings
    assert row['method']=='diffpano' and row['layout'] in LAYOUTS
    info,_=prompt(row['prompt'])
    assert row['prompt_path']==str(prompt_path(row['prompt']))
    c,_,old_cameras,old_ids,old=configuration('flux',row['prompt_path'])
    c.global_pipeline.refinement=RefinementConfig(0.)
    layout=row['layout'];n=row['camera_count'];layout_source=None
    if layout=='old89':cameras=tuple(old_cameras);slots=list(old_ids)
    elif n in (70,50):
        strategy='old' if layout.startswith('old') else 'fibonacci'
        path=layout_path(strategy,n);saved=read(path)
        cameras=restore(saved,c.view);slots=list(saved['camera_ids'])
        assert cameras==build(strategy,n,view=c.view) and slots==count_ids(strategy,n)
        assert saved['coverage']['diffpano']['passed']
        layout_source=dict(path=str(path),sha256=sha(path),angular_sha256=saved['angular_geometry_sha256'])
    else:
        assert layout=='uniform89'
        cameras=fibonacci_camera_cover(c.view,89,phase=0.);slots=stable_ids('fibonacci',89)
    owners=list(range(n));assert len(cameras)==len(slots)==len(set(slots))==n
    settings=GradientSettings(mode=row['gradient_mode'],reference_mode='screened',lambda_color=.1,
        relative_tolerance=1e-5,absolute_tolerance=1e-7,max_iterations=200,gradient_mix_alpha=None)
    assert 'gradient_mix_alpha' not in settings.to_record()
    assert settings.mode in ('rgb','poisson_select')
    if settings.mode=='poisson_select':assert n==89
    assert c.generation.num_inference_steps==20 and c.global_pipeline.refinement.last_fraction==0.
    assert not c.sampling.rotation.enabled
    geometry=dict(layout=layout,camera_count=n,angular_sha256=angular_hash(cameras),poses=[asdict(cam) for cam in cameras],
        owner_ids=owners,noise_slots=slots,slot_owner_mapping=[dict(noise_slot=s,owner_id=i) for i,s in enumerate(slots)],
        route=routing(cameras,info['effective_lines']),FOV_x=80,FOV_y=80,local_RGB_resolution=[1024,1024],
        local_native_resolution=[128,128],ERP_resolution=[2048,4096],
        fibonacci_phase=0. if layout.startswith('uniform') else None,saved_layout=layout_source)
    if layout=='old89':assert geometry['angular_sha256']==old['camera_geometry_sha256']
    return c,cameras,owners,slots,settings,geometry,old


def coverage_evidence(rows):
    """Reuse exact full-resolution geometry gates without enforcing old repo hashes."""
    from studies.camera_count_v2.common import geometry_identity
    records={}
    old_gpu_path=ROOT/'outputs/10.9ournew-flux28/seed0-v1/validation/gpu.json'
    old_gpu=read(old_gpu_path)
    assert old_gpu['passed']
    for row in rows:
        if row['method']!='diffpano' or row['layout'] in records:continue
        _,_,_,_,_,geo,_=resolved(row);layout=row['layout']
        if geo['saved_layout']:
            source=geo['saved_layout'];record=read(source['path'])
            assert record['geometry_identity']==geometry_identity(),'Historical camera implementation changed'
            assert record['angular_geometry_sha256']==geo['angular_sha256']
            assert record['coverage']['geometric']['passed'] and record['coverage']['diffpano']['passed']
            coverage=record['coverage'];p=Path(source['path'])
        else:
            coverage=old_gpu['coverage'][layout]
            assert coverage['geometry_sha256']==geo['angular_sha256']
            assert coverage['all_pixels_covered'] and coverage['all_supported_edges_connected']
            assert coverage['coverage_resolution']==[2048,4096];p=old_gpu_path
        records[layout]=dict(passed=True,source_path=str(p),source_sha256=sha(p),geometry_sha256=geo['angular_sha256'],
            coverage=coverage,method='verified archived full-resolution geometry gate; current generation rechecks support')
    assert set(records)==set(LAYOUTS)
    return records
