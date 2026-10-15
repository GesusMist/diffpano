"""Saved Old89 and the existing path-aware baseline configuration, with one new guidance field."""
from dataclasses import asdict
from studies.mean_max_old89.common import *

def resolved(row):
    from studies.all_prompts.audit import configuration
    from studies.camera_patching.cameras import angular_hash,routing
    from diffpano.gradient_fusion import GradientSettings
    from diffpano.refinement import RefinementConfig
    info,_=prompt(row['prompt'])
    assert row['prompt_path']==str(prompt_path(row['prompt']))
    c,_,cameras,slots,old=configuration('flux',row['prompt_path'])
    c.global_pipeline.refinement=RefinementConfig(0.)
    cameras=tuple(cameras);owners=list(range(89));slots=list(slots)
    assert slots==owners and len(cameras)==89
    settings=GradientSettings(mode=MODE,gradient_mix_alpha=row['gradient_mix_alpha'],reference_mode=row['reference_mode'],
        lambda_color=.1,coarse_eta=.1,coarse_rows=16,coarse_cols=32,relative_tolerance=1e-4 if row['reference_mode']=='coarse_color' else 1e-5,
        absolute_tolerance=1e-7,max_iterations=200,constrained_max_iterations=400)
    assert c.generation.num_inference_steps==20 and c.global_pipeline.refinement.last_fraction==0 and not c.sampling.rotation.enabled
    geometry=dict(layout='old89',angular_sha256=angular_hash(cameras),poses=[asdict(cam) for cam in cameras],
        noise_slots=slots,owner_ids=owners,slot_owner_mapping=[dict(noise_slot=s,owner_id=o) for s,o in zip(slots,owners)],
        route=routing(cameras,info['effective_lines']),FOV_x=80,FOV_y=80,local_RGB_resolution=[1024,1024],
        local_native_resolution=[128,128],ERP_resolution=[2048,4096])
    assert geometry['angular_sha256']==old['camera_geometry_sha256']
    return c,cameras,owners,slots,settings,geometry,old
