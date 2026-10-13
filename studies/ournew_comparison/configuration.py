"""Adapters for full-path prompts and real saved/canonical camera banks."""
from dataclasses import asdict
from studies.ournew_comparison.common import *

def resolved(row):
    from studies.all_prompts.audit import configuration
    from studies.camera_patching.cameras import fibonacci_camera_cover,stable_ids,angular_hash,routing
    from diffpano.refinement import RefinementConfig
    from diffpano.gradient_fusion import GradientSettings
    record,path=prompt(row['prompt'])
    c,_,old_cameras,old_ids,old=configuration('flux',str(path))
    c.global_pipeline.refinement=RefinementConfig(0.)
    if row['layout']=='old89':cameras=tuple(old_cameras);slots=list(old_ids)
    else:
        assert row['layout']=='uniform89'
        cameras=fibonacci_camera_cover(c.view,89,phase=0.0);slots=stable_ids('fibonacci',89)
    owner_ids=list(range(89));assert len(cameras)==len(slots)==len(set(slots))==89
    settings=GradientSettings(mode=row['gradient_mode'],reference_mode=row['reference_mode'] or 'screened',
        relative_tolerance=1e-4 if row['reference_mode']=='coarse_color' else 1e-5,single_pixel_source='fused_rgb_erp')
    assert c.generation.num_inference_steps==20 and c.global_pipeline.refinement.last_fraction==0.
    assert not c.sampling.rotation.enabled
    geometry=dict(layout=row['layout'],angular_sha256=angular_hash(cameras),poses=[asdict(cam) for cam in cameras],
        owner_ids=owner_ids,noise_slots=slots,slot_owner_mapping=[dict(noise_slot=s,owner_id=i) for i,s in enumerate(slots)],
        route=routing(cameras,record['effective_lines']),FOV_x=80,FOV_y=80,local_RGB_resolution=[1024,1024],
        local_native_resolution=[128,128],ERP_resolution=[2048,4096],fibonacci_phase=0.0 if row['layout']=='uniform89' else None)
    return c,cameras,owner_ids,slots,settings,geometry,old


def anchor_location(cameras,owners,device='cpu'):
    import math
    from diffpano.projection import perspective_to_erp_grid,ProjectionCache
    ids=dict(zip(cameras,owners))
    camera=min(cameras,key=lambda c:(-math.cos(c.pitch)*math.cos(c.yaw),ids[c]))
    grid,valid=perspective_to_erp_grid(camera,2048,4096,device=device,cache=ProjectionCache(max_entries=1,cpu_fallback=False))
    distance=grid.square().sum(-1)[0].masked_fill(~valid[0,0].bool(),float('inf'))
    flat=int(distance.argmin());y,x=divmod(flat,4096);gx,gy=grid[0,y,x].tolist()
    return dict(location_selection_camera_id=ids[camera],erp_y=y,erp_x=x,
        source_x=(gx+1)*camera.width/2-.5,source_y=(gy+1)*camera.height/2-.5,
        policy='existing_patch_center_erp_mapping')
