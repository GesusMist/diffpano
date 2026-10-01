import math
from dataclasses import replace,asdict
from collections import Counter
from diffpano.camera import PerspectiveCamera,camera_for_direction
from diffpano.bridge_factorial import angular_geometry,digest
from diffpano.conditioning import camera_prompt_indices,expand_directional_prompts
from studies.tt_cea.common import ROOT,read,immutable

def fibonacci_centers():
    golden_ratio=(1+math.sqrt(5))/2
    return [dict(yaw=2*math.pi*((j/golden_ratio+.5)%1-.5),pitch=math.asin(1-2*(j+.5)/89),
        roll=0.,fov_x=80.,fov_y=80.) for j in range(89)]

def serialize_covers(old):
    records={}
    for name,poses,ids in (('old89',angular_geometry(old),list(range(89))),
                         ('ea89',fibonacci_centers(),['ea89:%03d'%i for i in range(89)])):
        records[name]=dict(name=name,cameras=poses,camera_geometry_sha256=digest(poses),innovation_ids=ids,
            construction='saved canonical baseline cover' if name=='old89' else 'approximately equal-area-distributed Fibonacci centers; not an exact footprint partition')
        immutable(ROOT/'geometry'/(name+'.json'),records[name])
    return records

def load_cover(name,height,width):
    data=read(ROOT/'geometry'/(name+'.json'))
    assert digest(data['cameras'])==data['camera_geometry_sha256']
    cams=tuple(PerspectiveCamera(**v,height=height,width=width) for v in data['cameras'])
    assert len(cams)==89 and all(c.fov_x==c.fov_y==80 for c in cams)
    return cams,data['innovation_ids']

def routing(cams,lines):
    bank=expand_directional_prompts(lines)
    slots=camera_prompt_indices(cams,bank.directions).tolist()
    return dict(indices=slots,prompt_band_counts=dict(Counter(str(i//4) for i in slots)))

def report_cameras():
    return tuple(camera_for_direction(y,p,height=1024,width=1024,fov_x=80,fov_y=80) for p,y in
        ((0,0),(0,90),(0,180),(0,-90),(60,0),(60,180),(-60,0),(-60,180),(90,0),(-90,0)))
