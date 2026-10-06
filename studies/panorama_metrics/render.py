"""Fixed world-coordinate perspective renderer; no generated-image correction."""
import functools
import math
import numpy as np
import cv2

HORIZONTAL=tuple((float(y),0.) for y in range(0,360,45))
POLAR=((0.,90.),(0.,-90.))
CUBE=((0.,0.),(90.,0.),(180.,0.),(-90.,0.),(0.,90.),(0.,-90.))
FACE_NAMES=('front','right','back','left','up','down')
PROTOCOL=dict(version='pixel-center-periodic-v1',world_axes='x right, y north, z yaw0; yaw atan2(x,z)',
    horizontal_views=HORIZONTAL,polar_views=POLAR,cubemap_views=CUBE,roll=0,fov_x=90,fov_y=90,
    view_width=512,view_height=512,resampler='OpenCV remap INTER_LINEAR, longitude wrap padding, latitude clamp',
    pixel_centers='ray x=2*(col+0.5)/W-1; y=1-2*(row+0.5)/H; ERP u=(lon/(2pi)+0.5)*W-0.5',
    cs_directional_prompt_lines=dict(horizontal=2,north=0,south=4),indices='zero based')

def rotation(yaw,pitch):
    a,b=math.radians(yaw),math.radians(pitch)
    f=np.array([math.cos(b)*math.sin(a),math.sin(b),math.cos(b)*math.cos(a)])
    r=np.array([math.cos(a),0.,-math.sin(a)]);u=np.cross(f,r)
    return np.stack([r,u,f],axis=1)

def rays(yaw,pitch,size=512):
    x=(np.arange(size)+.5)*2/size-1;y=1-(np.arange(size)+.5)*2/size
    xx,yy=np.meshgrid(x,y);v=np.stack([xx,yy,np.ones_like(xx)],-1)
    v=v@rotation(yaw,pitch).T
    return v/np.linalg.norm(v,axis=-1,keepdims=True)

@functools.lru_cache(maxsize=32)
def maps(h,w,yaw,pitch,size=512):
    v=rays(yaw,pitch,size);lon=np.arctan2(v[...,0],v[...,2]);lat=np.arcsin(v[...,1])
    x=((lon/(2*np.pi)+.5)*w-.5+1).astype(np.float32)
    y=np.clip((.5-lat/np.pi)*h-.5,0,h-1).astype(np.float32)
    return x,y

def render(image,views=HORIZONTAL,size=512):
    assert image.ndim==3 and image.shape[2]==3
    h,w=image.shape[:2];padded=np.concatenate([image[:,-1:],image,image[:,:1]],axis=1)
    return np.stack([cv2.remap(padded,*maps(h,w,y,p,size),interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE) for y,p in views])

def edge_rays(face,edge,n=2,centers=False):
    t=np.linspace(-1,1,n) if not centers else (np.arange(n)+.5)*2/n-1
    if edge=='L':v=np.stack([-np.ones(n),-t,np.ones(n)],-1)
    elif edge=='R':v=np.stack([np.ones(n),-t,np.ones(n)],-1)
    elif edge=='T':v=np.stack([t,np.ones(n),np.ones(n)],-1)
    else:v=np.stack([t,-np.ones(n),np.ones(n)],-1)
    v=v@rotation(*CUBE[face]).T
    return v/np.linalg.norm(v,axis=-1,keepdims=True)

@functools.lru_cache(None)
def edge_pairs():
    groups={}
    for i in range(6):
        for e in 'LRTB':
            endpoints=edge_rays(i,e)
            key=tuple(sorted(tuple(x.round(7)) for x in endpoints))
            groups.setdefault(key,[]).append((i,e))
    assert len(groups)==12 and all(len(v)==2 for v in groups.values())
    pairs=[]
    for a,b in groups.values():
        reverse=not np.allclose(edge_rays(*a),edge_rays(*b))
        assert np.allclose(edge_rays(*a),edge_rays(*b)[::-1] if reverse else edge_rays(*b))
        pairs.append((a,b,reverse))
    return tuple(pairs)

def oriented_edge(image,edge):
    # First column is the boundary; columns progress into the face.
    if edge=='L':return image
    if edge=='R':return image[:,::-1]
    if edge=='T':return image.transpose(1,0,2)
    return image[::-1].transpose(1,0,2)
