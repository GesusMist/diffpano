"""ERP delegation and study-local clean-RGB CEA with explicit polar boundaries."""
import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from diffpano.fusion import RGBFusionAccumulator
from diffpano.projection import ProjectionCache,ERPContribution,perspective_world_rays
from diffpano.geometry import erp_world_directions
from diffpano.warp import StandardWarpOperator

@dataclass(frozen=True)
class CanvasSpec:
    projection:str
    height:int
    width:int
    def __post_init__(self):
        if self.projection not in ('erp','cea') or min(self.height,self.width)<1:raise ValueError('Invalid canvas')

@dataclass
class CanvasResult:
    rgb:torch.Tensor
    contributor_count:torch.Tensor
    weight_sum:torch.Tensor
    pole_rgb:torch.Tensor=None
    pole_weight_sum:torch.Tensor=None

def cea_rays(height,width,device,*,row_start=0,row_stop=None):
    stop=height if row_stop is None else row_stop
    s=1-2*(torch.arange(row_start,stop,device=device,dtype=torch.float32)+.5)/height
    lon=2*math.pi*((torch.arange(width,device=device,dtype=torch.float32)+.5)/width-.5)
    ss,ll=torch.meshgrid(s,lon,indexing='ij');r=(1-ss.square()).clamp_min(0).sqrt()
    return torch.stack((r*ll.sin(),ss,r*ll.cos()),-1)

def cea_uv(directions):
    d=directions.float()
    u=torch.remainder(torch.atan2(d[...,0],d[...,2])/(2*math.pi)+.5,1.)
    v=(1-d[...,1].clamp(-1,1))/2
    return torch.stack((u,v),-1)

def uv_directions(uv):
    lon=2*math.pi*(uv[...,0]-.5);s=1-2*uv[...,1];r=(1-s.square()).clamp_min(0).sqrt()
    return torch.stack((r*lon.sin(),s,r*lon.cos()),-1)

def project_rays(rays,camera):
    d=torch.einsum('ji,...j->...i',camera.rotation(rays.device),rays)
    x,y,z=d.unbind(-1);safe=torch.where(z.abs()>1e-12,z,torch.ones_like(z))
    xx=x/safe/math.tan(math.radians(camera.fov_x)/2)
    yy=y/safe/math.tan(math.radians(camera.fov_y)/2)
    valid=(z>0)&(xx.abs()<=1)&(yy.abs()<=1)
    return torch.stack((xx,-yy),-1).unsqueeze(0),valid.float().unsqueeze(0).unsqueeze(0)

def sample_cea(result,rays,interpolation='nearest'):
    if interpolation not in ('nearest','bilinear'):raise ValueError(interpolation)
    rgb=result.rgb.float();h,w=rgb.shape[-2:];uv=cea_uv(rays)
    gx=2*(uv[...,0]*w+1)/(w+2)-1;gy=2*uv[...,1]-1
    padded=torch.cat((rgb[...,-1:],rgb,rgb[...,:1]),-1)
    grid=torch.stack((gx,gy),-1).unsqueeze(0).expand(rgb.shape[0],-1,-1,-1)
    output=F.grid_sample(padded,grid,mode=interpolation,padding_mode='border',align_corners=False)
    if result.pole_rgb is None:raise ValueError('CEA requires fused pole boundary values')
    # atan2 is stable at exact poles; equivalent to acos(abs(y)) for unit rays.
    theta=torch.atan2(torch.linalg.vector_norm(rays[...,[0,2]],dim=-1),rays[...,1].abs())
    theta0=math.acos(1-1/h);cap=theta<theta0
    if bool(cap.any()):
        north=rays[...,1]>=0
        # Use the same longitude interpolation as the selected interior sampler,
        # ensuring continuity at theta0 for nearest and bilinear conventions.
        edge_y=torch.where(north,torch.full_like(gy,-1+1/h),torch.full_like(gy,1-1/h))
        edge_grid=torch.stack((gx,edge_y),-1).unsqueeze(0).expand(rgb.shape[0],-1,-1,-1)
        edge=F.grid_sample(padded,edge_grid,mode=interpolation,padding_mode='border',align_corners=False)
        pn=result.pole_rgb[:,:,0,:].unsqueeze(-1);ps=result.pole_rgb[:,:,1,:].unsqueeze(-1)
        pole=torch.where(north[None,None],pn,ps)
        a=(theta/theta0)[None,None]
        output=torch.where(cap[None,None],(1-a)*pole+a*edge,output)
    return output

class CanvasOperator:
    def __init__(self,spec,warp_config,fusion_config,device,*,cache=None):
        self.spec=spec;self.device=torch.device(device)
        if warp_config.mode!='standard' or fusion_config.mode!='weighted_average' or fusion_config.weight_mode!='spherediff_center':
            raise ValueError('This study fixes B0C0D1')
        self.cache=cache if cache is not None else ProjectionCache(max_entries=2,cpu_fallback=True)
        self.standard=StandardWarpOperator(warp_config,fusion_config,self.cache)
        self.fusion_config=fusion_config
    def make_accumulator(self,batch_size):
        previous=torch.zeros(batch_size,3,self.spec.height,self.spec.width,device=self.device,dtype=torch.float32)
        result={'main':RGBFusionAccumulator(previous,self.fusion_config)}
        if self.spec.projection=='cea':
            result['pole']=RGBFusionAccumulator(torch.zeros(batch_size,3,2,1,device=self.device),self.fusion_config)
        return result
    def _grid(self,camera,poles=False):
        kind='poles' if poles else 'raster'
        key=('cea','view_to_canvas',kind,str(self.device),'float32',self.spec.height,self.spec.width,'bilinear',*camera.cache_key())
        found=self.cache.get(self.cache.view_to_erp,key)
        if found is not None:return found
        if poles:rays=torch.tensor([[[0.,1.,0.]],[[0.,-1.,0.]]],device=self.device)
        else:
            rk=('cea','rays',str(self.device),'float32',self.spec.height,self.spec.width)
            rays=self.cache.get(self.cache.erp_rays,rk)
            if rays is None:
                rays=cea_rays(self.spec.height,self.spec.width,self.device);self.cache.put(self.cache.erp_rays,rk,rays)
        value=project_rays(rays,camera);self.cache.put(self.cache.view_to_erp,key,value)
        return value
    def _contribution(self,rgb,camera,poles=False):
        grid,mask=self._grid(camera,poles);grid=grid.expand(rgb.shape[0],-1,-1,-1)
        weight=self.standard._weight_map(camera,rgb.device).expand(rgb.shape[0],-1,-1,-1)
        view=F.grid_sample(rgb.float(),grid,mode='bilinear',padding_mode='border',align_corners=False)
        weights=F.grid_sample(weight,grid,mode='bilinear',padding_mode='border',align_corners=False)
        mask=mask.expand(rgb.shape[0],-1,-1,-1)
        return ERPContribution(view*mask,mask,weights*mask)
    def accumulate(self,acc,perspective_rgb,camera):
        if self.spec.projection=='erp':
            acc['main'].accumulate(self.standard.perspective_to_erp(perspective_rgb,camera,(self.spec.height,self.spec.width)))
        else:
            acc['main'].accumulate(self._contribution(perspective_rgb,camera))
            acc['pole'].accumulate(self._contribution(perspective_rgb,camera,True))
    def finalize(self,acc):
        main=acc['main'].finalize()
        if not bool(torch.isfinite(main.erp_rgb).all()) or not bool((main.accumulated_weight>0).all()) or not bool((main.contributor_count>0).all()):
            raise ValueError('Canvas has nonfinite values or uncovered texels; no fallback allowed')
        result=CanvasResult(main.erp_rgb,main.contributor_count,main.accumulated_weight)
        if self.spec.projection=='cea':
            pole=acc['pole'].finalize()
            if not bool((pole.accumulated_weight>0).all()) or not bool(torch.isfinite(pole.erp_rgb).all()):raise ValueError('Uncovered/nonfinite pole')
            result.pole_rgb=pole.erp_rgb;result.pole_weight_sum=pole.accumulated_weight
        return result
    def sample_view(self,result,camera,interpolation=None):
        mode=interpolation or self.standard.warp_config.erp_to_perspective.interpolation
        if self.spec.projection=='erp':
            if interpolation is None:return self.standard.erp_to_perspective(result.rgb,camera)
            from diffpano.projection import erp_to_perspective
            return erp_to_perspective(result.rgb,camera,interpolation=mode,cache=self.cache,
                vertical_padding_mode=self.standard.warp_config.lpw.vertical_padding_mode)
        key=('cea','canvas_to_view',str(self.device),'float32',self.spec.height,self.spec.width,mode,*camera.cache_key())
        rays=self.cache.get(self.cache.erp_to_view,key)
        if rays is None:
            rays=perspective_world_rays(camera,device=self.device);self.cache.put(self.cache.erp_to_view,key,rays)
        return sample_cea(result,rays,mode)
    def export_erp(self,result,height,width):
        if self.spec.projection=='erp':
            if result.rgb.shape[-2:]!=(height,width):raise ValueError('ERP export cannot silently resize generation')
            return result.rgb
        return sample_cea(result,erp_world_directions(height,width,device=self.device),'bilinear')
    def clear_cache(self):
        for name in ('erp_to_view','view_to_erp','erp_rays','lod_maps','host_cache'):getattr(self.cache,name).clear()
