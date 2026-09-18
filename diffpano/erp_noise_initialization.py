"""V: one-time nearest ERP-indexed native Gaussian initialization.

Scalar marginals stay Gaussian; duplicate source indices induce covariance.
The independent/shared controls have identical within-view sampling matrices.
No random source field or map is returned to the denoising pipeline.
"""
import hashlib
import math
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn.functional as F
from diffpano.erp_local_consensus import camera_digest, native_digest
from diffpano.projection import erp_to_perspective_grid, _erp_grid_to_padded, spherical_pad_erp


@dataclass(frozen=True)
class ERPNoiseConfig:
    variant: str
    height: int
    width: int
    seed: int
    rule: str = 'H=ceil(pi*max_camera(max(w_native/(2*tan(FOVx/2)),h_native/(2*tan(FOVy/2))))); W=2*H'

    def validate(self):
        if self.variant not in ('S-direct-local', 'V-independent-erp', 'V-shared-erp'):
            raise ValueError('Unknown noise initialization variant')
        if self.height < 1 or self.width != 2*self.height:
            raise ValueError('Noise grid must be positive and 2:1')
        if self.width >= 2**24:
            raise ValueError('Coordinate field requires exactly representable FP32 integers')


def native_cameras(backend, cameras):
    return tuple(replace(c, height=backend.native_spatial_shape_for_rgb(c.height,c.width)[0],
                         width=backend.native_spatial_shape_for_rgb(c.height,c.width)[1]) for c in cameras)


def primary_noise_size(cameras):
    density=max(max(c.width/(2*math.tan(math.radians(c.fov_x)/2)),
                    c.height/(2*math.tan(math.radians(c.fov_y)/2))) for c in cameras)
    h=math.ceil(math.pi*density)
    return h,2*h


class NearestERPIndexProjector:
    """Exact existing CPU FP32 grid_sample nearest convention, arbitrary channels.

    Sample two integer-valued coordinate channels instead of a flattened float
    ID (which would lose integer precision on large grids). This delegates tie,
    wrap and reflected-pole behavior to the historical RGB projector itself.
    Only the coordinate field is retained; it contains no random samples.
    """
    def __init__(self, height, width, vertical_padding_mode='reflect'):
        ERPNoiseConfig('V-shared-erp',height,width,0).validate()
        self.size=(height,width)
        self.vertical_padding_mode=vertical_padding_mode
        y=torch.arange(height,dtype=torch.float32)[:,None].expand(height,width)
        x=torch.arange(width,dtype=torch.float32)[None,:].expand(height,width)
        self.coordinates=spherical_pad_erp(torch.stack((y,x))[None],1,1,vertical_padding_mode)

    def index_map(self,camera):
        h,w=self.size
        grid=erp_to_perspective_grid(camera,h,w,device=torch.device('cpu'))
        padded=_erp_grid_to_padded(grid,h,w)
        locations=F.grid_sample(self.coordinates,padded,mode='nearest',padding_mode='border',align_corners=False)[0]
        if not bool(torch.isfinite(locations).all()) or not torch.equal(locations,locations.round()):
            raise AssertionError('Nonintegral/invalid source coordinates')
        y,x=locations.long()
        if not bool(((y>=0)&(y<h)&(x>=0)&(x<w)).all()):
            raise AssertionError('All rays must select valid source cells')
        return y*w+x


def sample_source(source,indices):
    if source.ndim!=4 or source.dtype!=torch.float32 or source.device.type!='cpu':
        raise ValueError('Source Gaussian must be CPU FP32 [B,C,H,W]')
    if indices.dtype!=torch.int64 or indices.device.type!='cpu':
        raise ValueError('Index map must be CPU int64')
    if int(indices.min())<0 or int(indices.max())>=source.shape[-2]*source.shape[-1]:
        raise ValueError('Source index outside field')
    return source.flatten(2).index_select(2,indices.flatten()).reshape(*source.shape[:2],*indices.shape)


def map_statistics(indices):
    _,counts=torch.unique(indices,return_counts=True)
    unique=counts.numel()/indices.numel()
    return dict(unique_source_fraction=unique,duplicate_source_fraction=1-unique,
                max_source_reuse=int(counts.max()),
                horizontal_collision_rate=float((indices[:,1:]==indices[:,:-1]).float().mean()) if indices.shape[1]>1 else 0.,
                vertical_collision_rate=float((indices[1:]==indices[:-1]).float().mean()) if indices.shape[0]>1 else 0.)


def states_digest(states):
    return hashlib.sha256(''.join(native_digest(s) for s in states).encode()).hexdigest()


@torch.no_grad()
def initialize_erp_noise(backend,cameras,config,batch_size=1):
    config.validate()
    cams=native_cameras(backend,cameras)
    generator=torch.Generator(device='cpu').manual_seed(config.seed)
    states=[];map_hashes=[];source=None;source_draws=0
    projector=None if config.variant=='S-direct-local' else NearestERPIndexProjector(config.height,config.width)
    for camera in cams:  # Stable canonical slots, never denoiser execution order.
        if config.variant=='S-direct-local':
            state=backend.sample_initial_native_state(batch_size=batch_size,native_height=camera.height,
                                                     native_width=camera.width,generator=generator)
        else:
            if config.variant=='V-independent-erp' or source is None:
                source=torch.randn(batch_size,backend.native_channels,config.height,config.width,
                                   generator=generator,device='cpu',dtype=torch.float32)
                source_draws+=1
            indices=projector.index_map(camera)
            map_hashes.append(hashlib.sha256(indices.numpy().tobytes()).hexdigest())
            epsilon=sample_source(source,indices)
            state=backend.initialize_native_state(epsilon)  # Existing backend convention, exactly once.
            del epsilon,indices
            if config.variant=='V-independent-erp':source=None
        expected=(batch_size,backend.native_channels,camera.height,camera.width)
        if tuple(state.shape)!=expected or not bool(torch.isfinite(state).all()):
            raise AssertionError('Invalid native initialized state')
        states.append(state.detach().cpu());del state
    source=None;projector=None  # No source reference leaves this one-time function.
    record=dict(config=asdict(config),native_channels=backend.native_channels,
                native_local_shapes=[[c.height,c.width] for c in cams],dtype='torch.float32',
                scaling=float(backend.native_initial_noise_sigma),scaling_applications=1,
                scaling_helper='NativeStateMixin.initialize_native_state',rng='dedicated torch CPU Generator (MT19937)',
                canonical_camera_sha256=camera_digest(cameras),source_draws=source_draws,
                source_shape=[batch_size,backend.native_channels,config.height,config.width],
                source_field_bytes=batch_size*backend.native_channels*config.height*config.width*4,
                sampling='CPU FP32 grid_sample nearest, align_corners=False, spherical longitude/pole padding; integer row/column map',
                map_sha256=hashlib.sha256(''.join(map_hashes).encode()).hexdigest() if map_hashes else None,
                initial_local_sha256=states_digest(states),first_camera_sha256=native_digest(states[0]),
                source_released=True,persistent_erp_native_state=False,vae_calls=0,
                empirical_normalization=False,clipping=False,fixed_noise_renoising=False)
    return states,record
