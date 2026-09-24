"""GWTFlow-style one-hop ERP-to-native Gaussian transport (paper Algorithm 1).

Burgert et al., arXiv:2501.08331v4, Algorithm 1 and Proposition 1.
This is a spherical adaptation, not the released temporal video implementation.
Historical S/V initializers are untouched. See studies/gwtf_noise/METHOD.md.
"""
import hashlib
import json
import math
import resource
import time
from dataclasses import asdict, dataclass

import torch
from diffpano.erp_local_consensus import camera_digest
from diffpano.erp_noise_initialization import (ERPNoiseConfig, NearestERPIndexProjector,
    native_cameras, states_digest)
from diffpano.projection import perspective_to_erp_grid


@dataclass(frozen=True)
class GWTFlowNoiseConfig:
    height: int
    width: int
    seed: int
    channel_chunk: int = 4

    def validate(self):
        ERPNoiseConfig('V-shared-erp', self.height, self.width, self.seed).validate()
        if self.channel_chunk < 1:
            raise ValueError('Positive channel chunk required')


def innovation_seed(seed, slot):
    key = json.dumps(['gwtflow-paper-algorithm1-v1', int(seed), slot], sort_keys=True)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') % (2**63-1)


@dataclass
class TransportGraph:
    """CPU geometry only; all outgoing edges of every included source are present.

    target_index=-1 denotes an edge needed for a selected source's conditional
    mean but whose target is not requested by a statistical subset. Full graphs
    have no such edges. source_ids index the global flattened ERP field.
    """
    source_size: int
    target_shape: tuple
    source_ids: torch.Tensor
    source_index: torch.Tensor
    target_index: torch.Tensor
    degree: torch.Tensor
    density: torch.Tensor
    normalizer: torch.Tensor
    target_density: torch.Tensor
    forward_edges: int = 0
    backward_edges: int = 0

    @property
    def target_size(self):
        return math.prod(self.target_shape)

    @property
    def bytes(self):
        return sum(v.numel()*v.element_size() for v in vars(self).values() if isinstance(v, torch.Tensor))

    def summary(self):
        return dict(edges=self.source_index.numel(), used_sources=self.source_ids.numel(),
            forward_edges=self.forward_edges, backward_fill_edges=self.backward_edges,
            expansion_sources=int((self.degree>1).sum()),
            max_source_degree=float(self.degree.max()) if self.degree.numel() else 0.,
            contraction_targets=int((torch.bincount(self.target_index[self.target_index>=0], minlength=self.target_size)>1).sum()),
            unconnected_targets=int((self.normalizer==0).sum()), geometry_bytes=self.bytes,
            density_min=float(self.target_density.min()), density_max=float(self.target_density.max()))

    def select(self, target_ids):
        """Exact marginal subgraph, including every sibling innovation edge."""
        ids = torch.unique(target_ids.flatten().long(), sorted=True)
        if ids.numel()==0 or int(ids.min())<0 or int(ids.max())>=self.target_size:
            raise ValueError('Invalid target subset')
        hit = torch.isin(self.target_index, ids)
        used = torch.unique(self.source_index[hit], sorted=True)
        edges = torch.isin(self.source_index, used)
        old_targets = self.target_index[edges]
        pos = torch.searchsorted(ids, old_targets).clamp_max(ids.numel()-1)
        new_targets = torch.where(ids[pos]==old_targets, pos, -torch.ones_like(pos))
        return TransportGraph(self.source_size, (ids.numel(),), self.source_ids[used],
            torch.searchsorted(used,self.source_index[edges]), new_targets,
            self.degree[used], self.density[used], self.normalizer[ids], self.target_density[ids]), ids


def graph_from_maps(forward, backward, target_shape, density=None):
    """Algorithm 1: forward edges first; backward edges only for empty targets.

    forward[s] is target index or -1; backward[t] is source index or -1.
    Unlike a nearest backward-only warp, this handles contraction and expansion
    together. No random value participates in graph construction.
    """
    f, b = forward.reshape(-1), backward.reshape(-1)
    if f.dtype!=torch.int64 or b.dtype!=torch.int64 or f.device.type!='cpu' or b.device.type!='cpu':
        raise ValueError('Maps must be CPU int64')
    ns, nt = f.numel(), math.prod(target_shape)
    if ns<1 or b.numel()!=nt or nt<1 or bool(((f < -1)|(f>=nt)).any()) or bool(((b < -1)|(b>=ns)).any()):
        raise ValueError('Invalid forward/backward maps')
    src = torch.nonzero(f>=0).flatten(); dst = f[src]
    nf = src.numel()
    indegree = torch.bincount(dst,minlength=nt)
    missing = torch.nonzero((indegree==0)&(b>=0)).flatten()
    src = torch.cat((src,b[missing])); dst = torch.cat((dst,missing))
    # Canonical source-then-target edge order, independent of processing order.
    order = torch.argsort(src*nt+dst,stable=True)
    src, dst = src[order], dst[order]
    ids, inverse, counts = torch.unique(src,sorted=True,return_inverse=True,return_counts=True)
    p = torch.ones(ns,dtype=torch.float32) if density is None else density.reshape(-1)
    if p.shape!=(ns,) or p.dtype!=torch.float32 or p.device.type!='cpu' or not bool(torch.isfinite(p).all()) or not bool((p>0).all()):
        raise ValueError('Source density must be finite positive CPU FP32')
    d=counts.float(); p=p[ids]; alpha=p[inverse]/d[inverse]
    variance=torch.zeros(nt); variance.scatter_add_(0,dst,alpha.square()/d[inverse])
    target_density=torch.zeros(nt); target_density.scatter_add_(0,dst,alpha)
    return TransportGraph(ns,tuple(target_shape),ids,inverse,dst,d,p,variance.sqrt(),target_density,nf,missing.numel())


def camera_graph(camera, height, width, projector=None):
    """Use existing pixel-centred FP32 forward/backward spherical projections."""
    projector = projector or NearestERPIndexProjector(height,width)
    if projector.size!=(height,width):
        raise ValueError('Projector/source shape mismatch')
    grid, mask = perspective_to_erp_grid(camera,height,width,device=torch.device('cpu'))
    uv=grid[0].reshape(-1,2); visible=mask.flatten()>0
    forward=torch.full((height*width,),-1,dtype=torch.int64)
    # border semantics and round-to-even match nearest grid_sample.
    xy=((uv[visible]+1)*torch.tensor([camera.width,camera.height])/2-.5).round().long()
    xy[:,0].clamp_(0,camera.width-1);xy[:,1].clamp_(0,camera.height-1)
    forward[visible]=xy[:,1]*camera.width+xy[:,0]
    backward=projector.index_map(camera).flatten()
    return graph_from_maps(forward,backward,(camera.height,camera.width))


def transport_values(values, graph, generator, *, channel_chunk=4, innovations=None):
    """Transport [..., U] source values; return [..., target_shape], CPU FP32.

    X_se=q_s/d_s+(Z_se-mean_e Z_se)/sqrt(d_s), Var(X)=1/d_s.
    alpha_se=p_s/d_s; q'_t=sum(alpha*X)/sqrt(sum(alpha^2/d_s)).
    Fresh iid N(0,1) fills any unconnected target. Source density is geometric,
    initially one; no empirical normalization or arbitrary noise mixture.
    """
    if values.device.type!='cpu' or values.dtype!=torch.float32 or values.shape[-1]!=graph.source_ids.numel():
        raise ValueError('Expected CPU FP32 values over included sources')
    if channel_chunk<1 or generator.device.type!='cpu':raise ValueError('CPU generator / positive chunk required')
    leading=values.shape[:-1];flat=values.reshape(math.prod(leading),values.shape[-1])
    n, u=flat.shape; e=graph.source_index.numel(); nt=graph.target_size
    output=torch.empty((n,nt),dtype=torch.float32)
    d=graph.degree[graph.source_index]; alpha=graph.density[graph.source_index]/d
    active=graph.target_index>=0; targets=graph.target_index[active]
    den=torch.where(graph.normalizer>0,graph.normalizer,torch.ones_like(graph.normalizer))
    if innovations is not None and innovations.shape!=(n,e):raise ValueError('Wrong explicit innovation shape')
    for start in range(0,n,channel_chunk):
        stop=min(start+channel_chunk,n);rows=stop-start
        z=(torch.randn(rows,e,generator=generator,dtype=torch.float32) if innovations is None else innovations[start:stop])
        sums=torch.zeros(rows,u); sums.scatter_add_(1,graph.source_index.expand(rows,-1),z)
        x=flat[start:stop,graph.source_index]/d+(z-sums[:,graph.source_index]/d)/d.sqrt()
        out=torch.zeros(rows,nt);out.scatter_add_(1,targets.expand(rows,-1),(x*alpha)[:,active]);out/=den
        holes=graph.normalizer==0
        if bool(holes.any()):out[:,holes]=torch.randn(rows,int(holes.sum()),generator=generator)
        output[start:stop]=out
    return output.reshape(*leading,*graph.target_shape)


@torch.no_grad()
def initialize_gwtf_shared_noise(backend,cameras,config,batch_size=1,*,camera_slots=None,execution_order=None):
    config.validate()
    cameras=tuple(cameras);cams=native_cameras(backend,cameras)
    if not cameras or batch_size<1:raise ValueError('Positive batch and nonempty camera set required')
    slots=(list(camera_slots) if camera_slots is not None else
           [hashlib.sha256(json.dumps(asdict(c),sort_keys=True).encode()).hexdigest() for c in cameras])
    if len(slots)!=len(cams) or len(set(slots))!=len(slots):raise ValueError('Unique stable camera identities required')
    order=list(range(len(cams))) if execution_order is None else list(execution_order)
    if sorted(order)!=list(range(len(cams))):raise ValueError('Invalid execution order')
    started=time.perf_counter();source_generator=torch.Generator(device='cpu').manual_seed(config.seed)
    source=torch.randn(batch_size,backend.native_channels,config.height,config.width,generator=source_generator,dtype=torch.float32)
    source_sha=hashlib.sha256(source.numpy().tobytes()).hexdigest()
    projector=NearestERPIndexProjector(config.height,config.width)
    states=[None]*len(cams);graphs=[None]*len(cams);geometry_seconds=sampling_seconds=0.;peak_graph=0
    for i in order:
        mark=time.perf_counter();graph=camera_graph(cams[i],config.height,config.width,projector)
        geometry_seconds+=time.perf_counter()-mark;graphs[i]=graph.summary();peak_graph=max(peak_graph,graph.bytes)
        mark=time.perf_counter();g=torch.Generator(device='cpu').manual_seed(innovation_seed(config.seed,slots[i]))
        epsilon=transport_values(source.flatten(2)[:,:,graph.source_ids],graph,g,channel_chunk=config.channel_chunk)
        state=backend.initialize_native_state(epsilon)  # Native scaling exactly once.
        expected=(batch_size,backend.native_channels,cams[i].height,cams[i].width)
        if state.shape!=expected or not bool(torch.isfinite(state).all()):raise AssertionError('Invalid initialized state')
        states[i]=state.detach().cpu();sampling_seconds+=time.perf_counter()-mark
        del graph,epsilon,state,g
    source_bytes=source.numel()*source.element_size(); del source,projector,source_generator
    record=dict(method='GWTFlow-style shared noise',algorithm='arXiv:2501.08331v4 Algorithm 1, one-hop ERP-to-native adaptation',
        config=asdict(config),camera_slots=slots,canonical_camera_sha256=camera_digest(cameras),
        native_channels=backend.native_channels,native_local_shapes=[[c.height,c.width] for c in cams],
        source_shape=[batch_size,backend.native_channels,config.height,config.width],source_sha256=source_sha,
        source_draws=1,source_density='one per ERP source cell; no solid-angle correction',
        scaling=float(backend.native_initial_noise_sigma),scaling_applications=1,
        dtype='torch.float32',rng='CPU MT19937 source seed; SHA256-derived independent per-slot innovation generators',
        graph_statistics=graphs,initial_local_sha256=states_digest(states),source_field_bytes=source_bytes,
        max_geometry_bytes=peak_graph,persistent_state_bytes=sum(s.numel()*s.element_size() for s in states),
        geometry_seconds=geometry_seconds,sampling_and_scaling_seconds=sampling_seconds,
        initialization_seconds=time.perf_counter()-started,host_process_max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        source_released=True,transport_buffers_released=True,persistent_erp_native_state=False,
        denoiser_calls=0,vae_calls=0,empirical_normalization=False,clipping=False,fixed_noise_renoising=False)
    return states,record
