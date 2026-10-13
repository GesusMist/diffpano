"""Study-only injection into the existing clean-consensus and terminal hooks."""
from dataclasses import replace
import math
from diffpano.poisson_reference import ConnectivityCache
from diffpano.projection import perspective_to_erp_grid
import time
import torch
from diffpano.gradient_fusion import GuidanceAccumulator, reconstruct
from studies.tt_cea.canvas import CanvasOperator
from studies.tt_cea.pipeline import ExperimentalPipeline


class StudyCanvas(CanvasOperator):
    def __init__(self, original, settings, cameras, ids, observer=None):
        if original.spec.projection != 'erp':
            raise ValueError('Gradient study supports ERP only, including RGB control')
        if len(cameras) != len(ids) or len(set(ids)) != len(ids):
            raise ValueError('One unique stable ID per camera required')
        if any(isinstance(i,bool) or not isinstance(i,int) or i < 0 for i in ids):
            raise ValueError('Stable integer camera IDs required')
        # Preserve the actual original operator, cache, weights, and guards.
        self.__dict__.update(original.__dict__)
        self.settings = settings
        self.ids = {camera: i for camera, i in zip(cameras, ids)}
        if len(self.ids) != len(ids):raise ValueError('Duplicate camera keys')
        self.observer = observer
        self.stage = None
        self.records = []
        self.last_reference = None
        self.owner_cache = None
        self.diagnostic_seconds = 0.
        self.connectivity = ConnectivityCache()
        self.anchor_camera = min(cameras,key=lambda c:(-math.cos(c.pitch)*math.cos(c.yaw),self.ids[c]))
        self.anchor_geometry = None

    def observe(self, kind, *args):
        if self.observer is None:return
        if self.device.type == 'cuda':torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        self.observer(kind, self.stage, *args)
        if self.device.type == 'cuda':torch.cuda.synchronize(self.device)
        self.diagnostic_seconds += time.perf_counter()-started

    def make_accumulator(self,batch_size):
        acc = super().make_accumulator(batch_size)
        capture = self.observer is not None and self.observer.capture_statistics and self.stage in (1,10,20)
        mode = 'both' if capture and self.settings.mode != 'poisson_max' else self.settings.mode
        if mode != 'rgb':
            if self.device.type == 'cuda':torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            acc['gradient'] = GuidanceAccumulator(acc['main'].previous, mode)
            if self.settings.mode == 'rgb':
                if self.device.type == 'cuda':torch.cuda.synchronize(self.device)
                self.diagnostic_seconds += time.perf_counter()-started
        return acc

    def accumulate(self,acc,perspective_rgb,camera):
        if 'gradient' not in acc:
            return super().accumulate(acc,perspective_rgb,camera)
        with torch.autocast(device_type=self.device.type, enabled=False):
            contribution = self.standard.perspective_to_erp(perspective_rgb,camera,(self.spec.height,self.spec.width))
            acc['main'].accumulate(contribution)
            if self.settings.reference_mode == 'single_pixel' and camera == self.anchor_camera:
                if self.anchor_geometry is None:
                    grid,valid=perspective_to_erp_grid(camera,self.spec.height,self.spec.width,device=self.device,cache=self.standard.cache)
                    distance=grid.square().sum(-1)[0].masked_fill(~valid[0,0].bool(),float('inf'))
                    flat=int(distance.argmin());y,x=divmod(flat,self.spec.width)
                    if not bool(valid[0,0,y,x]):raise ValueError('Source center has no valid ERP mapping')
                    gx,gy=grid[0,y,x].tolist()
                    self.anchor_geometry=dict(camera_id=self.ids[camera],erp_y=y,erp_x=x,
                        source_x=(gx+1)*camera.width/2-.5,source_y=(gy+1)*camera.height/2-.5,
                        selection='closest valid ERP ray to source optical axis; stable ERP row-major tie',fallback=False)
                a=self.anchor_geometry
                acc['anchor']=dict(a,color=contribution.rgb[...,a['erp_y'],a['erp_x']].clone())
            if self.settings.mode == 'rgb' and self.device.type == 'cuda':torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            acc['gradient'].accumulate(contribution,self.ids[camera])
            if self.settings.mode == 'rgb':
                if self.device.type == 'cuda':torch.cuda.synchronize(self.device)
                self.diagnostic_seconds += time.perf_counter()-started
            if self.observer is not None and self.stage in (1,10,20) and self.ids[camera] in self.observer.view_ids:
                self.observe('contribution',self.ids[camera],perspective_rgb,contribution)

    def finalize(self,acc):
        if self.device.type == 'cuda':torch.cuda.synchronize(self.device)
        finalization_start = time.perf_counter()
        reference = super().finalize(acc)
        self.last_reference = reference.rgb
        record = dict(stage=self.stage,mode=self.settings.mode,reference_mode=self.settings.reference_mode,terminal=self.stage=='terminal')
        result = reference
        statistics = acc.get('gradient')
        if self.settings.mode != 'rgb':
            guidance,support = statistics.guidance(self.settings.mode)
            if bool((reference.contributor_count <= 0).any()):raise ValueError('Gradient study requires full pixel coverage')
            graph=self.connectivity.check(support)
            rgb,diagnostics = reconstruct(reference.rgb,guidance,support,self.settings,anchor=acc.get('anchor'),connectivity=self.connectivity)
            diagnostics['connectivity']=graph
            result = replace(reference,rgb=rgb)
            record.update(diagnostics)
            record['owner_boundaries'] = statistics.owner_summary()
        if statistics is not None and statistics.owners is not None and self.settings.mode != 'poisson_max':
            if self.owner_cache is None:
                # Constant-size geometry cache; compare ownership on every observed interval.
                self.owner_cache = tuple(owner.clone() for owner in statistics.owners)
            elif not all(torch.equal(a,b) for a,b in zip(self.owner_cache,statistics.owners)):
                raise AssertionError('Geometry-driven gradient ownership changed during generation')
            record['ownership_consistent'] = True
        if self.device.type == 'cuda':torch.cuda.synchronize(self.device)
        record['fusion_finalize_seconds'] = time.perf_counter()-finalization_start
        self.records.append(record)
        if self.observer is not None:self.observe('canvas',reference,result,statistics)
        return result


class StudyPipeline(ExperimentalPipeline):
    """Only wrap existing hooks; the denoising loop and all bridge arithmetic are inherited."""
    def __init__(self,backend,cameras,config,settings,ids,*,observer=None,projection='erp',size=(2048,4096)):
        super().__init__(backend,cameras,config,projection=projection,size=size)
        self.canvas = StudyCanvas(self.canvas,settings,cameras,ids,observer)
        self.interval_records = []

    def advance_interval(self,states,conditionings,interval,canvas_operator=None,*,pass_kind,diagnostics=None):
        if canvas_operator is not None:raise ValueError('Study canvas cannot be replaced implicitly')
        self.canvas.stage = interval.k+1
        before = self.canvas.diagnostic_seconds
        output,summary = super().advance_interval(states,conditionings,interval,pass_kind=pass_kind,diagnostics=diagnostics)
        summary['diagnostic_seconds'] = self.canvas.diagnostic_seconds-before
        self.interval_records.append(summary)
        return output,summary

    def terminal(self,states):
        self.canvas.stage = 'terminal'
        return super().terminal(states)
