"""Isolated ruins factorial framework; historical dense modes keep their guards."""
import hashlib
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from diffpano.camera import PerspectiveCamera
from diffpano.config import load_experiment_config
from diffpano.dense_consensus import DenseERPLocalCurrentStatePipeline
from diffpano.erp_local_consensus import camera_digest
from diffpano.projection import ProjectionCache
from diffpano.vae_residual import vae_residual, recover_identity
from diffpano.warp import StandardWarpOperator, LaplacianPyramidWarpOperator

BACKENDS = ('sd2', 'sana', 'flux', 'sd35', 'pixeldit')
REVISIONS = {'sd2':'f5bc1bd97485577aa0b946fa8a9004e2ec147402', 'sana':'e2b3c0cbffebcd09d83805e88b9f5f106afc74ac', 'flux':'fa45a9eb6808ba8fdfc7cc2756f7f1a16e0921f4', 'sd35':'b940f670f0eda2d07fbb75229e779da1ad11eb80'}
PILOT = ('A0B0C0D0', 'A1B0C0D1', 'A0B1C1D0', 'A1B1C1D1')
GEOMETRY = 'outputs/vae-residual-controls/20260915-dense-lm/geometry/experiment_l.json'
SAVED_POSES = 'outputs/vae-residual-controls/20260917-noise-v/initialization-preflight.json'
PROMPT_SHA = 'e74ca0410b7f22a43842a41a08ecfe857ac196cbf7379ae2c11585017cf79de0'


def factor_cells():
    return [''.join(k+str(v) for k,v in zip('ABCD',bits)) for bits in itertools.product((0,1), repeat=4)]


def factors(cell):
    if cell not in factor_cells():
        raise ValueError('Unknown factorial cell: '+cell)
    return {k:int(cell[2*i+1]) for i,k in enumerate('ABCD')}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def angular_geometry(cameras):
    return [{k:v for k,v in asdict(c).items() if k not in ('height','width')} for c in cameras]


def saved_cameras(view, erp_size):
    """Reuse recorded canonical poses; raster size never regenerates a cover."""
    rows=json.loads(Path(SAVED_POSES).read_text())['backends']['pixeldit']['camera_poses']
    cams=tuple(PerspectiveCamera(**dict(row, height=view.height, width=view.width)) for row in rows)
    g=json.loads(Path(GEOMETRY).read_text())
    check=next(r for r in g['full_resolution_verification'] if r['view_size']==[view.height,view.width] and r['erp_size']==list(erp_size))
    if len(cams)!=89 or camera_digest(cams)!=check['camera_sha256']:
        raise AssertionError('Saved poses disagree with verified L camera digest')
    if any(c.fov_x!=80 or c.fov_y!=80 for c in cams):
        raise AssertionError('All patches must be 80 degree perspective frusta')
    return cams


def make_config(backend, cell):
    if backend not in BACKENDS:raise ValueError('Unknown backend')
    f=factors(cell)
    c=load_experiment_config('configs/experiments/erp_later/'+backend+'-l.yaml')
    c=replace(c, experiment=replace(c.experiment,name='bridge-ruins-'+backend+'-'+cell,seed=0),
        global_pipeline=replace(c.global_pipeline,mode='erp_bridge_factorial'),
        prompt=replace(c.prompt,path='prompts/ruins.txt'),
        dense_consensus=replace(c.dense_consensus,experiment='bridge_ruins_factorial',geometry_file=GEOMETRY),
        consensus_transition=replace(c.consensus_transition,mode='preserve_current_state',vae_residual_correction=backend!='pixeldit'),
        warp=replace(c.warp,mode='lpw' if f['B'] else 'standard',lpw=replace(c.warp.lpw,levels=5,lod_mode='none')),
        fusion=replace(c.fusion,mode='detail_preserving_average' if f['C'] else 'weighted_average',
                       weight_mode='spherediff_center' if f['D'] else 'uniform',spherediff_temperature=.1,alpha=1.,power=1.,epsilon=1e-6))
    if backend in REVISIONS:
        c=replace(c, model=replace(c.model, revision=REVISIONS[backend], additional_pipeline_kwargs=dict(c.model.additional_pipeline_kwargs, local_files_only=True)))
    c.validate()
    return c


def common_settings(c):
    value=c.to_dict()
    value['experiment'].pop('name')
    value['warp'].pop('mode')
    value['fusion'].pop('mode');value['fusion'].pop('weight_mode')
    return value


def validate_factorial_config(c):
    if c.model.pipeline not in BACKENDS or c.experiment.seed!=0 or c.prompt.path!='prompts/ruins.txt':
        raise ValueError('Factorial study fixes backend set, seed 0 and exact ruins prompt')
    if c.canvas.mode!='erp' or c.sampling.strategy!='spherediff_fixed' or c.sampling.rotation.enabled:
        raise ValueError('Factorial study requires fixed perspective cameras')
    if c.performance.view_batch_size!=1 or c.view.fov_x!=80 or c.view.fov_y!=80:
        raise ValueError('Factorial views are individual 80 degree frusta')
    if c.dense_consensus is None or (c.dense_consensus.experiment,c.dense_consensus.geometry_file,c.dense_consensus.prompt_assignment)!=('bridge_ruins_factorial',GEOMETRY,'spherediff_directional'):
        raise ValueError('Factorial study requires saved L geometry and directional routing')
    if c.consensus_transition.mode!='preserve_current_state' or c.consensus_transition.vae_residual_correction!=(c.model.pipeline!='pixeldit'):
        raise ValueError('Factorial study requires current-state interpolation and local identity bridge')
    if c.fusion.mode not in ('weighted_average','detail_preserving_average') or c.fusion.weight_mode not in ('uniform','spherediff_center'):
        raise ValueError('Factorial C/D must be independent weighted reducer/spatial weight choices')
    if (c.fusion.alpha,c.fusion.power,c.fusion.epsilon,c.fusion.spherediff_temperature)!=(1.,1.,1e-6,.1):
        raise ValueError('Canonical DPA and center-weight parameters are fixed')
    if (c.warp.lpw.levels,c.warp.lpw.lod_mode)!=(5,'none') or c.warp.erp_to_perspective.interpolation!='nearest' or c.warp.perspective_to_erp.interpolation!='bilinear':
        raise ValueError('Factorial study fixes the existing O/Q warp conventions')


def make_operator(c):
    cache=ProjectionCache(max_entries=2,cpu_fallback=True)
    if c.warp.mode=='lpw':
        return LaplacianPyramidWarpOperator(c.warp,c.fusion,cache,periodic_reconstruction=True)
    return StandardWarpOperator(c.warp,c.fusion,cache)


class BridgeFactorialPipeline(DenseERPLocalCurrentStatePipeline):
    """Reuse the validated Jacobi loop with local residual hooks only."""
    def __init__(self, *, backend, cameras, erp_size, warp_operator, backend_name, view_order=None):
        f=warp_operator.fusion_config
        if f.mode not in ('weighted_average','detail_preserving_average') or f.weight_mode not in ('uniform','spherediff_center'):
            raise ValueError('Invalid factorial reducer/weight combination')
        expected=LaplacianPyramidWarpOperator if warp_operator.warp_config.mode=='lpw' else StandardWarpOperator
        if type(warp_operator) is not expected:raise ValueError('Warp/config mismatch')
        self.backend=backend;self.cameras=tuple(cameras);self.erp_size=tuple(erp_size)
        if not self.cameras:raise ValueError('Fixed cameras required')
        self.camera_sha256=camera_digest(self.cameras);self.operator=warp_operator
        self.flow_transition=backend_name!='sd2';self.backend_name=backend_name
        self.uses_local_vae_bridge=backend_name!='pixeldit'
        self.bridge_mode='local_identity_preserving' if self.uses_local_vae_bridge else 'not_applicable_identity'
        self.view_order=list(range(len(cameras))) if view_order is None else list(view_order)
        if sorted(self.view_order)!=list(range(len(cameras))):raise ValueError('Invalid view order')
        self.diagnostic_operator=StandardWarpOperator(replace(warp_operator.warp_config,mode='standard'),f,warp_operator.cache)

    def _local_clean_residual(self, clean, decoded_rgb, timings):
        if not self.uses_local_vae_bridge:return None
        rt=self._timed(timings,'local_residual_encode',lambda:self.backend.encode_clean(decoded_rgb))
        # Same exact decode contributes to RGB consensus; residual stays local.
        recover_identity(clean,rt)
        return vae_residual(clean,rt).detach().cpu()

    def _consensus_native_clean(self, rgb, local_residual, timings):
        if not self.uses_local_vae_bridge:
            if local_residual is not None:raise AssertionError('PixelDiT cannot have a VAE residual')
            return rgb.float()
        synced=self._timed(timings,'synchronized_clean_encode',lambda:self.backend.encode_clean(rgb))
        if local_residual is None:raise AssertionError('Missing local VAE bridge residual')
        bridged=synced+local_residual.to(synced)
        if not bool(torch.isfinite(bridged).all()):raise AssertionError('Nonfinite bridged clean')
        return bridged
