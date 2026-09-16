"""Geometry-only L/M preparation using the existing standard projector."""
import math
from collections import Counter
from dataclasses import asdict
import torch
from diffpano.camera import spherediff_camera_cover
from diffpano.conditioning import expand_directional_prompts, camera_prompt_indices
from diffpano.config import ViewConfig, WarpConfig, FusionConfig
from diffpano.erp_local_consensus import camera_digest
from diffpano.projection import ProjectionCache
from diffpano.warp import StandardWarpOperator

# Bounded, deterministic candidates; 0.60 includes the official default.
OVERLAPS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
RESOLUTIONS = ((512, 1024, 512), (1024, 2048, 1024))


def contributor_stats(counts):
    x = counts.flatten().float()
    q = torch.quantile(x, x.new_tensor([.01, .05, .5]))
    return dict(coverage_percent=float((x > 0).float().mean()*100), minimum=int(x.min()),
                p01=float(q[0]), p05=float(q[1]), median=float(q[2]), mean=float(x.mean()),
                maximum=int(x.max()), multi_contributor_percent=float((x > 1).float().mean()*100))


@torch.no_grad()
def measure_cover(cameras, erp_size, *, device=torch.device('cpu')):
    # Keep one camera's full projection at a time. No models or saved tensors.
    op = StandardWarpOperator(WarpConfig(mode='standard'), FusionConfig(mode='average', weight_mode='uniform'),
                              ProjectionCache(max_entries=1))
    count = torch.zeros(1, 1, *erp_size, device=device)
    for camera in cameras:
        image = torch.ones(1, 3, camera.height, camera.width, device=device)
        contribution = op.perspective_to_erp(image, camera, erp_size)
        count.add_((contribution.valid_mask > 0).float())
    return contributor_stats(count)


def geometry_record(overlap, resolutions=RESOLUTIONS):
    records = []
    for h, w, v in resolutions:
        cameras = spherediff_camera_cover(ViewConfig(height=v, width=v, fov_x=80.0, fov_y=80.0), overlap_fraction=overlap)
        stats = measure_cover(cameras, (h, w))
        records.append(dict(erp_size=[h,w], view_size=[v,v], camera_sha256=camera_digest(cameras), **stats))
    return records


def search_dense_cover(overlaps=OVERLAPS, cap=300, resolutions=RESOLUTIONS):
    view = ViewConfig(height=32, width=32, fov_x=80.0, fov_y=80.0)
    candidates = sorted(((len(spherediff_camera_cover(view, overlap_fraction=o)), o) for o in overlaps))
    trials = []
    for count, overlap in candidates:
        if count > cap:
            continue
        cameras = spherediff_camera_cover(view, overlap_fraction=overlap)
        low = measure_cover(cameras, (128, 256))
        record = dict(camera_count=count, overlap_fraction=overlap, coarse=low)
        trials.append(record)
        print('M candidate', count, overlap, low, flush=True)
        if low['minimum'] < 5 or low['coverage_percent'] != 100:
            continue
        full = geometry_record(overlap, resolutions)
        record['full_resolution'] = full
        if all(r['minimum'] >= 5 and r['coverage_percent'] == 100 for r in full):
            return overlap, trials, full
    raise RuntimeError('No tested cover at or below %d views passes full-resolution min>=5: %s' % (cap, trials))


def preset_record(experiment, overlap, verified):
    cameras = spherediff_camera_cover(ViewConfig(height=1024, width=1024, fov_x=80.0, fov_y=80.0), overlap_fraction=overlap)
    bank = expand_directional_prompts(['top','upper','equator','lower','bottom'])
    indices = camera_prompt_indices(cameras, bank.directions).tolist() if experiment == 'L' else [8]*len(cameras)
    return dict(experiment=experiment, camera_count=len(cameras), fov=[80,80],
        construction='SphereDiff dense-equator latitude rings', overlap_fraction=overlap, yaw_count_offset=3,
        latitude_rings_degrees=dict(sorted(Counter(round(math.degrees(c.pitch),5) for c in cameras).items())),
        prompt_assignment='spherediff_directional' if experiment == 'L' else 'original_k_global_slot_8',
        prompt_slot_histogram=dict(sorted(Counter(indices).items())),
        semantic_band_histogram=dict(sorted(Counter(i//4 for i in indices).items())),
        full_resolution_verification=verified)
