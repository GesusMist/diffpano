"""One compact three-method comparison sheet per backend; no inference."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from diffpano.camera import spherediff_camera_cover
from diffpano.consensus_audit import camera_slots,montage,thumbnail
from diffpano.projection import ProjectionCache
from diffpano.warp import StandardWarpOperator
from scripts.noise_v_common import ROOT,S_ROOT,read,s_config


def main(name):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Full-resolution comparison requires compute allocation')
    torch.set_num_threads(2);c,s=s_config(name);cams=spherediff_camera_cover(c.view)
    op=StandardWarpOperator(c.warp,c.fusion,ProjectionCache(max_entries=1))
    methods=['S-direct-local','V-independent-erp','V-shared-erp'];images={};metadata={}
    for method in methods:
        folder=S_ROOT/name if method=='S-direct-local' else ROOT/name/method
        metadata[method]=read(folder/'metadata.json')
        images[method]=torch.from_numpy(np.array(Image.open(folder/'final_result.png'),copy=True)).permute(2,0,1)[None].float()/127.5-1
    left=metadata['V-independent-erp']['initialization'];right=metadata['V-shared-erp']['initialization']
    for key in ('first_camera_sha256','map_sha256','native_channels','native_local_shapes','scaling','source_shape'):
        if left[key]!=right[key]:raise AssertionError('Independent/shared pairing failed: '+key)
    rows=[('terminal ERP',[(method,thumbnail(image,(320,320))) for method,image in images.items()])]
    for i in camera_slots(cams):
        rows.append((f'final camera {i}',[(method,thumbnail(op.erp_to_perspective(image,cams[i]),(320,320))) for method,image in images.items()]))
    for key,label in [('aligned_local_a_png','local A in B frame'),('corresponding_local_b_png','local B, same footprint')]:
        cells=[('S local prediction not saved',Image.new('RGB',(320,320),(245,245,245)))]
        for method in methods[1:]:
            encoded=metadata[method]['audit']['stage_audit']['final_local_overlap_thumbnails'][key]
            cells.append((method,Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGB')))
        rows.append((label,cells))
    montage(rows,ROOT/(name+'-comparison.png'),cell=(320,320))
    path=ROOT/'results.json';result=read(path) if path.exists() else dict(experiment='V',backends={})
    result['backends'][name]={method:dict(final_metrics=m['final_metrics'],runtime_seconds=m['runtime_seconds'],
        peak_allocated_gib=m['peak_allocated_gib'],host_max_rss_gib=m['host_max_rss_gib'],
        guided_predictions=m['audit']['guided_predictions'],actual_forwards=m['actual_transformer_forward_invocations'],
        milestone_agreement=m.get('milestone_agreement'),aligned_local_overlap=m['audit']['stage_audit'].get('aligned_local_overlap'),
        local_consensus_stages=m['audit']['stage_audit'].get('stages'),
        initialization=m.get('initialization'),slurm_job_id=m['slurm_job_id']) for method,m in metadata.items()}
    result['historical_S_milestones_available']=False
    result['comparison_job_'+name]=os.environ['SLURM_JOB_ID']
    path.write_text(json.dumps(result,indent=2)+'\n')
    execution=read(ROOT/'execution.json')
    changed=[p for p,h in execution['historical_hashes'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    if changed:raise AssertionError('Historical artifact changed: '+str(changed))
    for method in methods[1:]:
        if len(list((ROOT/name/method).iterdir()))!=2:raise AssertionError('Unexpected per-run artifacts')
    print('Completed',name,'comparison; historical hashes preserved:',len(execution['historical_hashes']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=['pixeldit','flux']);a=p.parse_args();main(a.backend)
