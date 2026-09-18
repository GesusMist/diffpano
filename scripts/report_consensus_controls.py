"""Compact artifact audit and common-angle P/S/T display comparison."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from diffpano.camera import spherediff_camera_cover
from diffpano.consensus_audit import camera_slots,montage,thumbnail
from diffpano.projection import ProjectionCache
from diffpano.warp import StandardWarpOperator
from scripts.consensus_dense_controls import make_config
from scripts.consensus_spatial_controls import ROOT
from scripts.dense_erp_experiment import read


def main():
    torch.set_num_threads(2)
    rows=[]
    for name in ('flux','pixeldit'):
        config=make_config('P',name);cameras=spherediff_camera_cover(config.view)
        operator=StandardWarpOperator(config.warp,config.fusion,ProjectionCache(max_entries=1))
        images={label:torch.from_numpy(np.array(Image.open(ROOT/label/name/('terminal_result.png' if label=='P' else 'final_result.png')),copy=True)).permute(2,0,1)[None].float()/127.5-1 for label in ('P','S','T')}
        rows.append((name+' ERP display',[(label,thumbnail(rgb,(320,320))) for label,rgb in images.items()]))
        for i in camera_slots(cameras):
            rows.append((name+f' camera {i}',[(label,thumbnail(operator.erp_to_perspective(rgb,cameras[i]),(320,320))) for label,rgb in images.items()]))
    montage(rows,ROOT/'P-S-T.png',cell=(320,320))
    execution=read(ROOT/'execution.json')
    changed=[p for p,digest in execution['baseline_hashes'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest]
    expected={'P':4,'S':2,'T':2}
    for label,count in expected.items():
        for name in ('flux','pixeldit'):
            files=list((ROOT/label/name).iterdir())
            if len(files)!=count:raise AssertionError(f'Unexpected artifacts: {label}/{name}: {files}')
    if len(list((ROOT/'Q').iterdir()))>3 or len(list((ROOT/'R').iterdir()))!=5:raise AssertionError('Q/R output budget exceeded')
    if changed:raise AssertionError('Historical files changed: '+str(changed))
    files=[p for p in ROOT.rglob('*') if p.is_file()]
    # Store the storage audit within the existing execution manifest.
    execution['artifact_audit']=dict(historical_files_verified=len(execution['baseline_hashes']),historical_files_changed=changed,
        generated_files=len(files),file_paths=[str(p.relative_to(ROOT)) for p in files],total_bytes=sum(p.stat().st_size for p in files),
        note='Count includes shared execution/validation/review JSONs, excludes Slurm logs and source files; source comparison sheet uses PNG display values, raw metrics remain in run metadata.')
    execution['status']='all P–T runs and compact artifact audit complete; final visual review pending'
    (ROOT/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    for _ in range(8):
        size=sum(p.stat().st_size for p in files)
        if execution['artifact_audit']['total_bytes']==size:break
        execution['artifact_audit']['total_bytes']=size
        (ROOT/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    else:raise AssertionError('Storage manifest size did not stabilize')
    print(json.dumps(execution['artifact_audit'],indent=2))

if __name__=='__main__':main()
