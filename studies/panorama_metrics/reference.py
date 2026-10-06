"""Freeze the complete author-released test split, independently of generated scores."""
import argparse
import concurrent.futures
import io
import zipfile
import numpy as np
from PIL import Image
from .common import *

def freeze():
    assert os.environ.get('SLURM_JOB_ID')
    asset=read(ASSETS/'asset_status.json')['sun360'];assert asset['status']=='available'
    archive=Path(asset['path']);assert sha(archive)==asset['sha256']
    with zipfile.ZipFile(archive) as z:
        names=sorted(n for n in z.namelist() if n.startswith('SUN360/test/') and n.lower().endswith('.jpg'))
        assert len(names)==4260
        records=[]
        for i,name in enumerate(names):
            data=z.read(name)
            with Image.open(io.BytesIO(data)) as im:
                assert im.width==2*im.height and im.mode=='RGB';im.load();size=list(im.size)
            records.append(dict(member=name,sha256=hashlib.sha256(data).hexdigest(),width=size[0],height=size[1]))
            if i%500==0:print('REFERENCE_HASHED',i,len(names),flush=True)
    m=dict(dataset='zhengli1013/Pano_data',revision=read(ASSETS/'dataset-source.json')['sha'],
        archive=str(archive),archive_sha256=asset['sha256'],split='SUN360/test',panorama_count=len(records),
        policy=read(ASSETS/'reference-plan.json'),records=records,
        limitations='Real mixed indoor/outdoor photographs. The custom prompts include stylization and unusual scenes; domain/content mismatch makes distribution scores exploratory.')
    path=ROOT/'reference_manifest.json'
    if path.exists():assert read(path)==m
    else:atomic(path,m)
    print('REFERENCE_FROZEN',len(records),sha(path),flush=True)

def extract():
    import torch
    import cv2
    from .features import Extractor,save_features
    cv2.setNumThreads(1);torch.set_num_threads(4)
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    m=read(ROOT/'reference_manifest.json');ex=Extractor(with_clip=False)
    assert ex.inception is not None
    gate=read(ROOT/'validation/gpu.json');assert gate['passed'] and gate['protocol_sha256']==ex.identity
    with zipfile.ZipFile(m['archive']) as z:
        for i,r in enumerate(m['records']):
            k=digest(dict(image=r['sha256'],evaluator=ex.identity));npz=CACHE/'reference_features'/(k+'.npz')
            meta=npz.with_suffix('.json')
            if npz.exists() and meta.exists():
                try:
                    old=read(meta)
                    if old['image_sha256']==r['sha256'] and old['protocol_sha256']==ex.identity and old['feature_sha256']==sha(npz):continue
                except (ValueError,KeyError,OSError):pass
            data=z.read(r['member']);assert hashlib.sha256(data).hexdigest()==r['sha256']
            with Image.open(io.BytesIO(data)) as im:rgb=np.asarray(im.convert('RGB'))
            f=ex.image_features(rgb);f.pop('logits',None)
            save_features(npz,f)
            atomic(meta,dict(image_sha256=r['sha256'],protocol_sha256=ex.identity,feature_sha256=sha(npz),feature_path=str(npz)))
            if i%100==0:print('REFERENCE_EXTRACTED',i,len(m['records']),flush=True)
    atomic(ROOT/'reference_completion.json',dict(passed=True,manifest_sha256=sha(ROOT/'reference_manifest.json'),
        protocol_sha256=ex.identity,panorama_count=len(m['records']),job=os.environ['SLURM_JOB_ID']))
    print('REFERENCE_FEATURES_COMPLETE',len(m['records']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--extract',action='store_true');a=p.parse_args()
    extract() if a.extract else freeze()
