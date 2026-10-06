import concurrent.futures
import numpy as np
from PIL import Image
from .common import *
from .inventory import scan
from .protocols import cpu_identity
from .render import render,CUBE
from .seams import ds,cubemap_seams

def main():
    assert os.environ.get('SLURM_JOB_ID'),'CPU batch required'
    rows=scan();identity=cpu_identity()
    def score(r):
        if r['completion_status']!='complete':return
        h=r['image_sha256'];k=digest(dict(image=h,protocol=identity));path=CACHE/'cpu'/(k+'.json')
        if path.exists():return
        with Image.open(r['output_path']) as im:rgb=np.asarray(im.convert('RGB'))
        result=dict(DS=ds(rgb));result.update(cubemap_seams(render(rgb,CUBE)))
        atomic(path,dict(image_sha256=h,protocol_sha256=identity,metrics=result,key=k))
        print('CPU_SCORED',r['id'],'DS',result['DS'],flush=True)
    # OpenCV internal threads are limited by the batch process.
    import cv2;cv2.setNumThreads(1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(score,rows))
    from .report import update
    update(rows)
if __name__=='__main__':main()
