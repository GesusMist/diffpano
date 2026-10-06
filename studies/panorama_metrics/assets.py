"""Pinned, bounded public downloads. Records individual failures without blocking other metrics."""
import ast
import concurrent.futures
import re
import subprocess
import urllib.request
from .common import *

def download(name,url,destination,expected=None):
    p=Path(destination);p.parent.mkdir(parents=True,exist_ok=True)
    try:
        if p.exists() and (expected is None or sha(p)==expected):
            return dict(name=name,status='available',path=str(p),sha256=sha(p),url=url,bytes=p.stat().st_size)
        temp=p.with_name(p.name+'.part')
        with urllib.request.urlopen(url,timeout=90) as r:
            if 'text/html' in r.headers.get('Content-Type',''):raise ValueError('Download returned HTML, not evaluator/data bytes')
            with temp.open('wb') as f:
                while True:
                    b=r.read(4*1024*1024)
                    if not b:break
                    f.write(b)
        h=sha(temp)
        if expected and h!=expected:raise ValueError('Checkpoint hash does not match published digest')
        os.replace(temp,p)
        return dict(name=name,status='available',path=str(p),sha256=h,url=url,bytes=p.stat().st_size)
    except Exception as e:return dict(name=name,status='blocked_download',url=url,error=str(e))

def main():
    assert os.environ.get('SLURM_JOB_ID'),'Downloads and archive audit run in CPU batch'
    assert __import__('shutil').disk_usage(str(ASSETS)).free>20*1024**3
    sources={}
    for p in sorted((ASSETS/'sources').iterdir()):
        if (p/'.git').is_dir():sources[p.name]=dict(path=str(p),commit=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip())
    atomic(ASSETS/'source_revisions.json',sources)
    def assignment(path,key):
        tree=ast.parse(path.read_text())
        for n in tree.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key for t in n.targets):return ast.literal_eval(n.value)
        raise KeyError(key)
    clipurl=assignment(ASSETS/'sources/CLIP/clip/clip.py','_MODELS')['ViT-B/32']
    incurl=assignment(ASSETS/'sources/torch-fidelity/torch_fidelity/feature_extractor_inceptionv3.py','URL_INCEPTION_V3')
    faedurl=re.search(r'\[faed.ckpt\]\(([^)]+)\)',(ASSETS/'sources/PanFusion/README.md').read_text()).group(1)+'&download=1'
    revision=read(ASSETS/'dataset-source.json')['sha']
    plan=dict(dataset='zhengli1013/Pano_data',revision=revision,archive='SUN360.zip',
      policy='Use the released SUN360 test split in full if explicitly identifiable; otherwise block reference-dependent scores until split is verified. No outcome-based selection.',
      frozen_before_metrics=True,created=now(),expected_archive_bytes_approx=4360000000)
    if not (ASSETS/'reference-plan.json').exists():atomic(ASSETS/'reference-plan.json',plan)
    tasks=[('clip',clipurl,ASSETS/'weights/ViT-B-32.pt',clipurl.split('/')[-2]),
           ('inception',incurl,ASSETS/'weights/inception-v3-compat.pth',None),
           ('faed',faedurl,ASSETS/'weights/faed.ckpt',None),
           ('sun360','https://huggingface.co/datasets/zhengli1013/Pano_data/resolve/'+revision+'/SUN360.zip',ASSETS/'references/SUN360.zip',None)]
    results={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(download,*t):t[0] for t in tasks}
        for f in concurrent.futures.as_completed(futures):
            r=f.result();results[r['name']]=r;atomic(ASSETS/'asset_status.json',results);print(json.dumps(r),flush=True)
    if results['sun360']['status']=='available':
        import zipfile
        with zipfile.ZipFile(results['sun360']['path']) as z:
            names=z.namelist();atomic(ASSETS/'reference-archive-inventory.json',dict(names=names,archive_sha256=results['sun360']['sha256']))
            print('ARCHIVE_SAMPLE',names[:40],flush=True)
            print('SPLIT_CANDIDATES',[n for n in names if n.endswith(('.txt','.json','.csv'))][:60],flush=True)

if __name__=='__main__':main()
