import csv
import hashlib
import json
import os
import math
from pathlib import Path
from datetime import datetime,timezone
REPO=Path('/home/shig/diffpano')
ROOT=REPO/'outputs/metrics-evaluation/seed0-v1'
ASSETS=Path('/scratch/user/shig/diffpano/evaluation-assets')
CACHE=ROOT/'cache'
METRICS=('FAED','DS','OmniFID','Distort-FID','Seam-SSIM','Seam-Sobel','FID','KID','IS','CS')
PER_IMAGE=('DS','Seam-SSIM','Seam-Sobel','CS','CS_polar','CS_north','CS_south')

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return '-Infinity' if x<0 else 'Infinity' if x>0 else 'NaN'
    return x

def read(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(clean(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()))
    with tmp.open('w') as f:json.dump(clean(v),f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def csv_atomic(p,records,fields):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()))
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
    os.replace(tmp,p)
def now():return datetime.now(timezone.utc).isoformat()
def source_hashes():return {str(p.relative_to(REPO)):sha(p) for p in sorted((REPO/'studies/panorama_metrics').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
