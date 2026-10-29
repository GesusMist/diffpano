"""Resume the official public ODI-SR archive without extracting its whole tree."""
import argparse
import hashlib
import os
from pathlib import Path
import urllib.request
from .common import RUN, write, now
URL='https://drive.usercontent.google.com/download?id=1t3eXJvMOtgBVU7lbYM4uyBIvR1uLnBVQ&export=download&authuser=0&confirm=t'
EXPECTED=1682923599
DEST=Path('/scratch/user/shig/diffpano/paper-eval-v2-assets/ODISR/lau_dataset.rar')


def download(run=RUN):
    assert os.environ.get('SLURM_JOB_ID'), 'Download in a finite CPU allocation'
    DEST.parent.mkdir(parents=True,exist_ok=True)
    partial=DEST.with_suffix('.rar.partial')
    if not DEST.exists():
        offset=partial.stat().st_size if partial.exists() else 0
        req=urllib.request.Request(URL,headers={'Range':'bytes=%d-'%offset} if offset else {})
        with urllib.request.urlopen(req,timeout=90) as response:
            if offset and response.status!=206:raise RuntimeError('Server refused a ranged resume; partial preserved')
            with partial.open('ab' if offset else 'wb') as stream:
                while True:
                    data=response.read(4*1024*1024)
                    if not data:break
                    stream.write(data);offset+=len(data)
                    if offset%(128*1024*1024)<len(data):print('DOWNLOADED',offset,flush=True)
        if partial.stat().st_size!=EXPECTED:raise RuntimeError('Incomplete official archive; partial preserved')
        with partial.open('rb') as stream:
            if stream.read(4)!=b'Rar!':raise RuntimeError('Not a RAR archive')
        partial.replace(DEST)
    assert DEST.stat().st_size==EXPECTED
    h=hashlib.sha256()
    with DEST.open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    record=dict(complete=True,downloaded_utc=now(),path=str(DEST),bytes=EXPECTED,sha256=h.hexdigest(),url=URL,
        source='https://github.com/wangh-allen/LAU-Net',drive_file_id='1t3eXJvMOtgBVU7lbYM4uyBIvR1uLnBVQ',
        job=os.environ['SLURM_JOB_ID'],contents_verified=False)
    write(Path(run)/'dataset/odisr-archive.json',record)
    print('ARCHIVE_DOWNLOADED',str(DEST),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=RUN);a=p.parse_args();download(a.run)
