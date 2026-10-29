"""Finite real-browser check using synthetic local fixtures, never study ratings."""
import argparse
import functools
import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import threading
import urllib.request
from .common import ROOT,RUN,read,write,sha,now


def check(run=RUN):
    import numpy as np
    from PIL import Image
    from .human import PROTOCOL
    run=Path(run);out=run/'human/browser-validation';out.mkdir(parents=True,exist_ok=True)
    observation=run/'human/ratings/observations.json'
    original=sha(observation) if observation.exists() else None
    result=dict(state='running',started_utc=now(),synthetic_only=True,real_rating_observations_before=original,
        external_page_requests_allowed=False,system_dependencies_installed=False,shared_environment_mutated=False,
        source_hashes={str(path.relative_to(ROOT)):sha(path) for path in [Path(__file__),Path(__file__).with_suffix('.js')]+sorted(Path(__file__).with_name('human_assets').glob('*')) if path.is_file()})
    write(out/'status.json',result)
    try:
        nodes=sorted((Path.home()/'.vscode-server/cli/servers').glob('Stable-*/server/node'))
        if not nodes:raise RuntimeError('No existing Node runtime found')
        node=nodes[-1];result['node']=dict(path=str(node),sha256=sha(node),version=subprocess.check_output([str(node),'--version'],text=True).strip())
        with tempfile.TemporaryDirectory(prefix='paper-eval-browser-') as temporary:
            work=Path(temporary);fixture=work/'fixture';fixture.mkdir()
            for source in Path(__file__).with_name('human_assets').iterdir():
                if source.is_file():shutil.copyfile(source,fixture/source.name)
            y,x=np.mgrid[:1024,:2048]
            pixels=np.stack([x*255//2047,y*255//1023,((x//128+y//128)%2)*96+64],axis=-1).astype(np.uint8)
            Image.fromarray(pixels).save(fixture/'synthetic-panorama.png')
            assignments=[]
            for slot in range(1,6):
                for index in range(4):
                    assignments.append(dict(assignment_id='synthetic-%d-%d'%(slot,index),pair_id='synthetic-pair-'+str(index),scene_id='SYNTHETIC',rater_slot=slot,
                        left_asset='synthetic-A',right_asset='synthetic-B',protocol_sha256='synthetic-browser-protocol',
                        prompt_description='SYNTHETIC SOFTWARE FIXTURE ONLY: longitude/latitude color chart; these are not generated scenes or participant data.',
                        state='ready' if index<3 else 'pending_assets_or_reviewed_caption'))
            public=dict(study_id='synthetic-browser-check',protocol_sha256='synthetic-browser-protocol',questions=PROTOCOL['questions'],display=PROTOCOL['display'],assignments=assignments,
                assets={name:dict(state='ready',image_path='synthetic-panorama.png') for name in ('synthetic-A','synthetic-B')})
            (fixture/'manifest.js').write_text('window.HUMAN_STUDY = '+json.dumps(public)+';\n')
            version='1.56.1';url='https://registry.npmjs.org/playwright-core/'+version
            with urllib.request.urlopen(url,timeout=60) as response:metadata=json.load(response)
            package=work/'playwright-core.tgz'
            with urllib.request.urlopen(metadata['dist']['tarball'],timeout=120) as response:package.write_bytes(response.read())
            import base64
            algorithm,expected=metadata['dist']['integrity'].split('-',1)
            if algorithm!='sha512' or base64.b64encode(hashlib.sha512(package.read_bytes()).digest()).decode()!=expected:raise ValueError('Pinned npm package integrity mismatch')
            with tarfile.open(package) as archive:
                for member in archive.getmembers():
                    if not (work/member.name).resolve().is_relative_to(work):raise ValueError('Unsafe npm archive member')
                archive.extractall(work)
            result['playwright']=dict(version=version,metadata_url=url,tarball_url=metadata['dist']['tarball'],sha256=sha(package),npm_integrity=metadata['dist']['integrity'])
            env={key:value for key,value in os.environ.items() if key!='OPENROUTER_API_KEY'}
            env.update(PLAYWRIGHT_BROWSERS_PATH=str(work/'browsers'),PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS='1')
            install=subprocess.run([str(node),str(work/'package/cli.js'),'install','chromium','--only-shell'],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=900)
            (out/'browser-install.log').write_text(install.stdout)
            if install.returncode:raise RuntimeError('Headless browser download failed; see browser-install.log')
            binaries=list((work/'browsers').glob('**/chrome-headless-shell'))
            if binaries:result['browser_executable_sha256']=sha(binaries[0])
            class QuietHandler(http.server.SimpleHTTPRequestHandler):
                def log_message(self,*unused):pass
            handler=functools.partial(QuietHandler,directory=str(fixture))
            server=http.server.ThreadingHTTPServer(('127.0.0.1',0),handler)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                url='http://127.0.0.1:%d/'%server.server_port
                executed=subprocess.run([str(node),str(Path(__file__).with_suffix('.js')),str(work/'package'),url,str(out)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
                (out/'browser-workflow.log').write_text(executed.stdout)
                if executed.returncode:raise RuntimeError('Browser workflow failed; see browser-workflow.log')
                result.update(read(out/'workflow.json'))
            finally:server.shutdown();server.server_close();thread.join(timeout=10)
        result.update(state='passed',completed_utc=now())
    except Exception as exc:
        result.update(state='blocked_or_failed',error=type(exc).__name__+': '+str(exc),completed_utc=now())
    after=sha(observation) if observation.exists() else None
    result.update(real_rating_observations_after=after,real_ratings_unchanged=after==original)
    if after!=original:raise RuntimeError('Real observations changed during synthetic test')
    result['artifacts']={path.name:sha(path) for path in out.iterdir() if path.is_file() and path.name!='status.json'}
    write(out/'status.json',result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=RUN);args=parser.parse_args()
    result=check(args.run);print(json.dumps(result,indent=2));raise SystemExit(0 if result['state']=='passed' else 1)
