"""Pinned feature extraction; views are transient, compact features persist by content hash."""
import functools
import io
import math
import numpy as np
from PIL import Image
from .common import *
from .render import render,HORIZONTAL,POLAR
from .protocols import protocols

class Extractor:
    def __init__(self,device='cuda',with_clip=True):
        import torch
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        self.torch=torch;self.device=device;self.p=protocols();self.assets=read(ASSETS/'asset_status.json')
        self.inception=None;self.clip=None;self.text_cache={}
        if self.assets.get('inception',{}).get('status')=='available':
            p=Path(self.assets['inception']['path']);assert sha(p)==self.assets['inception']['sha256']
            self.inception=FeatureExtractorInceptionV3('inception-v3-compat',['2048','logits_unbiased'],
                feature_extractor_weights_path=str(p)).to(device).eval()
        if with_clip and self.assets.get('clip',{}).get('status')=='available':
            import clip
            p=Path(self.assets['clip']['path']);assert sha(p)==self.assets['clip']['sha256']
            self.clip,self.preprocess=clip.load(str(p),device=device,jit=False)
            self.clip=self.clip.float().eval();self.tokenizer=clip.clip._tokenizer
        self.identity=gpu_identity(self.p)

    def image_features(self,rgb):
        torch=self.torch;out={};views=render(rgb,HORIZONTAL+POLAR)
        with torch.inference_mode():
            if self.inception is not None:
                images=torch.from_numpy(views[:8].transpose(0,3,1,2).copy()).to(self.device)
                f,l=self.inception(images);out['inception']=f.cpu().numpy();out['logits']=l.cpu().numpy()
                import py360convert
                # Published grouping, not the survey's average-of-four-FIDs shortcut.
                erp=np.asarray(Image.fromarray(rgb).resize((1024,512),Image.Resampling.BILINEAR))
                faces=py360convert.e2c(erp,face_w=256,mode='bilinear',cube_format='list')
                cube=np.stack(faces).astype(np.uint8)
                x=torch.from_numpy(cube.transpose(0,3,1,2).copy()).to(self.device)
                cf,_=self.inception(x);out['omni_faces']=cf.cpu().numpy()
                out['omni_horizontal']=out['omni_faces'][:4].mean(0)
                out['omni_up']=out['omni_faces'][4];out['omni_down']=out['omni_faces'][5]
            if self.clip is not None:
                inputs=torch.stack([self.preprocess(Image.fromarray(v)) for v in views]).to(self.device)
                e=self.clip.encode_image(inputs).float();e=e/e.norm(dim=-1,keepdim=True)
                out['clip_views']=e.cpu().numpy()
        return out

    def text_embedding(self,text):
        if text in self.text_cache:return self.text_cache[text]
        torch=self.torch;ids=self.tokenizer.encode(text)
        chunks=[ids[i:i+75] for i in range(0,len(ids),75)] or [[]]
        token=torch.zeros((len(chunks),77),dtype=torch.long,device=self.device)
        for i,chunk in enumerate(chunks):token[i,:len(chunk)+2]=torch.tensor([49406]+chunk+[49407],device=self.device)
        with torch.inference_mode():
            e=self.clip.encode_text(token).float();e=e/e.norm(dim=-1,keepdim=True)
            weights=torch.tensor([max(1,len(x)) for x in chunks],device=self.device,dtype=torch.float32)
            e=(e*weights[:,None]).sum(0)/weights.sum();e=e/e.norm()
        result=(e.cpu().numpy(),dict(payload_tokens=len(ids),chunks=[len(c) for c in chunks],truncated=False))
        self.text_cache[text]=result;return result

    def cs(self,features,prompt):
        if 'clip_views' not in features:return {},{}
        lines=prompt['effective_lines'];text=[lines[2]]*8+[lines[0],lines[4]]
        embeddings,records=zip(*(self.text_embedding(t) for t in text))
        similarities=(features['clip_views']*np.stack(embeddings)).sum(-1)*100.
        return dict(CS=float(similarities[:8].mean()),CS_polar=float(similarities[8:].mean()),
            CS_north=float(similarities[8]),CS_south=float(similarities[9])),dict(views=similarities.tolist(),tokenization=list(records))


def gpu_identity(p=None):
    p=p or protocols()
    return digest(dict(protocols={k:p[k] for k in ('CS','IS','FID','KID','OmniFID','FAED')},
        source={n:sha(REPO/'studies/panorama_metrics'/n) for n in ('features.py','render.py')}))

def save_features(path,features):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temp.open('wb') as f:np.savez_compressed(f,**features)
    os.replace(temp,path)


def main():
    import torch
    import cv2
    cv2.setNumThreads(1);torch.set_num_threads(4)
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available(),'GPU batch required'
    from .protocols import cpu_identity
    gate=read(ROOT/'validation/cpu.json');assert gate['passed'] and gate['protocol_sha256']==cpu_identity()
    from .inventory import scan
    from .report import update
    rows=scan();ex=Extractor();identity=ex.identity
    # Validate installed official models, text tokenization and numerical metrics before extracting real outputs.
    from .validate_gpu import validate
    validate(ex)
    for i,r in enumerate(rows):
        if r['completion_status']!='complete':continue
        feature_key=digest(dict(image=r['image_sha256'],evaluator=identity))
        key=digest(dict(features=feature_key,prompt=r['effective_prompt_sha256']))
        path=CACHE/'gpu'/(key+'.json');npz=CACHE/'features'/(feature_key+'.npz')
        if path.exists() and npz.exists():
            try:
                old=read(path)
                if old['key']==key and old['protocol_sha256']==identity and old['image_sha256']==r['image_sha256'] and old['effective_prompt_sha256']==r['effective_prompt_sha256'] and old['feature_sha256']==sha(npz):continue
            except (ValueError,KeyError,OSError):pass
            # Recompute corrupted or uncommitted feature pairs; never alter generation files.
            npz.unlink(missing_ok=True)
        elif npz.exists():
            npz.unlink()
        if npz.exists():
            with np.load(npz,allow_pickle=False) as f:features=dict(f)
        else:
            with Image.open(r['output_path']) as im:rgb=np.asarray(im.convert('RGB'))
            features=ex.image_features(rgb);save_features(npz,features)
        scores,details=ex.cs(features,r['prompt'])
        record=dict(validated=True,image_sha256=r['image_sha256'],effective_prompt_sha256=r['effective_prompt_sha256'],
            protocol_sha256=identity,feature_path=str(npz),feature_sha256=sha(npz),metrics=scores,cs_details=details,
            job=os.environ['SLURM_JOB_ID'],torch=torch.__version__,key=key)
        atomic(path,record)
        print('GPU_SCORED',r['id'],scores,flush=True)
        if (i+1)%42==0:update(rows)
    update(rows)

if __name__=='__main__':main()
