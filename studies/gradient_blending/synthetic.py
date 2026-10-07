"""Known-target operator experiments; selection is allowed to perform worse."""
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from diffpano.gradient_fusion import GradientFusionAccumulator,GradientSettings,edges
from diffpano.config import FusionConfig
from diffpano.projection import ERPContribution
from studies.gradient_blending.common import OUT,write

def run():
    torch.manual_seed(71);h,w=64,128
    yy,xx=torch.meshgrid(torch.arange(h)/h,torch.arange(w)/w,indexing='ij')
    texture=.35*torch.sin(xx*2*torch.pi*12)+.2*torch.cos(yy*2*torch.pi*8)
    lines=((torch.arange(w)%16)==0).float()[None].expand(h,w)*.65
    target=torch.stack((texture+lines,texture*.7,texture*.4+yy*.25))[None]
    smooth=torch.stack((.4*torch.sin(xx*2*torch.pi),yy*.6-.3,.2*torch.cos(xx*2*torch.pi)+yy*.15))[None]
    blur=F.avg_pool2d(F.pad(target,(2,2,2,2),mode='circular'),5,stride=1)
    alpha=(.5+.45*torch.cos(xx*2*torch.pi))[None,None]
    blend=(xx<.5)[None,None]
    cases=[('identical periodic thin lines',target,target,target,alpha,1-alpha),
           ('smooth color disagreement',smooth,smooth+.08,smooth-.08,alpha,1-alpha),
           ('one-pixel misalignment',target,target,torch.roll(target,1,-1),alpha,1-alpha),
           ('complementary sharp blurred regions',target,torch.where(blend,target,blur),torch.where(blend,blur,target),alpha,1-alpha)]
    rows=[];canvas=Image.new('RGB',(6*256,len(cases)*154),'white');draw=ImageDraw.Draw(canvas)
    cfg=FusionConfig(mode='weighted_average',weight_mode='spherediff_center')
    for row,(name,truth,a,b,wa,wb) in enumerate(cases):
        values=[truth,a,b];record=dict(case=name,metrics={})
        for mode in ('rgb','poisson_mean','poisson_select'):
            acc=GradientFusionAccumulator(torch.zeros_like(truth),cfg,GradientSettings(mode))
            for i,(v,weight) in enumerate([(a,wa),(b,wb)]):acc.accumulate(ERPContribution(v,torch.ones_like(weight),weight),i)
            image=acc.finalize().erp_rgb;values.append(image)
            record['metrics'][mode]=dict(reconstruction_mse=float((image-truth).square().mean()),
                gradient_mse=float(sum((u-v).square().mean() for u,v in zip(edges(image),edges(truth)))/2),
                texture_std_ratio=float(image.std()/truth.std()),solver=acc.diagnostics)
        for col,(label,image) in enumerate(zip(('target','source 0','source 1','rgb','poisson_mean','poisson_select'),values)):
            rgb=((image[0].permute(1,2,0).clamp(-1,1)+1)*127.5).round().byte().numpy()
            canvas.paste(Image.fromarray(rgb).resize((256,128)),(col*256,row*154+24))
            draw.text((col*256+4,row*154+4),label,fill='black')
        rows.append(record)
    folder=OUT/'synthetic';folder.mkdir(exist_ok=True)
    canvas.save(folder/'known-target-comparisons.png')
    write(folder/'metrics.json',dict(cases=rows,display='fixed [-1,1] for all panels',
        caveat='Confidence is geometric, not an oracle for sharpness. Misalignment and ownership transitions may produce double edges or halos; compare target errors, not energy alone.'))
    return rows
