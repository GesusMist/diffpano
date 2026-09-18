"""Raw-float P/Q metrics and compact detached visualization helpers."""
import math
from dataclasses import asdict
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from diffpano.diagnostics import tensor_to_pil


def milestones(steps):
    return sorted({(p * steps + 99) // 100 for p in (10, 50, 90, 100)})


def camera_slots(cameras):
    # Physical forward vectors select two adjacent equatorial directions,
    # then an upper latitude and north pole. Stable lowest-slot tie breaking.
    targets = [(0, 0), (24, 0), (0, 45), (0, 90)]
    selected = []
    for target_index, (yaw, pitch) in enumerate(targets):
        y, p = math.radians(yaw), math.radians(pitch)
        direction = torch.tensor([math.cos(p)*math.sin(y), math.sin(p), math.cos(p)*math.cos(y)])
        if target_index == 1:
            direction = cameras[selected[0]].forward()
        scores = torch.stack([c.forward() for c in cameras]) @ direction
        if target_index == 1:
            for i,c in enumerate(cameras):
                if abs(c.pitch) > 1e-6:
                    scores[i] = -2
        for i in selected:
            scores[i] = -2
        selected.append(int(scores.argmax()))
    return selected


def blur(image, sigma):
    radius = math.ceil(3*sigma)
    x = torch.arange(-radius, radius+1, device=image.device, dtype=image.dtype)
    k = torch.exp(-x.square()/(2*sigma*sigma)); k = k/k.sum()
    channels = image.shape[1]
    mode = 'reflect' if min(image.shape[-2:]) > radius else 'replicate'
    out = F.conv2d(F.pad(image, (radius,radius,0,0), mode=mode), k[None,None,None,:].expand(channels,1,1,-1), groups=channels)
    return F.conv2d(F.pad(out, (0,0,radius,radius), mode=mode), k[None,None,:,None].expand(channels,1,-1,1), groups=channels)


def image_metrics(image, reference=None, *, spherical=False):
    x = image.detach().float()
    h,w = x.shape[-2:]
    weights = torch.ones_like(x[:,:1])
    if spherical:
        weights *= torch.cos((.5-(torch.arange(h,device=x.device)+.5)/h)*math.pi)[None,None,:,None]
    def mean(value):
        return float((value*weights).sum()/(weights.sum()*value.shape[1]))
    mu = mean(x); std = math.sqrt(max(0, mean((x-mu).square())))
    record = dict(mean=mu, std=std, rgb_mean=x.mean((0,2,3)).tolist(),
                  rgb_std=x.std((0,2,3),unbiased=False).tolist(), out_of_range_fraction=mean(((x < -1)|(x > 1)).float()),
                  area_weighted=spherical, hf={})
    for scale in (1,2,4):
        hf = math.sqrt(mean((x-blur(x,scale)).square()))
        record['hf'][str(scale)] = dict(rms=hf, normalized=hf/max(std,1e-8))
    if reference is not None:
        r = reference.detach().to(x); delta=x-r
        rmean=mean(r); rstd=math.sqrt(max(0,mean((r-rmean).square())))
        record.update(mae=mean(delta.abs()),rmse=math.sqrt(mean(delta.square())),
                      contrast_ratio=std/max(rstd,1e-8),
                      retained_amplitude=mean((x-mu)*(r-rmean))/max(rstd*rstd,1e-16))
    return record


def view_metrics(image, reference=None):
    h,w=image.shape[-2:]; crop=(slice(None),slice(None),slice(h//4,3*h//4),slice(w//4,3*w//4))
    return dict(full=image_metrics(image,reference),center=image_metrics(image[crop],None if reference is None else reference[crop]))


def thumbnail(tensor, size=(240,240)):
    rendered=ImageOps.contain(tensor_to_pil(tensor.detach().cpu().clone()[0]),size)
    cell=Image.new('RGB',size,(245,245,245))
    cell.paste(rendered,((size[0]-rendered.width)//2,(size[1]-rendered.height)//2))
    return cell


def montage(rows, path, cell=(240,240)):
    # rows: (label, [(column label, PIL image), ...]); no raw tensors retained.
    width=max(len(cells) for _,cells in rows)*(cell[0]+8)+180
    sheet=Image.new('RGB',(width,len(rows)*(cell[1]+30)),(245,245,245)); draw=ImageDraw.Draw(sheet)
    for r,(label,cells) in enumerate(rows):
        y=r*(cell[1]+30);draw.text((5,y+20),label,fill='black')
        for col,(title,img) in enumerate(cells):
            x=180+col*(cell[0]+8);draw.text((x,y+4),title,fill='black');sheet.paste(img.resize(cell),(x,y+25))
    sheet.save(path)


class StageAudit:
    def __init__(self,cameras,steps,backend_name):
        self.slots=camera_slots(cameras);self.steps=milestones(steps);self.backend_name=backend_name
        self.records=[];self.rows=[];self.last_clean=None;self.extra_decodes=0
        self.cameras=[dict(slot=i,**asdict(cameras[i])) for i in self.slots]

    def wants(self,step,slot):
        return step in self.steps and slot in self.slots

    def observe(self,step,slot,local,consensus,roundtrip):
        self.records.append(dict(step=step,slot=slot,local=view_metrics(local),
            consensus=view_metrics(consensus,local),
            projection=view_metrics(roundtrip,consensus),identity_conversion=self.backend_name=='pixeldit'))
        self.rows.append((f'step {step}, camera {slot}',[(name,thumbnail(value)) for name,value in
            [('local clean',local),('returned consensus',consensus),('VAE roundtrip' if self.backend_name=='flux' else 'identity',roundtrip)]]))

    def finish(self,last_clean,terminal,cameras,operator,coefficients):
        self.last_clean=last_clean.detach().cpu().clone()
        return dict(cameras=self.cameras,milestones=self.steps,stages=self.records,
                    final_coefficients=[float(x) for x in coefficients],
                    final_output=image_metrics(terminal,last_clean,spherical=True),
                    last_clean=image_metrics(last_clean,spherical=True),
                    terminal=image_metrics(terminal,spherical=True),
                    final_views=[dict(slot=i,**view_metrics(operator.erp_to_perspective(terminal,cameras[i]),
                        operator.erp_to_perspective(last_clean,cameras[i]))) for i in self.slots],
                    diagnostic_vae_decodes=self.extra_decodes,diagnostic_vae_encodes=0,extra_denoiser_calls=0)
