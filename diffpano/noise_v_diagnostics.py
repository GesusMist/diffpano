"""Bounded V milestone diagnostics; no extra model or VAE calls."""
import base64
import io
import math
import torch
import torch.nn.functional as F
from diffpano.consensus_audit import camera_slots,milestones,view_metrics,thumbnail
from diffpano.projection import perspective_world_rays


def align_clean_views(a,b,ca,cb):
    rays=perspective_world_rays(cb,device=b.device)
    local=rays@ca.rotation(b.device)
    x,y,z=local.unbind(-1);safe=torch.where(z.abs()>1e-12,z,torch.ones_like(z))
    u=x/safe/math.tan(math.radians(ca.fov_x)/2)
    v=-y/safe/math.tan(math.radians(ca.fov_y)/2)
    mask=((z>0)&(u.abs()<=1)&(v.abs()<=1))[None,None]
    warped=F.grid_sample(a,torch.stack((u,v),-1)[None],mode='bilinear',padding_mode='border',align_corners=False)
    if not bool(mask.any()):raise AssertionError('Diagnostic views do not overlap')
    error=(warped-b).abs();values=b.expand_as(warped).masked_select(mask.expand_as(warped));std=float(values.std(unbiased=False))
    mae=float(error.masked_select(mask.expand_as(error)).mean())
    return dict(mae=mae,normalized_mae=mae/max(std,1e-8),target_std=std,valid_fraction=float(mask.float().mean()),
                convention='local clean RGB camera A bilinearly projected directly into camera B, shared visible angular footprint'),warped,mask


def encoded_thumbnail(value):
    out=io.BytesIO();thumbnail(value,(320,320)).save(out,format='PNG')
    return base64.b64encode(out.getvalue()).decode()


class NoiseVAudit:
    backend_name='v_no_extra_vae_decode'
    def __init__(self,cameras,steps):
        self.cameras=cameras;self.slots=camera_slots(cameras);self.steps=milestones(steps)
        self.records=[];self.pending={};self.overlaps=[];self.thumbnails={}

    def wants(self,step,slot):return step in self.steps and slot in self.slots

    def observe(self,step,slot,local,consensus,unused_roundtrip):
        self.records.append(dict(step=step,slot=slot,local=view_metrics(local),consensus=view_metrics(consensus,local)))
        if slot in self.slots[:2]:self.pending[(step,slot)]=local.detach().cpu().clone()
        a,b=self.slots[:2]
        if (step,a) in self.pending and (step,b) in self.pending:
            left=self.pending.pop((step,a));right=self.pending.pop((step,b))
            metrics,aligned,mask=align_clean_views(left,right,self.cameras[a],self.cameras[b])
            self.overlaps.append(dict(step=step,slots=[a,b],**metrics))
            if step==self.steps[-1]:
                # Embedded compact visualization copies, not raw states or per-camera files.
                self.thumbnails=dict(aligned_local_a_png=encoded_thumbnail(torch.where(mask,aligned,torch.zeros_like(aligned))),
                                     corresponding_local_b_png=encoded_thumbnail(torch.where(mask,right,torch.zeros_like(right))))

    def finish(self,last_clean,terminal,cameras,operator,coefficients):
        if self.pending:raise AssertionError('Incomplete matched local diagnostic pair')
        return dict(milestones=self.steps,slots=self.slots,stages=self.records,aligned_local_overlap=self.overlaps,
                    final_local_overlap_thumbnails=self.thumbnails,final_coefficients=[float(x) for x in coefficients],
                    extra_denoiser_calls=0,diagnostic_vae_decodes=0,diagnostic_vae_encodes=0,
                    note='S historical local/milestone tensors were not saved; use S final metrics only. New local-pair MAE differs from existing view-to-consensus MAE.')
