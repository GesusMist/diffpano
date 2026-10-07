"""Bounded lossless snapshots and fixed displays, never denoiser/VAE calls."""
import torch
from diffpano.diagnostics import tensor_to_pil
from studies.gradient_blending.common import write,sha

VIEW_IDS=(7,22,59,75)
STEPS=(1,10,20)

def save_rgb(path,image):
    tensor_to_pil(image[0].detach().cpu()).save(path)

class Capture:
    view_ids=VIEW_IDS
    def __init__(self,folder,cameras,*,sufficient_statistics=False):
        self.folder=folder;self.cameras=cameras;self.capture_statistics=sufficient_statistics
        self.pending={};self.canvas=None;self.records=[]
    def __call__(self,kind,stage,*args):
        if stage not in STEPS:return
        folder=self.folder/'intermediate'/('step-%02d'%stage);folder.mkdir(parents=True,exist_ok=True)
        if kind=='contribution':
            index,rgb,contribution=args
            if index not in self.view_ids:return
            h,w=contribution.rgb.shape[-2:];camera=self.cameras[index]
            import math
            cy=int((.5-camera.pitch/math.pi)*h);cx=int((camera.yaw/(2*math.pi)+.5)*w)%w
            y0=min(max(0,cy-128),h-256);ix=(torch.arange(256,device=rgb.device)+cx-128)%w
            crop=contribution.rgb[...,y0:y0+256,:].index_select(-1,ix).detach().cpu()
            mask=contribution.valid_mask[...,y0:y0+256,:].index_select(-1,ix).detach().cpu()
            weight=contribution.weight[...,y0:y0+256,:].index_select(-1,ix).detach().cpu()
            record=dict(decoded_clean_rgb=rgb.detach().cpu(),warped_rgb_crop=crop,valid_mask=mask,weight=weight,
                        erp_crop_y0=y0,erp_crop_x_indices=ix.cpu(),camera_id=index,
                        rgb_shape=list(rgb.shape),rgb_dtype=str(rgb.dtype),rgb_min=float(rgb.min()),rgb_max=float(rgb.max()),
                        rgb_out_of_range=float(((rgb < -1)|(rgb > 1)).float().mean()))
            self.pending[index]=record
            save_rgb(folder/('view-%02d-proposal.png'%index),rgb)
            save_rgb(folder/('view-%02d-warped-crop.png'%index),crop)
        elif kind=='canvas':
            reference,result,stats=args
            save_rgb(folder/'reference.png',reference.rgb);save_rgb(folder/'consensus.png',result.rgb)
            for i in self.view_ids:
                save_rgb(folder/('view-%02d-consensus.png'%i),self.canvas.sample_view(result,self.cameras[i]))
            if self.capture_statistics:
                assert stats.mode=='both'
                mean,support=stats.guidance('poisson_mean');selected,_=stats.guidance('poisson_select')
                snapshot=dict(reference=reference.rgb.detach().cpu(),mean=[v.detach().cpu() for v in mean],
                    selected=[v.detach().cpu() for v in selected],support=[v.detach().cpu() for v in support],
                    owners=[v.detach().cpu() for v in stats.owners],selected_views=self.pending,executed_step=stage,
                    ownership=stats.owner_summary(),display_range=[-1,1],semantics='identical baseline proposals, before feedback')
                path=folder/'sufficient-statistics.pt';torch.save(snapshot,path)
                self.records.append(dict(step=stage,path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
                write(self.folder/'snapshot-inventory.json',dict(records=self.records,estimated_uncompressed_gib=2.5,
                    view_ids=self.view_ids,steps=STEPS,not_cached='all full-ERP source contributions'))
            self.pending={}
