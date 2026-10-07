"""Replay two guidance rules at three lambdas on identical frozen proposals."""
import os
from dataclasses import replace
import torch
from studies.gradient_blending.common import *
from studies.gradient_blending.diagnostics import save_rgb,VIEW_IDS
from diffpano.gradient_fusion import GradientSettings,reconstruct,edges

@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    gate=read(OUT/'validation.json');assert gate['passed'] and gate['core_hashes']==core_hashes()
    base=OUT/'cases/ruins/rgb';assert read(base/'status.json')['state']=='complete'
    inventory=read(base/'snapshot-inventory.json');assert [r['step'] for r in inventory['records']]==[1,10,20]
    from studies.all_prompts.audit import configuration
    from studies.tt_cea.canvas import CanvasOperator,CanvasSpec,CanvasResult
    c,_,cameras,_,_=configuration('sana')
    op=CanvasOperator(CanvasSpec('erp',2048,4096),c.warp,c.fusion,'cuda')
    folder=OUT/'offline';folder.mkdir(exist_ok=True);records=[]
    for snapshot_record in inventory['records']:
        assert sha(snapshot_record['path'])==snapshot_record['sha256']
        data=torch.load(snapshot_record['path'],map_location='cpu',weights_only=True)
        reference=data['reference'].cuda();support=[v.cuda() for v in data['support']]
        stage=data['executed_step']
        for lam in [.01,.1,1.]:
            out=folder/('step-%02d-lambda-%g'%(stage,lam));out.mkdir(exist_ok=True)
            save_rgb(out/'rgb.png',reference)
            ones=torch.ones_like(reference[:,:1]);rgb_canvas=CanvasResult(reference,ones,ones)
            for i in VIEW_IDS:save_rgb(out/('rgb-view-%02d.png'%i),op.sample_view(rgb_canvas,cameras[i]))
            for mode,key in [('poisson_mean','mean'),('poisson_select','selected')]:
                guidance=[v.cuda() for v in data[key]]
                torch.cuda.reset_peak_memory_stats()
                image,diagnostic=reconstruct(reference,guidance,support,GradientSettings(mode,lambda_color=lam))
                diagnostic['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
                diagnostic['guidance_residual_rms']=float((sum(torch.sum(torch.where(m,d-g,0.).square()) for m,d,g in zip(support,edges(image),guidance)) /
                    (3*sum(m.sum() for m in support))).sqrt())
                diagnostic['wrap_adjacent_difference_mae']=float((image[...,0]-image[...,-1]).abs().mean())
                diagnostic['gradient_energy']=float(sum(d.square().mean() for d in edges(image))/2)
                record=dict(step=stage,mode=mode,lambda_color=lam,diagnostics=diagnostic,
                    input_snapshot_sha256=snapshot_record['sha256'],target_available=False)
                save_rgb(out/(mode+'.png'),image)
                result=CanvasResult(image,ones,ones)
                for i in VIEW_IDS:save_rgb(out/(mode+'-view-%02d.png'%i),op.sample_view(result,cameras[i]))
                write(out/(mode+'.json'),record);records.append(record)
                print('REPLAY',stage,mode,lam,diagnostic,flush=True)
                del image,result,guidance
        del data,reference,support,rgb_canvas,ones
    assert len(records)==18 and all(r['diagnostics']['converged'] for r in records)
    preserved()
    write(folder/'replay.json',dict(passed=True,core_hashes=core_hashes(),records=records,
        pilot_lambda_color=.1,lambda_decision='Keep the requested common initial value 0.1; no per-prompt tuning.',
        semantics='single-fusion comparisons on identical proposals; none feed back into baseline trajectory',
        limitations='No target for real proposals; higher edge energy does not establish better detail.',job=os.environ['SLURM_JOB_ID']))
if __name__=='__main__':main()
