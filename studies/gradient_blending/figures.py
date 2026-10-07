"""Matched scientific comparison sheets with fixed crops and display scale."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from studies.gradient_blending.common import OUT,BACKEND,PROMPTS,MODES,read,write


def font(size):
    return ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf',size)

def sheet(path,columns,rows,*,width=768,height=384,title=None):
    pad=12;label=36;top=54 if title else 0
    result=Image.new('RGB',(len(columns)*(width+pad)+pad,top+len(rows)*(height+label+pad)+pad),'white')
    draw=ImageDraw.Draw(result)
    if title:draw.text((pad,10),title,font=font(24),fill='black')
    for y,row in enumerate(rows):
        for x,item in enumerate(row):
            image,name=item;pos=(pad+x*(width+pad),top+pad+y*(height+label+pad))
            draw.text(pos,columns[x]+' | '+name,font=font(18),fill='black')
            shown=ImageOps.contain(image,(width,height),Image.Resampling.LANCZOS)
            result.paste(shown,(pos[0]+(width-shown.width)//2,pos[1]+label+(height-shown.height)//2))
    path.parent.mkdir(parents=True,exist_ok=True);result.save(path)

def load(path):
    with Image.open(path) as im:return im.convert('RGB')

def details(image):
    w,h=image.size
    return [image.crop((w//2-256,h//2-256,w//2+256,h//2+256)),
            image.crop((w//4-256,h//2-256,w//4+256,h//2+256)),
            Image.fromarray(np.concatenate((np.asarray(image)[h//2-256:h//2+256,-256:],np.asarray(image)[h//2-256:h//2+256,:256]),axis=1))]

def main():
    import os,torch
    from studies.panorama_metrics.render import render
    assert os.environ.get('SLURM_JOB_ID')
    target=OUT/'figures';target.mkdir(exist_ok=True)
    for prompt in PROMPTS:
        images=[load(OUT/'cases'/prompt/mode/'final.png') for mode in MODES]
        sheet(target/(prompt+'-final-erp.png'),MODES,[[ (v,'final ERP') for v in images]],width=1024,height=512,title=prompt+' | complete generation trajectories')
        crops=[details(v) for v in images]
        labels=['equator center detail','equator quarter detail','longitude-wrap detail']
        sheet(target/(prompt+'-detail-wrap.png'),MODES,[[(crops[x][i],labels[i]) for x in range(3)] for i in range(3)],width=512,height=512,title=prompt+' | fixed full-resolution crops')
        views=[render(np.asarray(v),((0.,0.),(90.,0.),(180.,0.),(0.,60.))) for v in images]
        sheet(target/(prompt+'-fixed-views.png'),MODES,[[(Image.fromarray(views[x][i]),'view '+str(i)) for x in range(3)] for i in range(4)],width=512,height=512,title=prompt+' | fixed 90-degree report views')
    if BACKEND!='sana':
        write(target/'display.json',dict(backend=BACKEND,display='Matched fixed full-resolution crops and fixed report views',offline_replay_backend='sana',intermediate_baseline='not available for reused '+BACKEND+' controls'))
        return
    for step in (1,10,20):
        row=[(load(OUT/'cases/ruins'/mode/'intermediate'/('step-%02d'%step)/'consensus.png'),'step '+str(step)) for mode in MODES]
        sheet(target/('ruins-trajectory-step-%02d.png'%step),MODES,[row],width=1024,height=512,title='ruins | different trajectories, same executed step')
        for lam in (.01,.1,1.):
            folder=OUT/'offline'/('step-%02d-lambda-%g'%(step,lam))
            ims=[load(folder/(mode+'.png')) for mode in MODES]
            rows=[[(v,'ERP') for v in ims]]
            rows += [[(load(folder/(mode+'-view-%02d.png'%i)),'camera '+str(i)) for mode in MODES] for i in (7,59)]
            sheet(target/('replay-step-%02d-lambda-%g.png'%(step,lam)),MODES,rows,width=768,height=384,title='identical proposals | step %d | lambda %g'%(step,lam))
        folder=OUT/'cases/ruins/rgb/intermediate'/('step-%02d'%step)
        snapshot=torch.load(folder/'sufficient-statistics.pt',map_location='cpu',weights_only=True)
        for axis,(owner,support) in enumerate(zip(snapshot['owners'],snapshot['support'])):
            ids=owner[0,0].numpy().astype(np.int64);valid=support[0,0].numpy()
            color=np.stack(((ids*67+41)%211+30,(ids*107+31)%211+30,(ids*139+7)%211+30),axis=-1).astype(np.uint8)
            color[~valid]=160
            boundary=(ids!=np.roll(ids,-1,axis=1)) & valid & np.roll(valid,-1,axis=1)
            boundary[:-1] |= (ids[:-1]!=ids[1:]) & valid[:-1] & valid[1:]
            color[boundary]=255
            Image.fromarray(color).save(target/('ownership-step-%02d-%s.png'%(step,'x' if axis==0 else 'y')))
    write(target/'display.json',dict(final_display='Existing tensor_to_pil conversion, same nominal [-1,1] scaling',
        fixed_detail_boxes='512px equator-centered crops at ERP x=W/2 and W/4; wrap joins last/first 256 columns',
        ownership='color encodes stable camera ID; white marks changes between neighboring supported edge owners',
        note='Final comparisons are different trajectories. Offline replay compares identical proposals. Report views use fixed 90-degree camera for all methods; generation cameras remain 80 degrees.'))
if __name__=='__main__':main()
