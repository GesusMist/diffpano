from PIL import Image,ImageDraw,ImageFont
from studies.gwtf_erp4k.common import *
require_validation();rows=[];font=ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf',20)
sheet=Image.new('RGB',(1024,5*548),'white');draw=ImageDraw.Draw(sheet)
for i,n in enumerate(BACKENDS):
    m=read(ROOT/n/'metadata.json');old=read(BASE/n/'metadata.json')
    assert m['initial_local_states_identical_to_baseline']
    assert m['actual_transformer_forward_invocations']==89*m['config']['generation']['num_inference_steps']
    p=ROOT/n/'gwtf-final.png'
    with Image.open(p) as im:
        assert im.size==(4096,2048)
        sheet.paste(im.convert('RGB').resize((1024,512),Image.Resampling.LANCZOS),(0,i*548+32))
    draw.text((8,i*548+5),n+' | GWTFlow | 4096 x 2048 ERP',font=font,fill='black')
    rows.append(dict(backend=n,output=str(p),metadata=str(ROOT/n/'metadata.json'),job=m['environment']['job'],
        runtime_seconds=m['runtime_seconds'],peak_gpu_gib=m['peak_allocated_gib'],summary=m['summary'],baseline_summary=old['summary']))
sheet.save(ROOT/'all-five-gwtf-4096x2048.png')
write(ROOT/'summary.json',dict(complete=True,width=4096,height=2048,models=rows,source_hashes=source_hashes()))
print('ALL FIVE 4096x2048 GWTFlow ERPs COMPLETE',flush=True)
