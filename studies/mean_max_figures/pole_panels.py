"""ERP plus two perspective pole views for report figures; original RGB preserved."""
import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

RENDERER=Path('/home/shig/diffpano/studies/panorama_metrics/render.py')
spec=importlib.util.spec_from_file_location('report_world_renderer',RENDERER)
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
ERP_SIZE=(1536,768)
POLE_SIZE=352
POLE_LABEL=24
POLE_GAP=16
ERP_GAP=12
RENDER_SIZE=704
PANEL_SIZE=(ERP_SIZE[0]+ERP_GAP+POLE_SIZE,ERP_SIZE[1])
FONT=ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf',22)
PROTOCOL=dict(renderer=str(RENDERER),renderer_protocol=renderer.PROTOCOL,
    directions=[dict(label='North pole',yaw=0,pitch=90,roll=0,position='upper right'),
                dict(label='South pole',yaw=0,pitch=-90,roll=0,position='lower right')],
    fov_x=90,fov_y=90,render_size=RENDER_SIZE,display_size=POLE_SIZE,
    erp_display_size=list(ERP_SIZE),panel_size=list(PANEL_SIZE),
    source='Original full-resolution final.png, before ERP display downsampling',
    display='Bilinear periodic-longitude perspective projection, then Lanczos downsampling; unchanged exposure/color')

def validate_geometry():
    h,w=512,1024
    lon=((np.arange(w)+.5)/w-.5)*2*np.pi
    lat=(.5-(np.arange(h)+.5)/h)*np.pi
    xx,yy=np.meshgrid(lon,lat)
    world=np.stack([np.sin(xx)*np.cos(yy),np.sin(yy),np.cos(xx)*np.cos(yy)],-1).astype(np.float32)
    checks=[]
    for yaw,pitch in renderer.POLAR:
        actual=renderer.render(world,((yaw,pitch),),size=65)[0]
        expected=renderer.rays(yaw,pitch,size=65)
        error=float(np.max(np.abs(actual-expected)))
        assert error<.007,(yaw,pitch,error)
        assert np.allclose(actual[32,32],renderer.rotation(yaw,pitch)[:,2],atol=.007)
        checks.append(dict(yaw=yaw,pitch=pitch,max_world_direction_error=error))
    return dict(passed=True,checks=checks)

def make_panel(image):
    assert image.mode=='RGB' and image.size==(4096,2048),image.size
    result=Image.new('RGB',PANEL_SIZE,'white')
    result.paste(image.resize(ERP_SIZE,Image.Resampling.LANCZOS),(0,0))
    views=renderer.render(np.asarray(image),renderer.POLAR,size=RENDER_SIZE)
    draw=ImageDraw.Draw(result);x=ERP_SIZE[0]+ERP_GAP
    for i,(label,view) in enumerate(zip(('North pole','South pole'),views)):
        y=i*(POLE_SIZE+POLE_LABEL+POLE_GAP)
        bounds=draw.textbbox((0,0),label,font=FONT)
        draw.text((x+(POLE_SIZE-(bounds[2]-bounds[0]))//2,y-2),label,font=FONT,fill='black')
        result.paste(Image.fromarray(view).resize((POLE_SIZE,POLE_SIZE),Image.Resampling.LANCZOS),(x,y+POLE_LABEL))
    return result
