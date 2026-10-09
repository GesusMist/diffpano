"""Matched ERP, fixed perspective, detail and wrap sheets, no automatic exposure."""
import itertools,os
import numpy as np
from PIL import Image
from studies.gradient_refinement.common import *
from studies.gradient_blending.figures import sheet,load,details

def main():
    from studies.panorama_metrics.render import render
    assert os.environ.get('SLURM_JOB_ID')
    manifest=read(OUT/'manifest.json');lookup={(r['prompt'],r['reference_mode'],r['gradient_mode'],r['last_fraction']):r for r in manifest['rows']}
    target=OUT/'figures';inventory=[];views=((0.,0.),(90.,0.),(180.,0.),(0.,60.))
    def compare(key,items,labels):
        rows=[lookup[item] for item in items]
        if not all(case_complete(r) for r in rows):return
        images=[load(folder(r)/'final.png') for r in rows]
        base=target/key
        sheet(base.with_name(base.name+'-erp.png'),labels,[[(im,'ERP') for im in images]],width=1024,height=512,title=key)
        crops=[details(im) for im in images];crop_labels=['equator center','equator quarter','longitude wrap']
        sheet(base.with_name(base.name+'-details.png'),labels,[[(cs[i],crop_labels[i]) for cs in crops] for i in range(3)],width=512,height=512,title=key)
        rendered=[render(np.asarray(im),views) for im in images]
        sheet(base.with_name(base.name+'-views.png'),labels,[[(Image.fromarray(v[i]),str(views[i])) for v in rendered] for i in range(4)],width=512,height=512,title=key)
        inventory.append(dict(key=key,cases=[r['key'] for r in rows]))
    for prompt,g,f in itertools.product(PROMPTS,GRADIENTS,FRACTIONS):
        compare('references/'+prompt+'-'+g+'-f'+str(f),[(prompt,r,g,f) for r in REFERENCES],list(REFERENCES))
    for prompt,r,f in itertools.product(PROMPTS,REFERENCES,FRACTIONS):
        compare('gradients/'+prompt+'-'+r+'-f'+str(f),[(prompt,r,g,f) for g in GRADIENTS],['select','max'])
    for prompt,r,g in itertools.product(PROMPTS,REFERENCES,GRADIENTS):
        compare('refinement/'+prompt+'-'+r+'-'+g,[(prompt,r,g,f) for f in FRACTIONS],['off','last 10% independent'])
    write(target/'inventory.json',dict(comparisons=inventory,png_count=3*len(inventory),
        display='Fixed nominal [-1,1] tensor_to_pil conversion; no per-image contrast/exposure normalization',
        crops='512px at equator center and quarter; longitude wrap joins last/first 256 columns',
        report_views=views,report_FOV=90,generation_FOV=80))
if __name__=='__main__':main()
