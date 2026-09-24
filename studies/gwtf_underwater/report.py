import csv
from PIL import Image,ImageDraw,ImageFont
from studies.gwtf_underwater.common import *

require_validation();rows=[];pairs=[]
font=ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf',24)
sheet=Image.new('RGB',(2048,5*556),'white')
for i,n in enumerate(BACKENDS):
    arm={m:read(folder(n,m)/'metadata.json') for m in METHODS}
    s,g=arm['shared'],arm['gwtf']
    assert s['config']==g['config']==read(ROOT/'manifest.json')['models'][n]['config']
    fields=g['control_audit']['checked_fields']
    assert all(s[k]==g[k] for k in fields),(n,'Unmatched runtime provenance')
    assert s['prompt_sha256']==g['prompt_sha256']==sha(PROMPT)
    pair=Image.new('RGB',(4096,1080),'white');draw=ImageDraw.Draw(pair)
    for j,m in enumerate(METHODS):
        meta=arm[m];f=folder(n,m)/('gwtf-final.png' if m=='gwtf' else 'final_result.png')
        assert meta['initialization']['initial_local_sha256']==expected_hash(n,m)
        assert meta['source_hashes']==source_hashes()
        expected=89*meta['config']['generation']['num_inference_steps']
        assert meta['actual_transformer_forward_invocations']==expected
        assert meta['vae_calls']==dict(encode=0 if n=='pixeldit' else expected*2,decode=0 if n=='pixeldit' else expected+89)
        with Image.open(f) as im:
            assert im.size==(4096,2048)
            pair.paste(im.convert('RGB').resize((2048,1024),Image.Resampling.LANCZOS),(2048*j,48))
        draw.text((2048*j+12,10),n+' | '+('nearest shared ERP' if m=='shared' else 'GWTFlow'),font=font,fill='black')
        rows.append(dict(backend=n,method=m,image=str(f),metadata=str(folder(n,m)/'metadata.json'),
            job=meta['environment']['job'],runtime_seconds=meta['runtime_seconds'],peak_gpu_gib=meta['peak_allocated_gib'],**meta['summary']))
    pair_path=ROOT/(n+'-shared-vs-gwtf.png');pair.save(pair_path)
    sheet.paste(pair.resize((2048,540),Image.Resampling.LANCZOS),(0,i*556))
    pairs.append(dict(backend=n,figure=str(pair_path),runtime_provenance_identical=True,checked_fields=fields))
sheet.save(ROOT/'all-five-shared-vs-gwtf.png')
keys=list(dict.fromkeys(k for r in rows for k in r))
with (ROOT/'metrics.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
write(ROOT/'summary.json',dict(complete=True,prompt=str(PROMPT),width=4096,height=2048,seed=0,
    pairs=pairs,runs=rows,source_hashes=source_hashes(),limitations=['One prompt set and one seed; metrics do not determine visual quality.']))
print('ALL TEN UNDERWATER 4096x2048 ERPS COMPLETE; PAIRS MATCH',flush=True)
