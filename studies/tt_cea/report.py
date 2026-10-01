"""Audit every planned row and render fixed comparisons, including failures."""
import csv
import math
from collections import Counter
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from diffpano.diagnostics import tensor_to_pil
from diffpano.projection import erp_to_perspective,ProjectionCache
from studies.tt_cea.common import *
from studies.tt_cea.metadata import compatible_source_hashes, validate_pole_record, recovered_pole_record
from studies.tt_cea.cameras import report_cameras
from studies.tt_cea.run import VIEW_LABELS,write_csv


def panel(path,size,text='unavailable'):
    if path and Path(path).exists():
        with Image.open(path) as im:return im.convert('RGB').resize(size,Image.Resampling.LANCZOS)
    im=Image.new('RGB',size,(225,225,225));ImageDraw.Draw(im).text((12,12),text,fill='black');return im

@torch.no_grad()
def baseline_views(path,target):
    if target.exists():return
    import numpy as np
    with Image.open(path) as im:erp=torch.from_numpy(np.asarray(im.convert('RGB')).copy()).permute(2,0,1).float()[None]/127.5-1
    sheet=Image.new('RGB',(2560,1072),'white');draw=ImageDraw.Draw(sheet);cache=ProjectionCache(max_entries=2,cpu_fallback=True)
    for i,cam in enumerate(report_cameras()):
        v=erp_to_perspective(erp,cam,interpolation='bilinear',cache=cache,vertical_padding_mode='reflect')
        x=(i%5)*512;y=(i//5)*536
        sheet.paste(tensor_to_pil(v[0]).resize((512,512),Image.Resampling.LANCZOS),(x,y+24))
        draw.text((x+4,y+4),VIEW_LABELS[i]+' | historical ERP PNG',fill='black')
    sheet.save(target)


def viewport(path,index,cea=False):
    if not path or not Path(path).exists():return panel(None,(512,512))
    with Image.open(path) as im:
        source_offset=1072 if cea else 0
        x=(index%5)*512;y=source_offset+(index//5)*536+24
        return im.crop((x,y,x+512,y+512)).convert('RGB')


def audit_row(row):
    n,p,c=row['backend'],row['prompt'],row['case']
    result=dict(backend=n,prompt=p,case=c,projection=row['projection'],cameras=row['cameras'],
        expected_guided_predictions=row['expected_guided_predictions'],image=row['output'],status='pending')
    if c=='R0':
        assert sha(row['output'])==row['baseline_image_sha256']
        m=read(row['baseline_metadata']);result.update(status='reused',actual_guided_predictions=m['actual_transformer_forward_invocations'],
            encode_calls=m['vae_calls']['encode'],decode_calls=m['vae_calls']['decode'],generation_seconds=m['runtime_seconds'],
            total_seconds=m['total_seconds'],initialization_seconds=m['initialization_seconds'],
            initialization_sha256=m['initialization']['initial_local_sha256'],peak_gpu_gib=m['peak_allocated_gib'],
            peak_host_gib=m['host_process_max_rss_gib'],current_state_error_max=m['aggregate_metrics']['current_state_error_max']['max'],
            viewport_source='historical quantized ERP PNG; original float not retained')
        return result
    folder=sample_dir(n,p,c);status=folder/'status.json'
    if not status.exists():return result
    s=read(status);result['status']=s['state'];result['job']=s['job']
    if s['state']!='complete':result['error']=s.get('error');return result
    try:
        assert sha(folder/'metadata.json')==s['metadata_sha256'];m=read(folder/'metadata.json')
        assert m['source_hashes'] in compatible_source_hashes(read(ROOT/'validation.json')) and m['manifest_sha256']==sha(ROOT/'manifest.json')
        gate=read(ROOT/'validation.json')
        result['source_validation_job']=gate['job'] if m['source_hashes']==gate['source_hashes'] else read(ROOT/'provenance/pre-pole-metadata/records/validation.json')['job']
        for file,h in m['artifacts'].items():assert sha(folder/file)==h,file
        with Image.open(folder/'final.png') as im:assert im.size==(4096,2048);im.verify()
        expected=row['expected_guided_predictions'];actual=m['counts']
        assert actual['denoiser']==expected and actual['initialize']==0
        assert actual['encode']==(0 if n=='pixeldit' else 2*expected)
        assert actual['decode']==(0 if n=='pixeldit' else expected+89)
        if row['projection']=='cea':
            poles=m.get('terminal_pole_rgb')
            if m['source_hashes']==study_hashes():assert poles is not None,'Missing terminal pole RGB in revised runner'
            recovered=False
            if poles is None:
                poles=recovered_pole_record(n,p,c,m);recovered=poles is not None
            if poles is not None:validate_pole_record(poles)
            if recovered:result['terminal_pole_recovery_job']=read(ROOT/'pole-metadata-recovery'/n/'certificate.json')['job']
            result.update(terminal_pole_rgb=poles,terminal_pole_metadata_status='recomputed with matching terminal states and artifacts' if recovered else 'recorded' if poles is not None else 'missing before metadata revision')
        diagnostics=m['diagnostics'];logs=diagnostics['diagnostics']
        assert len(logs)==row['downhill_passes']
        if c=='T1':assert diagnostics['original_noise_sha256_before']==diagnostics['original_noise_sha256_after']
        with (folder/'schedule.csv').open() as f:assert len(list(csv.DictReader(f)))==row['downhill_passes']
        stages=Counter()
        for d in logs:
            for key,seconds in d['stage_seconds'].items():stages[key]+=seconds
        result.update(actual_guided_predictions=actual['denoiser'],encode_calls=actual['encode'],decode_calls=actual['decode'],
            generation_seconds=m['generation_seconds'],total_seconds=m['total_seconds'],initialization_seconds=m['initialization_seconds'],
            normal_denoising_seconds=sum(d['stage_seconds'].get('model',0) for d in logs if d['pass_kind']=='initial'),
            replay_denoising_seconds=sum(d['stage_seconds'].get('model',0) for d in logs if d['pass_kind']=='replay'),
            backward_seconds=diagnostics['backward_seconds'],projection_fusion_seconds=sum(v for k,v in stages.items() if k in ('projection_fusion','fusion_finalize','canvas_to_view')),
            terminal_seconds=sum(diagnostics['terminal_seconds'].values()),peak_gpu_gib=m['peak_allocated_gib'],peak_host_gib=m['host_process_max_rss_gib'],
            current_state_error_max=max(d['current_state_error_max'] for d in logs),
            initialization_sha256=m['initialization']['initial_local_sha256'],original_noise_sha256=diagnostics['original_noise_sha256_after'],
            native_update_mean=sum(d['native_update_mean'] for d in logs)/len(logs),
            backward_move_mean=sum(d.get('backward_move_mean',0) for d in logs)/max(1,row['replay_intervals']),
            minimum_weight_sum=min(d['minimum_weight_sum'] for d in logs),out_of_range_fraction=m['output_range']['out_of_range_fraction'],
            viewport_source='floating final canvas before PNG clipping/quantization',
            export_view_mean_abs_difference=sum(d['mean_abs_export_difference'] for d in m['viewports']['export_differences'])/10 if row['projection']=='cea' else None)
    except Exception as e:result.update(status='artifact_audit_failed',error=repr(e))
    return result


def comparisons(rows,out):
    lookup={(r['backend'],r['prompt'],r['case']):r for r in rows};viewpaths={}
    for p in PROMPTS:
        for n in BACKENDS:
            baseline=out/(p+'-'+n+'-R0-viewports.png');baseline_views(lookup[n,p,'R0']['image'],baseline);viewpaths[n,p,'R0']=baseline
            for c in CASES:
                if c!='R0':viewpaths[n,p,c]=sample_dir(n,p,c)/'viewports.png'
            for label,cases in [('time-travel',('R0','T1','TB')),('projection',('R0','C0','P0','C1'))]:
                width=512;rowheight=548;sheet=Image.new('RGB',(width*len(cases),296+10*rowheight),'white');draw=ImageDraw.Draw(sheet)
                for j,c in enumerate(cases):
                    r=lookup[n,p,c];x=j*width
                    draw.text((x+4,5),n+' '+p+' '+c+' | '+r['status'],fill='black')
                    sheet.paste(panel(r['image'],(512,256),r['status']),(x,28))
                    for i,vlabel in enumerate(VIEW_LABELS):
                        y=296+i*rowheight
                        draw.text((x+4,y+4),vlabel+' | '+('historical PNG' if c=='R0' else 'final float ERP'),fill='black')
                        sheet.paste(viewport(viewpaths[n,p,c],i,c in ('C0','C1')),(x,y+28))
                sheet.save(out/(p+'-'+n+'-'+label+'.png'))
        sheet=Image.new('RGB',(6*384,5*224),'white');draw=ImageDraw.Draw(sheet)
        for i,n in enumerate(BACKENDS):
            for j,c in enumerate(CASES):
                r=lookup[n,p,c];x=j*384;y=i*224
                draw.text((x+4,y+4),n+' '+c+' | '+r['status'],fill='black')
                sheet.paste(panel(r['image'],(384,192),r['status']),(x,y+28))
        sheet.save(out/('all-five-'+p+'.png'))


def main(config):
    load_settings(config);plan=require_gate();out=ROOT/'report';out.mkdir(parents=True,exist_ok=True)
    rows=[audit_row(r) for r in plan['rows']]
    # Pairing audit: projection-only cases have exactly the same initial native states.
    for p in PROMPTS:
        for n in BACKENDS:
            group={r['case']:r for r in rows if r['prompt']==p and r['backend']==n}
            for cases in [('R0','T1','TB','C0'),('P0','C1')]:
                hashes={group[c]['initialization_sha256'] for c in cases if 'initialization_sha256' in group[c]}
                assert len(hashes)<=1,(p,n,cases,hashes)
    comparisons(rows,out);preserved=verify_preservation();counts=dict(Counter(r['status'] for r in rows))
    review=read(out/'visual_review.json') if (out/'visual_review.json').exists() else {'status':'pending direct visual inspection','observations':[]}
    pref={n:read(ROOT/'preflight'/(n+'.json')) if (ROOT/'preflight'/(n+'.json')).exists() else None for n in BACKENDS}
    missing_poles=[dict(backend=r['backend'],prompt=r['prompt'],case=r['case']) for r in rows
        if r.get('terminal_pole_metadata_status')=='missing before metadata revision']
    summary=dict(rows=rows,counts=counts,historical_unchanged=preserved,visual_review=review,
        validation=str(ROOT/'validation.json'),preflight=pref,source_hashes=study_hashes(),
        complete=counts.get('reused',0)==10 and counts.get('complete',0)==50 and not missing_poles,
        scientific_matrix_complete=counts.get('reused',0)==10 and counts.get('complete',0)==50,
        terminal_pole_metadata_missing=missing_poles,
        execution=read(ROOT/'execution.json') if (ROOT/'execution.json').exists() else None)
    atomic(out/'summary.json',summary)
    temp=out/'summary.csv.tmp'
    if temp.exists():temp.unlink()
    write_csv(temp,rows);temp.replace(out/'summary.csv')
    text=['# Original-noise time travel and CEA controlled study','',
      'One seed (0), two fixed prompt sets, five backends; 4096×2048 final ERP. No defaults changed.',
      '',f'Artifact status: {counts}. Historical byte preservation: {preserved}.',
      '', '## Correctness and cost',
      'The validation record contains actual commands, exits, test counts, logs, durations and source hashes. Per-backend preflight records compare one frozen two-pass interval with the experimental interval on identical private-prefix states. Production metadata verifies prepared endpoint coefficients, per-view current-state reconstruction, exact initialization pairing, immutable original-noise hashes and expected physical/guided/VAE counts. See summary.csv for per-sample errors, calls, timings and peak memory.',
      '', '| Backend | Frozen interval passed | Max native error | Old89 CEA | EA89 CEA |', '|---|---|---|---|---|']
    for n,v in pref.items():
        text.append('| '+n+' | '+('pending | pending | pending | pending' if v is None else f"{v['interval_passed']} | {v.get('equivalence',{}).get('native_state_max_abs_error')} | {v['cea_passed_by_cover']['old89']} | {v['cea_passed_by_cover']['ea89']}")+' |')
    text+=['','Cost separates initialization, ordinary/replay model calls, the model-free backward bridge, projection/fusion, terminal decode/export and main-run time (see timing_scope.md for exact boundaries and Slurm allocation elapsed time). These stage timings do not count model-loading overhead as denoising. Historical R0 used more diagnostic work, so its total runtime is not a pure implementation speed benchmark.',
      '', 'Terminal CEA pole metadata: '+str(len(missing_poles))+' pre-revision samples lack the actual saved floating-point pole RGB values: '+str(missing_poles)+'. These values were used during generation but cannot be recovered exactly from quantized PNGs. No values are estimated or fabricated. Later samples save both values directly. Any approved recovery is stored separately and accepted only after terminal-state, configuration, initialization and image-artifact hashes match the original. The immutable prior validation/source snapshot and metadata-only compatibility certificate preserve provenance; numerical operators, generation AST apart from the added metadata field, and report rendering are unchanged.',
      '', '## Visual comparisons',
      'Every condition is shown, including pending, failed and visually poor samples. Each backend/prompt has fixed R0–T1–TB and R0–C0–P0–C1 sheets. The first row is the ERP; the next ten rows are fixed independent 80° viewports. Yaw 180° crosses the longitude seam. New viewports use floating final results with bilinear spherical sampling, rendered at 1024² and reduced for display. R0 viewports necessarily use the historical quantized PNG. CEA sample viewports.png compares direct native CEA with its floating ERP export under the same cap convention.',
      '', 'Visual-review status: '+review.get('status','unspecified')]
    for note in review.get('observations',[]):text+=['',note if isinstance(note,str) else json.dumps(note)]
    text+=['','Required visual questions: compare T1 with both R0 and equal-budget TB, including ghosting, saturation, duplicated structure and blur. Compare C0 with R0, and C1 with P0, to isolate projection. Compare P0 with R0 to identify camera/noise/prompt effects. Examine columns, arches, masonry, terrain and stars for ruins; boundaries, coral/rock texture and water continuity underwater; inspect seams, poles, grain and distorted structures in both. Direct native CEA viewports distinguish generation defects from ERP export loss. A poor-looking result is not a numerical gate failure.',
      '', '## Limits',
      'Equal-area does not imply distortion-free. Fixed-noise time travel is not a proven exact learned-ODE inverse. Initial GWTFlow Gaussianity does not establish Gaussianity after replay. One seed and two prompts do not establish universal improvement. Changing the camera layout also changes local noise and directional-prompt allocation; R0→C1 cannot be attributed solely to projection. No combined CEA+TT case, perceptual metric project, parameter tuning or new default was introduced.',
      '', '## Artifacts and execution',
      f'Root: `{ROOT}`. `manifest.json` freezes all 60 rows and allowed differences; `baseline_audit.json` records the ten reused originals. `preservation.json` verifies historical bytes. `geometry/coverage.json` records full-raster and million-direction coverage. `geometry/operator_tests.json` records the reduced-raster resampling stress tests at 1, 5 and 20 rounds, including poles/equator. `execution.json` records submission observations and Slurm IDs. Failed attempts remain in the ledger and logs.',
      '', 'Files added (all under studies/tt_cea):', '',*['- `'+path+'`' for path in study_hashes()],
      '', 'Slurm IDs: '+', '.join(a.get('job_id','unresolved') for a in (summary['execution'] or {}).get('attempts',[])),
      '', 'Validation logs: '+str(ROOT/'validation-attempts')]
    (out/'report.md').write_text('\n'.join(text)+'\n')
    print('REPORT',counts,'visual review',review.get('status'),flush=True)

if __name__=='__main__':
    a=parser(__doc__).parse_args();main(a.config)
