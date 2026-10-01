"""One fresh production process; completion follows numerical and file audits."""
import csv
import gc
import os
import resource
import time
import traceback
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from diffpano.diagnostics import tensor_to_pil
from diffpano.projection import erp_to_perspective,ProjectionCache
from studies.tt_cea.runtime import *
from studies.tt_cea.metadata import terminal_pole_rgb
from studies.tt_cea.cameras import report_cameras,routing
from studies.tt_cea.schedule import records,execution_plan

VIEW_LABELS=('yaw 0 pitch 0','yaw 90 pitch 0','yaw 180 pitch 0 (seam)','yaw -90 pitch 0',
             'yaw 0 pitch +60','yaw 180 pitch +60','yaw 0 pitch -60','yaw 180 pitch -60',
             'north pole yaw 0','south pole yaw 0')

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def save_png(image,path):
    path=Path(path);tmp=path.with_name(path.name+'.tmp.png');image.save(tmp)
    with Image.open(tmp) as check:check.verify()
    os.replace(tmp,path)

@torch.no_grad()
def viewport_sheet(erp,path,*,native=None,op=None):
    direct=native is not None and op.spec.projection=='cea'
    panel=512;header=24;stride=panel+header
    sheet=Image.new('RGB',(5*panel,(4 if direct else 2)*stride),'white');draw=ImageDraw.Draw(sheet)
    cache=ProjectionCache(max_entries=2,cpu_fallback=True);errors=[]
    for i,cam in enumerate(report_cameras()):
        exported=erp_to_perspective(erp,cam,interpolation='bilinear',cache=cache,vertical_padding_mode='reflect')
        views=[('exported ERP float',exported)]
        if direct:
            raw=op.sample_view(native,cam,interpolation='bilinear')
            views=[('native CEA float',raw),('exported ERP float',exported)]
            errors.append(dict(view=VIEW_LABELS[i],mean_abs_export_difference=float((raw-exported).abs().mean()),max_abs_export_difference=float((raw-exported).abs().max())))
        for j,(source,v) in enumerate(views):
            x=(i%5)*panel;y=(j*2+i//5)*stride
            sheet.paste(tensor_to_pil(v[0].cpu()).resize((panel,panel),Image.Resampling.LANCZOS),(x,y+header))
            draw.text((x+4,y+4),VIEW_LABELS[i]+' | '+source,fill='black')
    save_png(sheet,path)
    return dict(render_raster=[1024,1024],display_panel=[panel,panel],FOV_x=80,FOV_y=80,
        interpolation='bilinear; circular longitude; fixed CEA cap rule; Lanczos display reduction',
        sources=['native CEA floating result','exported ERP floating result'] if direct else ['final ERP floating result'],
        views=list(VIEW_LABELS),export_differences=errors)

@torch.no_grad()
def run(name,prompt,case):
    require_gate(name,case);assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    if case=='R0':raise ValueError('R0 is an audited reused baseline, not a new generation')
    folder=sample_dir(name,prompt,case);folder.mkdir(parents=True,exist_ok=False)
    job=os.environ['SLURM_JOB_ID'];start=time.perf_counter()
    status=dict(state='running',backend=name,prompt=prompt,case=case,job=job)
    atomic(folder/'status.json',status)
    try:
        c,b,old,row,table,groups=setup(name,prompt,case)
        group=groups[row['cameras']];p=ExperimentalPipeline(b,group['cameras'],c,projection=row['projection'])
        provenance,audit=check_runtime(c,b,p,group,old,case)
        # Prepared tables must match the isolated real-backend preflight exactly.
        pref=read(ROOT/'preflight'/(name+'.json'))
        assert records(table)==pref['schedules']['matched_budget' if case=='TB' else 'original']
        states,init,init_audit=initialize(b,name,group,old if row['cameras']=='old89' else None)
        reference=pref['initialization' if row['cameras']=='old89' else 'ea_initialization']['initial_local_sha256']
        assert init['initial_local_sha256']==reference
        setup_peak=torch.cuda.max_memory_allocated()/1024**3
        expected=row['expected_guided_predictions'];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();generation_start=time.perf_counter()
        def progress(step,total,summary):
            print(name,prompt,case,step,'/',total,'state reconstruction',summary['current_state_error_max'],flush=True)
            atomic(folder/'status.json',dict(status,completed_original_intervals=step,total_original_intervals=total,
                elapsed_seconds=time.perf_counter()-start))
        with count_calls(b,name) as counts:
            native,erp,details=p.run(states,group['conditions'],table,time_travel=case=='T1',progress=progress)
        torch.cuda.synchronize();runtime=time.perf_counter()-generation_start;del states
        assert details['guided_predictions']==counts['denoiser']==expected
        assert counts['initialize']==0 and counts['encode']==(0 if name=='pixeldit' else 2*expected)
        assert counts['decode']==(0 if name=='pixeldit' else expected+89)
        assert len(details['diagnostics'])==row['downhill_passes']
        assert tuple(erp.shape)==(1,3,2048,4096) and bool(torch.isfinite(erp).all())
        if case=='T1':assert details['original_noise_sha256_before']==details['original_noise_sha256_after']
        save_png(tensor_to_pil(erp[0].cpu()),folder/'final.png')
        if row['projection']=='cea':save_png(tensor_to_pil(native.rgb[0].cpu()),folder/'final_cea.png')
        display_start=time.perf_counter();views=viewport_sheet(erp,folder/'viewports.png',native=native,op=p.canvas)
        report_seconds=time.perf_counter()-display_start
        executed=[]
        for interval,kind in execution_plan(table,case=='T1'):executed.append(dict(records([interval])[0],pass_kind=kind))
        write_csv(folder/'schedule.csv',executed)
        flattened=[]
        for d in details['diagnostics']:
            flattened.append({**{k:v for k,v in d.items() if k!='stage_seconds'},**{'seconds_'+k:v for k,v in d['stage_seconds'].items()}})
        write_csv(folder/'diagnostics.csv',flattened)
        files=['final.png','schedule.csv','diagnostics.csv','viewports.png']+(['final_cea.png'] if row['projection']=='cea' else [])
        hashes={f:sha(folder/f) for f in files}
        for file in ['final.png']+(['final_cea.png'] if row['projection']=='cea' else []):
            with Image.open(folder/file) as im:assert im.size==(4096,2048);im.verify()
        metadata=dict(backend=name,prompt=prompt,case=case,job=job,generation_configuration=c.to_dict(),study_overlay=row,
            active_clean_projection=row['projection'],noise_projection='erp',**provenance,runtime_audit=audit,
            camera_count=89,FOV_x=80,FOV_y=80,routing=routing(group['cameras'],load_directional_prompts(c.prompt.path)),
            initialization=init,initialization_audit=init_audit,diagnostics=details,counts=dict(counts),
            prepared_intervals=records(table),initialization_seconds=init_audit['seconds'],generation_seconds=runtime,
            report_viewports_seconds=report_seconds,total_seconds=time.perf_counter()-start,
            peak_allocated_gib=max(setup_peak,torch.cuda.max_memory_allocated()/1024**3),generation_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,setup_peak_allocated_gib=setup_peak,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            host_process_max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            output_range=dict(min=float(erp.min()),max=float(erp.max()),out_of_range_fraction=float(((erp<-1)|(erp>1)).float().mean())),
            viewports=views,artifacts=hashes,source_hashes=study_hashes(),manifest_sha256=sha(ROOT/'manifest.json'),
            environment=environment(),baseline_image_sha256=sha(BASES[prompt]/name/'gwtf-final.png'),
            terminal_pole_rgb=terminal_pole_rgb(native) if row['projection']=='cea' else None,
            native_CEA_note='final_cea.png is an equal-area canvas visualization, not an ERP skybox' if row['projection']=='cea' else None)
        require_gate(name,case);atomic(folder/'metadata.json',historical_scheduler_json(metadata))
        assert read(folder/'metadata.json')['counts']['denoiser']==expected
        atomic(folder/'status.json',dict(status,state='complete',seconds=time.perf_counter()-start,
            metadata_sha256=sha(folder/'metadata.json'),artifacts=hashes))
        print('COMPLETE',name,prompt,case,counts,flush=True)
    except Exception:
        atomic(folder/'status.json',dict(status,state='failed',elapsed_seconds=time.perf_counter()-start,error=traceback.format_exc()))
        raise

if __name__=='__main__':
    args=parser(__doc__);args.add_argument('--backend',choices=BACKENDS,required=True);args.add_argument('--prompt',choices=PROMPTS,required=True)
    args.add_argument('--case',choices=[c for c in CASES if c!='R0'],required=True);a=args.parse_args();load_settings(a.config);run(a.backend,a.prompt,a.case)
