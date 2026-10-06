import collections
import numpy as np
from .common import *
from .protocols import save_protocols,cpu_identity


def key(r):return '/'.join(str(r[k]) for k in ('method','backend','consensus_projection','steps','camera_strategy'))
def stats(values):
    a=np.asarray(values,dtype=np.float64)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),std=float(a.std(ddof=1)) if len(a)>1 else None)


def validate_camera_pair(a,b):
    c=a['configuration'];m=b['configuration']
    assert a['effective_prompt_sha256']==b['effective_prompt_sha256']
    assert a['model_revision']==b['model_revision']
    if str(a['model_id']).startswith('/'):
        assert Path(a['model_id']).resolve()==Path(b['model_id']).resolve()
    else:assert a['model_id']==b['model_id']
    for k in ('batch_size','num_inference_steps','true_cfg_scale'):
        assert c['generation'][k]==m['generation'][k],k
    guidance=c['pixeldit']['cfg_scale'] if a['backend']=='pixeldit' else c['generation']['guidance_scale']
    assert guidance==m['generation']['guidance_scale']
    if a['backend']=='pixeldit':
        for k in ('interval_guidance','negative_prompt'):assert c['pixeldit'][k]==m['generation'][k]
    assert c['view']['fov_x']==m['camera']['fov_x_deg']==80
    assert c['view']['fov_y']==m['camera']['fov_y_deg']==80
    assert c['view']['height']==m['backend']['local_rgb_height']
    assert c['view']['width']==m['backend']['local_rgb_width']
    assert c['erp']['height']==m['output']['height']==2048
    assert c['erp']['width']==m['output']['width']==4096
    for k in ('mode','weight_mode'):assert c['fusion'][k]==m['fusion'][k]
    assert c['fusion']['spherediff_temperature']==m['fusion']['temperature']
    assert c['warp']['mode']==m['fusion']['warp_mode']=='standard'
    assert c['warp']['erp_to_perspective']['interpolation']==m['fusion']['canvas_to_perspective_interpolation']
    assert c['warp']['perspective_to_erp']['interpolation']==m['fusion']['perspective_to_canvas_interpolation']
    assert c['consensus_transition']['mode']==m['trajectory']['transition']=='preserve_current_state'
    if a['backend']=='pixeldit':
        assert not c['consensus_transition']['vae_residual_correction'] and m['bridge']['mode']=='not_applicable_pixeldit'
    else:assert c['consensus_transition']['vae_residual_correction'] and m['bridge']['mode']=='local_identity_preserving'
    assert not m['trajectory']['time_travel']
    assert c['model']['precision']=='bf16' and m['backend']['precision']=='bfloat16'
    for k in ('cpu_offload','vae_tiling'):
        assert c['model'][k]==m['backend']['model_cpu_offload' if k=='cpu_offload' else k]
    init=a.get('initialization',{})
    init=init.get('record',init)
    assert 'gwtflow' in init['method'].lower() and m['initialization']['method']=='gwtflow'
    assert init['source_shape']==m['initialization']['source_shape']
    assert init['source_sha256']==m['initialization']['source_sha256']
    if a.get('local_native_resolution'):
        native=a['local_native_resolution']
        # Historical metadata records [height,width].
        assert list(native)==[m['backend']['native_height'],m['backend']['native_width']]
    return True

def update(rows=None):
    if rows is None:rows=read(ROOT/'inventory.json')['rows']
    p=save_protocols();identity=cpu_identity();cached={}
    for f in (CACHE/'cpu').glob('*.json'):
        c=read(f)
        if c['protocol_sha256']==identity:cached[c['image_sha256']]=c['metrics']
    # GPU extractor writes separate cache records so a missing dependency never blocks CPU scores.
    from .features import gpu_identity
    gpu={}
    for f in (CACHE/'gpu').glob('*.json'):
        try:
            c=read(f)
            if c.get('validated') and c.get('protocol_sha256')==gpu_identity(p) and sha(c['feature_path'])==c['feature_sha256']:
                gpu[(c['image_sha256'],c['effective_prompt_sha256'])]=c
        except (ValueError,KeyError,OSError):continue
    scored=[]
    for r in rows:
        scores=dict(cached.get(r.get('image_sha256'),{}));scores.update(gpu.get((r.get('image_sha256'),r['effective_prompt_sha256']),{}).get('metrics',{}))
        rec=dict(id=r['id'],group=key(r),prompt_id=r['prompt_id'],family=r['family'],completion_status=r['completion_status'],
                 image_sha256=r.get('image_sha256'),output_path=r['output_path'],**{k:scores.get(k) for k in PER_IMAGE})
        rec['evaluation_status']='computed_partial' if scores else 'excluded' if r['completion_status']=='excluded' else r['completion_status'] if r['completion_status'] in ('blocked','cancelled') else 'pending'
        rec['DS_status']='computed_linked_reimplementation' if rec['DS'] is not None else 'pending'
        rec['seam_status']='computed_ERP_derived_adaptation' if rec['Seam-SSIM'] is not None else 'pending'
        scored.append(rec);r['evaluation_status']=rec['evaluation_status']
    by_id={r['id']:r for r in scored};groups=collections.defaultdict(list)
    for r in rows:groups[key(r)].append(r)
    group_table=[]
    for name,rr in sorted(groups.items()):
        row=dict(group=name,expected=len(rr),complete=sum(r['completion_status']=='complete' for r in rr),
            excluded=sum(r['completion_status']=='excluded' for r in rr),failed=sum(r['completion_status']=='failed' for r in rr),
            blocked=sum(r['completion_status']=='blocked' for r in rr),cancelled=sum(r['completion_status']=='cancelled' for r in rr),
            family=rr[0]['family'],backend=rr[0]['backend'],method=rr[0]['method'],camera_count=rr[0]['camera_count'],camera_strategy=rr[0]['camera_strategy'],layout_seed=rr[0]['layout_seed'],
            missing=sum(r['completion_status'] not in ('complete','excluded','failed','blocked','cancelled') for r in rr),
            model_id=rr[0].get('model_id'),model_revision=rr[0].get('model_revision'),steps=rr[0]['steps'],projection=rr[0]['consensus_projection'])
        for metric in PER_IMAGE:
            v=[by_id[r['id']][metric] for r in rr if by_id[r['id']][metric] is not None]
            if v:
                s=stats(v);row.update({metric+'_'+k:value for k,value in s.items()})
                row[metric]=s['mean']
        group_table.append(row)
    paired=[]
    comparisons=[]
    for a,b in [('spherediff/flux/spherical/20/official_spherediff','diffpano/flux/erp/20/old89'),
                ('spherediff/flux/spherical/28/official_spherediff','diffpano/flux/erp/28/old89'),
                ('spherediff/flux/spherical/20/official_spherediff','spherediff/flux/spherical/28/official_spherediff'),
                ('diffpano/flux/erp/20/old89','diffpano/flux/erp/28/old89'),
                ('spherediff/sana/spherical/20/official_spherediff','diffpano/sana/erp/20/old89')]:comparisons.append((a,b))
    for backend,steps in [('sana',20),('flux',20),('sd35',40),('pixeldit',50)]:
        comparisons.append((f'diffpano/{backend}/erp/{steps}/old89',f'diffpano/{backend}/cea/{steps}/old89'))
    def paired_rows(a,b,allowed=None):
        aa={r['prompt_id']:r for r in groups.get(a,[]) if r['completion_status']=='complete' and (allowed is None or r['prompt_id'] in allowed)}
        bb={r['prompt_id']:r for r in groups.get(b,[]) if r['completion_status']=='complete' and (allowed is None or r['prompt_id'] in allowed)}
        common=sorted(set(aa)&set(bb));out=[]
        for metric in PER_IMAGE:
            usable=[n for n in common if by_id[aa[n]['id']][metric] is not None and by_id[bb[n]['id']][metric] is not None]
            if not usable:
                out.append(dict(baseline=a,comparison=b,metric=metric,status='pending_common_evaluated_prompts',n=0));continue
            d=np.array([by_id[bb[n]['id']][metric]-by_id[aa[n]['id']][metric] for n in usable])
            x=dict(baseline=a,comparison=b,metric=metric,status='paired',prompts=';'.join(usable),**stats(d),
                baseline_mean=float(np.mean([by_id[aa[n]['id']][metric] for n in usable])),
                comparison_mean=float(np.mean([by_id[bb[n]['id']][metric] for n in usable])),delta_convention='comparison minus baseline')
            if len(usable)>=10:
                rng=np.random.default_rng(0);bs=d[rng.integers(0,len(d),size=(2000,len(d)))].mean(1)
                x['ci_low'],x['ci_high']=map(float,np.quantile(bs,[.025,.975]))
                x['uncertainty']='prompt bootstrap2000; one diffusion seed; no seed-robustness inference'
            out.append(x)
        return out
    for a,b in comparisons:paired.extend(paired_rows(a,b))
    camera=[]
    for backend,steps in [('sana',20),('flux',20),('sd35',40),('pixeldit',50)]:
        for projection in ('erp','cea'):
            prefix=f'diffpano/{backend}/{projection}/{steps}/';names=[prefix+s for s in ('old89','fibonacci_n89','random_n89_seed0')]
            prompt_sets=[{r['prompt_id'] for r in groups.get(n,[]) if r['completion_status']=='complete' and r['prompt_id'] in ('ruins','underwater','firework')} for n in names]
            common=set.intersection(*prompt_sets)
            # Exact schedule equality is required before interpreting a layout contrast.
            schedules={r.get('schedule_sha256') for n in names for r in groups.get(n,[]) if r['prompt_id'] in common}
            if len(schedules)!=1:raise AssertionError(('Camera schedule mismatch',names,schedules))
            for b in names[1:]:
                for prompt in sorted(common):
                    validate_camera_pair(next(r for r in groups[names[0]] if r['prompt_id']==prompt),next(r for r in groups[b] if r['prompt_id']==prompt))
                for x in paired_rows(names[0],b,common):camera.append(dict(x,backend=backend,projection=projection,record_type='summary',common_prompts=';'.join(sorted(common))))
                for prompt in sorted(common):
                    for metric in PER_IMAGE:
                        arow=next(r for r in groups[names[0]] if r['prompt_id']==prompt);brow=next(r for r in groups[b] if r['prompt_id']==prompt)
                        av=by_id[arow['id']][metric];bv=by_id[brow['id']][metric]
                        camera.append(dict(backend=backend,projection=projection,baseline=names[0],comparison=b,prompt_id=prompt,metric=metric,
                            record_type='per_prompt',baseline_mean=av,comparison_mean=bv,mean=bv-av if av is not None and bv is not None else None,
                            status='paired' if av is not None and bv is not None else 'pending',n=1))
    from .camera_counts import comparisons as count_comparisons
    count_records=count_comparisons(groups,paired_rows,by_id)
    camera.extend(count_records)
    count_scopes={(r['baseline'],r['comparison']):r['comparison_scope'] for r in count_records}
    # Distribution metrics are added only after validated, real reference features exist.
    try:
        from .distributions import enrich
        enrich(group_table,paired,groups,gpu,rows,camera)
    except ImportError:pass
    for r in camera:
        r['comparison_scope']=count_scopes.get((r['baseline'],r['comparison']),'layout_n89')
    fields=['id','group','family','prompt_id','completion_status','evaluation_status','image_sha256','output_path',*PER_IMAGE,'DS_status','seam_status']
    csv_atomic(ROOT/'per_image_metrics.csv',scored,fields)
    gfields=['group','expected','complete','failed','excluded','missing','model_id','model_revision','steps','projection',*METRICS]
    gfields+=sorted({k for r in group_table for k in r}-set(gfields))
    csv_atomic(ROOT/'group_metrics.csv',group_table,gfields)
    pairfields=['baseline','comparison','metric','status','n','prompts','baseline_mean','comparison_mean','mean','median','std','ci_low','ci_high','delta_convention','uncertainty']
    csv_atomic(ROOT/'paired_comparisons.csv',paired,pairfields)
    csv_atomic(ROOT/'camera_comparisons.csv',camera,['backend','projection','comparison_scope','record_type','prompt_id','common_prompts','reason']+pairfields)
    count_table=[r for r in group_table if r['family']=='camera_count']
    csv_atomic(ROOT/'camera_count_metrics.csv',count_table,gfields)
    csv_atomic(ROOT/'camera_count_comparisons.csv',[r for r in camera if r['comparison_scope']!='layout_n89'],['backend','projection','comparison_scope','record_type','prompt_id','common_prompts','reason']+pairfields)
    statuses={}
    for metric in METRICS:
        count=sum(r.get(metric) is not None for r in scored) if metric in PER_IMAGE else sum(r.get(metric) is not None for r in group_table)
        state=p[metric]['status'] if 'status' in p[metric] else 'pending'
        if metric in ('FID','KID','OmniFID') and not count and (ROOT/'reference_manifest.json').exists():state='pending_reference_features'
        if count:state='computed_adaptation' if metric in ('DS','Seam-SSIM','Seam-Sobel','CS','FID','KID','IS') else 'computed'
        statuses[metric]=dict(status=state,computed_count=count,blocker=p[metric].get('blocker'),implementation=p[metric])
    atomic(ROOT/'metric_status.json',statuses)
    text=['# Panorama evaluation — seed 0','',f'Updated {now()}. Expected {len(rows)} logical outputs; complete {sum(r["completion_status"]=="complete" for r in rows)}; scored {sum(r["evaluation_status"]=="computed_partial" for r in scored)}. Inventory includes the original 300 rows and 126 camera-count rows. Blocked: {sum(r["completion_status"]=="blocked" for r in rows)}; cancelled: {sum(r["completion_status"]=="cancelled" for r in rows)}. Missing/pending and excluded cases remain in inventory.','',
        '## A. Existing benchmark and all-available descriptive scores','',
        '| Group | Complete / expected | DS ↓* | Seam-SSIM ↑† | Seam-Sobel ↓† | CS ↑ | IS ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'—' if v is None else f'{v:.5g}'
    for r in group_table:text.append('| '+r['group']+f' | {r["complete"]}/{r["expected"]} | '+' | '.join(fmt(r.get(k)) for k in ('DS','Seam-SSIM','Seam-Sobel','CS','IS'))+' |')
    text+=['','Real-reference collection metrics (exploratory):','','| Group | N panoramas | FID ↓ | KID ↓ | OmniFID ↓ | Horizontal | Up | Down | FAED ↓ | Distort-FID ↓ |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in group_table:
        text.append('| '+r['group']+' | '+str(r.get('panorama_count',0))+' | '+' | '.join(fmt(r.get(k)) for k in ('FID','KID','OmniFID','OmniFID_horizontal','OmniFID_up','OmniFID_down','FAED','Distort-FID'))+' |')
    text+=['','*DS is the author-linked survey implementation using first-order Scharr, not the unreleased original implementation (the paper describes second-order Scharr). Its normalization is preserved. †Cubemap seams are supplementary ERP-derived adaptations with explicitly fixed settings; the original code is unreleased. Their faces share the same ERP and are not independent generated views. See metric_protocols.json.','',
        '## B. FLUX step controls: strict matched prompt comparisons','',
        'Step-matched does not mean compute-matched. CEA-20 remains descriptive; CEA-28 is absent by authorization. Comparisons below use only the listed common evaluated prompts, never incomplete-versus-full groups.','',
        '| Baseline → comparison | Metric | n | Baseline | Comparison | Paired change |','|---|---|---:|---:|---:|---:|']
    for r in paired:
        if r['status'] in ('paired','paired_collection'):text.append(f'| {r["baseline"]} → {r["comparison"]} | {r["metric"]} | {r["n"]} | {fmt(r["baseline_mean"])} | {fmt(r["comparison_mean"])} | {fmt(r["mean"])} |')
    text+=['','Paired deltas are comparison minus baseline. Bootstrap intervals in CSV resample prompt IDs, never crops. One seed provides no evidence about seed robustness.','',
        '## C. Camera layout comparison','',
        'old89, Fibonacci89 phase0, and Random89 seed0 are paired on ruins, underwater, firework within backend, projection, diffusion seed and step count. Schedule hashes must match. Layout changes also affect initial noise transport, overlap and prompt routing. Three prompts support descriptive comparisons only.','',
        '| Backend / projection | Layout vs old89 | Metric | n | Mean change |','|---|---|---|---:|---:|']
    for r in camera:
        if r['comparison_scope']=='layout_n89' and r['record_type']=='summary' and r['status']=='paired':text.append(f'| {r["backend"]}/{r["projection"]} | {r["comparison"].split("/")[-1]} | {r["metric"]} | {r["n"]} | {fmt(r["mean"])} |')
    text+=['','## Camera-count study: N=70,50,30','',
        'These are 108 DiffPano and 18 requested reduced-ring SphereDiff results. SphereDiff FLUX and SANA use20 steps. SphereDiff N70 is blocked by the original assembly finite-value gate; it is not scored. Original89 references are distinct from reduced-ring SphereDiff.',
        'The table below contains three-prompt groups (firework, underwater, ruins). camera_count_comparisons.csv recomputes each89 reference on exactly these matching prompts. It also includes layout comparisons at fixed count and ring-matched method comparisons. No new generation is launched.',
        'Random N70/N50/N30 accepted seeds are0/1/7; layouts are coverage-conditioned and not nested. Camera count changes overlap, noise transport, and prompt routing as well as runtime. Equal counts or seeds do not imply equal compute or cross-method noise.', '',
        '| Group | Complete / expected | DS ↓ | Seam-SSIM ↑ | Seam-Sobel ↓ | CS ↑ | FID ↓ | KID ↓ | OmniFID ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in count_table:
        text.append('| '+r['group']+f' | {r["complete"]}/{r["expected"]} | '+' | '.join(fmt(r.get(k)) for k in ('DS','Seam-SSIM','Seam-Sobel','CS','FID','KID','OmniFID'))+' |')
    incompatible=[r for r in camera if r['comparison_scope']!='layout_n89' and r['status']=='incompatible_provenance']
    if incompatible:text+=['','Provenance-incompatible comparisons were excluded: '+json.dumps(incompatible,sort_keys=True)]
    text+=['','Per-prompt values and paired changes are in camera_comparisons.csv. No significance or generalization claim follows from three prompts.','',
        '## D. Metric availability and limitations','']
    for metric,s in statuses.items():text.append(f'- {metric}: {s["status"]}; {s["computed_count"]} scored images/groups. '+(s['blocker'] or ''))
    text+=['','Distribution scores use real references only. No SphereDiff image is a real-reference sample. SUN360 and the 21 heterogeneous custom prompts have content/domain mismatch; any distribution score is exploratory. Rank deficiency and effective panorama/view counts are retained. No overall combined quality score.','',
        '## E. Efficiency and provenance','',
        'Inventory entries retain pinned checkpoint IDs/revisions, schedule hashes, source records, scheduler accounting, runtime, memory and call counts where recorded. Missing historical measurements remain null; they are not inferred from step counts. checkpoint_weight_audit.json confirms all 21 shared loaded Diffusers weight/config files are byte-identical across the FLUX snapshots. Each new control preserves its own historical checkpoint source. Efficiency tables below report available measurements; equal steps do not imply equal compute.','',
        '## Resume','',
        '`python3 -m studies.panorama_metrics.launch` submits a bounded update if no evaluator is active. The batch updater rescans completed pairs, skips fresh content/protocol/checkpoint caches, and atomically refreshes these reports.','',
        'Primary sources and pinned revisions, preprocessing and explicit adaptations are in metric_protocols.json. Historical generation folders are read-only to evaluation.']
    text+=['','Validation records: validation/cpu.json and validation/gpu.json; numerical FID/KID fixtures are checked against pinned official implementations. Frozen evaluator: validation/frozen.json. Primary sources: [OmniFID/DS](https://arxiv.org/html/2407.18207), [JoPano](https://arxiv.org/html/2512.06885v1), [PanFusion FAED](https://github.com/chengzhag/PanFusion), [Distort-FID](https://github.com/iSEE-Laboratory/PanoDecouple), [torch-fidelity](https://github.com/toshas/torch-fidelity), [CLIP](https://github.com/openai/CLIP).']
    efficiencies=[]
    for r in rows:
        if r['completion_status']!='complete':continue
        e=r.get('efficiency',{});counts=e.get('counts') or {};vae=e.get('vae_calls') or {}
        if not isinstance(vae,dict):vae={}
        shapes=e.get('model_input_shapes',e.get('actual_model_input_shapes')) or {}
        if isinstance(shapes,dict):shape_calls=sum(v for v in shapes.values() if isinstance(v,int))
        else:shape_calls=None
        generation=e.get('generation_seconds')
        measured=e.get('runtime_seconds')
        x=dict(id=r['id'],group=key(r),prompt_id=r['prompt_id'],generation_seconds=generation,
            historical_runtime_seconds=measured,peak_allocated_gib=e.get('peak_allocated_gib',e.get('peak_gpu_allocated_gib')),
            denoiser_calls=counts.get('denoiser',e.get('actual_transformer_forward_invocations',shape_calls)),
            vae_encode_calls=counts.get('encode',vae.get('encode')),vae_decode_calls=counts.get('decode',vae.get('decode')),
            total_seconds=e.get('total_seconds'),model_loading_seconds=e.get('model_loading_seconds',e.get('model_loading_and_conditioning_seconds')),
            denoising_and_fusion_seconds=e.get('denoising_and_fusion_seconds'),pipeline_seconds=e.get('pipeline_seconds'),
            host_peak_rss_gib=e.get('host_peak_rss_gib'),model_input_shapes=json.dumps(shapes,sort_keys=True),measurement_note='Historical runtime_seconds may include initialization; it is not relabeled generation-only.')
        efficiencies.append(x)
    efields=['id','group','prompt_id','generation_seconds','historical_runtime_seconds','peak_allocated_gib','denoiser_calls','vae_encode_calls','vae_decode_calls','model_input_shapes','total_seconds','model_loading_seconds','denoising_and_fusion_seconds','pipeline_seconds','host_peak_rss_gib','measurement_note']
    csv_atomic(ROOT/'efficiency.csv',efficiencies,efields)
    text+=['','## Available measured efficiency','','Generation-only time uses only explicitly recorded generation_seconds. Different nodes and local input sizes also affect runtime. Missing measurements remain unavailable. Per-image counts and input shapes are in efficiency.csv.','','| Group | Timed N | Mean generation seconds | Mean peak allocated GiB | Mean denoiser calls | Mean VAE encodes | Mean VAE decodes |','|---|---:|---:|---:|---:|---:|---:|']
    for name in sorted(groups):
        er=[r for r in efficiencies if r['group']==name]
        vals=[]
        for metric in ('generation_seconds','peak_allocated_gib','denoiser_calls','vae_encode_calls','vae_decode_calls'):
            v=[r[metric] for r in er if isinstance(r.get(metric),(int,float))];vals.append(float(np.mean(v)) if v else None)
        text.append('| '+name+' | '+str(sum(r['generation_seconds'] is not None for r in er))+' | '+' | '.join(fmt(v) for v in vals)+' |')
    path=ROOT/'report.md';tmp=path.with_suffix('.md.tmp');tmp.write_text('\n'.join(text)+'\n');os.replace(tmp,path)
    atomic(ROOT/'inventory.json',dict(created=now(),expected=len(rows),counts=dict(collections.Counter(r['completion_status'] for r in rows)),rows=rows))
    print('REPORT_UPDATED',ROOT,'scored',sum(r['evaluation_status']=='computed_partial' for r in scored),flush=True)
    return group_table
if __name__=='__main__':update()
