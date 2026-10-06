"""Read-only integration of the corrected 126-row camera-count study."""
import collections
import concurrent.futures
from .common import *
from studies.camera_count_v2 import common as cc

PROMPTS = ('firework', 'ruins', 'underwater')

def descriptor(row, prompt, layout):
    sphere = row['family'] == 'spherediff_camera_override'
    strategy = row['strategy'] + '_n' + str(row['num_cameras'])
    if row['strategy'] == 'random': strategy += '_seed' + str(layout['accepted_seed'])
    method = 'spherediff_camera_override' if sphere else 'diffpano'
    projection = 'spherical' if sphere else 'erp'
    path = cc.folder(row) / 'final.png'
    return dict(id='/'.join(('camera_count',method,row['backend'],projection,str(row['steps']),strategy,row['prompt'])),
        family='camera_count',study_family=row['family'],memberships=['camera_count'],method=method,
        method_label='SphereDiff — reduced ring cameras' if sphere else 'DiffPano',
        backend=row['backend'],steps=row['steps'],seed=row['seed'],consensus_projection=projection,final_projection='erp',
        prompt_id=row['prompt'],prompt=prompt,original_prompt_sha256=prompt['original_sha256'],
        effective_prompt_sha256=prompt['effective_sha256'],directional_texts=prompt['effective_lines'],
        camera_strategy=strategy,camera_layout=row['strategy'],camera_count=row['num_cameras'],
        layout_seed=layout['accepted_seed'],fibonacci_phase=layout['fibonacci_phase'],
        camera_geometry_sha256=layout['angular_geometry_sha256'],layout_path=str(cc.layout_path(row['strategy'],row['num_cameras'])),
        output_path=str(path),resolved_path=str(path.resolve()),completion_status='missing',evaluation_status='pending',
        provenance_status='unresolved',model_verification_status='unresolved')

def inspect(row, prompt, layout, accounting):
    rec=descriptor(row,prompt,layout)
    gate='spherediff' if row['family']=='spherediff_camera_override' else 'diffpano'
    if not layout['coverage'][gate]['passed']:
        rec.update(completion_status='blocked',evaluation_status='blocked',
            exclusion_reason='Fixed camera layout fails original '+gate+' RGB support/finite-value gate.',
            source_provenance=[rec['layout_path']])
        return rec
    folder=cc.folder(row)
    if not (folder/'final.png').exists():
        status=cc.ROOT/'status'/(str(row['index'])+'.json')
        if status.exists():rec['completion_status']=read(status)['state']
        return rec
    try:
        assert cc.complete(row),'Final PNG/config/durable status does not pass the frozen generation validator'
        m=read(folder/'config.json')
        assert m['prompt']['original_sha256']==prompt['original_sha256']
        assert m['prompt']['effective_sha256']==prompt['effective_sha256']
        assert m['camera']['fov']==[80.,80.] and m['camera']['num_cameras']==row['num_cameras']
        if row['family']=='diffpano':
            model=m['runtime_audit']['model_checkpoint'];revision=m['runtime_audit']['model_revision']
        else:
            model=m['configuration']['model_source'];revision=m['configuration']['revision']
            assert m['label']=='SphereDiff — reduced ring cameras'
        job=str(m['execution']['job_id']);account=accounting.get(job)
        assert account and account['state']=='COMPLETED' and account['exit_code']=='0:0',('Missing scheduler success',job,account)
        rec.update(completion_status='complete',evaluation_status='eligible',provenance_status='verified',
            image_sha256=m['output']['sha256'],width=4096,height=2048,configuration=m,
            model_id=model,model_revision=revision,model_verification_status='pinned frozen generation provenance',
            schedule=m['schedule'],schedule_sha256=m['schedule_sha256'],initialization=m['initialization'],
            efficiency=m['execution'],local_native_resolution=m['backend'].get('local_native'),
            source_provenance=[str(folder/'config.json'),str(cc.ROOT/'status'/(str(row['index'])+'.json')),rec['layout_path']],
            job=job,accounting=account)
    except Exception as e:
        rec.update(completion_status='excluded',evaluation_status='excluded',exclusion_reason=repr(e))
    return rec

def scan(accounting,prompt_map):
    cc.validation_gate()
    layouts={(s,n):read(cc.layout_path(s,n)) for s in cc.STRATEGIES for n in cc.COUNTS}
    requested=cc.rows()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(lambda r:inspect(r,prompt_map[r['prompt']],layouts[r['strategy'],r['num_cameras']],accounting),requested))
    assert len(result)==126 and len({r['id'] for r in result})==126
    assert sum(r['study_family']=='diffpano' for r in result)==108
    assert all(r['camera_layout']=='old' for r in result if r['method']=='spherediff_camera_override')
    return result

def paired_compatible(a,b,scope):
    assert a['prompt_id']==b['prompt_id'] and a['effective_prompt_sha256']==b['effective_prompt_sha256']
    assert a['seed']==b['seed']==0 and a['backend']==b['backend'] and a['steps']==b['steps']
    if scope=='method_at_fixed_count':
        assert a['method']=='spherediff_camera_override' and b['method']=='diffpano'
        assert a['camera_count']==b['camera_count']
        assert a['camera_geometry_sha256']==b['camera_geometry_sha256']
        return True
    assert a['model_revision']==b['model_revision'] and a['model_id']==b['model_id']
    # Metadata formats differ between original89 and the camera-only wrapper;
    # require exact numerical trajectory plus class, not identical bookkeeping.
    if a.get('schedule_sha256')!=b.get('schedule_sha256'):
        sa=a.get('schedule',{});sb=b.get('schedule',{})
        for k in ('class_name','timesteps','sigmas'):
            assert k in sa and k in sb and sa[k]==sb[k],('Schedule mismatch',k,a['id'],b['id'])
    if b['method']=='diffpano':
        ia=a['initialization'];ia=ia.get('record',ia);ib=b['initialization']
        assert ia['source_shape']==ib['source_shape'] and ia['source_sha256']==ib['source_sha256']
    return True

def comparisons(groups,paired_rows,by_id):
    """Matched three-prompt count, layout, and method comparisons; never crops as n."""
    requests=[]
    for name,rr in sorted(groups.items()):
        r=rr[0]
        if r['family']!='camera_count':continue
        if r['method']=='diffpano':
            reference={'old':'old89','fibonacci':'fibonacci_n89','random':'random_n89_seed0'}[r['camera_layout']]
            baseline=f"diffpano/{r['backend']}/erp/{r['steps']}/{reference}"
            requests.append(('count_vs_89',baseline,name))
            if r['camera_layout']!='old':
                requests.append(('layout_at_fixed_count',f"diffpano/{r['backend']}/erp/{r['steps']}/old_n{r['camera_count']}",name))
        else:
            baseline=f"spherediff/{r['backend']}/spherical/{r['steps']}/official_spherediff"
            requests.append(('count_vs_89',baseline,name))
            requests.append(('method_at_fixed_count',name,f"diffpano/{r['backend']}/erp/{r['steps']}/old_n{r['camera_count']}"))
    result=[]
    for scope,a,b in requests:
        aa={r['prompt_id']:r for r in groups.get(a,[]) if r['completion_status']=='complete' and r['prompt_id'] in PROMPTS}
        bb={r['prompt_id']:r for r in groups.get(b,[]) if r['completion_status']=='complete' and r['prompt_id'] in PROMPTS}
        common=sorted(set(aa)&set(bb));reference=groups[b][0]
        base=dict(backend=reference['backend'],projection='erp',comparison_scope=scope,baseline=a,comparison=b,
            common_prompts=';'.join(common),uncertainty='Three prompts and one seed; descriptive only; no confidence interval or significance claim.')
        if not common:
            result.append(dict(base,record_type='summary',status='blocked_no_common_completed_prompts',n=0));continue
        try:
            for prompt in common:paired_compatible(aa[prompt],bb[prompt],scope)
        except (AssertionError,KeyError) as e:
            result.append(dict(base,record_type='summary',status='incompatible_provenance',n=0,common_prompts='',reason=repr(e)));continue
        for x in paired_rows(a,b,common):result.append(dict(base,**{k:v for k,v in x.items() if k not in base},record_type='summary'))
        for prompt in common:
            for metric in PER_IMAGE:
                av=by_id[aa[prompt]['id']][metric];bv=by_id[bb[prompt]['id']][metric]
                result.append(dict(base,record_type='per_prompt',prompt_id=prompt,metric=metric,n=1,
                    baseline_mean=av,comparison_mean=bv,mean=bv-av if av is not None and bv is not None else None,
                    status='paired' if av is not None and bv is not None else 'pending'))
    return result
