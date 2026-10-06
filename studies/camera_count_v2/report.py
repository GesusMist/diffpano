"""Honest completion and blocked-configuration accounting for 126 rows."""
import argparse
import collections
from .common import *

def events():
    p=ROOT/'execution.jsonl'
    if not p.exists():return []
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def layout_status(row):
    p=layout_path(row['strategy'],row['num_cameras'])
    if not p.exists():return 'pending','layout selection/coverage pending'
    d=read(p)
    if d['geometry_identity']!=geometry_identity():return 'blocked','geometry protocol mismatch'
    if not d['coverage']['geometric']['passed']:return 'blocked','fixed layout fails geometric coverage'
    key='spherediff' if row['family']=='spherediff_camera_override' else 'diffpano'
    if not d['coverage'].get(key,{}).get('passed'):return 'blocked',key+' effective RGB support/finite-value gate failed'
    return 'ready',''
def report(write=True,verify_images=True):
    detail=[];counts=collections.Counter()
    dispatcher=read(ROOT/'dispatcher-status.json') if (ROOT/'dispatcher-status.json').exists() else {}
    task_reasons={v['task']:v['reason'] for v in dispatcher.get('tasks',[])}
    for row in rows():
        state,reason=layout_status(row)
        if complete(row):
            state='completed';reason=''
        elif state!='blocked':
            p=ROOT/'status'/(str(row['index'])+'.json')
            status=read(p) if p.exists() else {}
            state='failed' if status.get('state')=='failed' else 'pending'
            reason=status.get('error','')[-1500:] if state=='failed' else reason
        if state=='pending':
            keys=['geometry/'+layout_key(row['strategy'],row['num_cameras']),'preflight/'+group_key(row)]
            pilot=next(r for r in rows() if group_key(r)==group_key(row) and r['prompt']=='firework')
            if row['prompt']!='firework':keys.append('run/'+str(pilot['index']))
            failures=[task_reasons.get(k,'') for k in keys if task_reasons.get(k,'').startswith('failed prerequisite/task')]
            if failures:state='blocked';reason='; '.join(failures)
        counts[state]+=1;detail.append(dict(row=row,state=state,reason=reason))
    efficiency={}
    for item in detail:
        if item['state']!='completed':continue
        row=item['row'];m=read(folder(row)/'config.json');ex=m['execution']
        key=f"{row['family']}/{row['backend']}/n{row['num_cameras']}"
        g=efficiency.setdefault(key,dict(completed=0,total_seconds=[],peak_gpu_allocated_gib=[],host_peak_rss_gib=[],call_counts=[],latent_utilization=[]))
        g['completed']+=1
        for field in ('total_seconds','peak_gpu_allocated_gib','host_peak_rss_gib'):g[field].append(ex[field])
        g['call_counts'].append(ex['counts'])
        if 'latent_utilization' in m:
            v=m['latent_utilization'];g['latent_utilization'].append({k:v[k] for k in ('spherical_points','selected_index_union_count','unused_spherical_points','all_decode_indices_updated')})
    for group in efficiency.values():
        for field in ('total_seconds','peak_gpu_allocated_gib','host_peak_rss_gib'):
            values=group[field];group[field+'_mean']=sum(values)/len(values);group[field+'_max']=max(values)
    layouts={}
    for strategy in STRATEGIES:
        for n in COUNTS:
            p=layout_path(strategy,n)
            if p.exists():
                d=read(p);layouts[layout_key(strategy,n)]=dict(angular_hash=d['angular_geometry_sha256'],
                    accepted_seed=d['accepted_seed'],random_rejections=d.get('random_search',{}).get('number_rejected'),
                    coverage={k:v['passed'] for k,v in d['coverage'].items()},ring_counts=d['ring_counts'])
    result=dict(time=now(),expected=126,completed=counts['completed'],failed=counts['failed'],
        blocked=counts['blocked'],pending=counts['pending'],layouts=layouts,rows=detail,
        scratch_destination=str(ROOT.resolve()),source_fingerprint=fingerprint(),runtime_by_family_backend_count=efficiency,
        code_files_added=list(source_hashes()),job_ids=[e['job'] for e in events() if e['event']=='SUBMITTED'])
    if write:
        atomic(ROOT/'completion.json',result)
        atomic(ROOT/'layouts/random_search_progress.json',{
            str(n):read(ROOT/'layouts'/('random_search_n'+str(n)+'.json'))
            for n in COUNTS if (ROOT/'layouts'/('random_search_n'+str(n)+'.json')).exists()})
        status=f"\n\nCurrent status ({result['time']}): {result['completed']} completed, {result['failed']} failed, {result['blocked']} blocked, {result['pending']} pending. Expected 126.\n\n"
        status+='Accepted random seeds: '+', '.join(f"N{n}: {layouts.get('random_n'+str(n),{}).get('accepted_seed','pending')}" for n in COUNTS)+'.\n'
        (ROOT/'README.md').write_text((STUDY/'README.md').read_text()+status)
        for family,text in [
            ('diffpano','108 ERP-only results. GWTFlow, local native states, center-weighted arithmetic RGB consensus, local VAE residual bridge, current-state-preserving transitions. No time travel.'),
            ('spherediff_camera_override','18 requested old/ring results only, labeled SphereDiff — reduced ring cameras. Original spherical SANA20/FLUX20 with camera-only override. Not untouched original89 baselines. No GWTFlow, RGB bridge, CEA, or time travel.')]:
            p=ROOT/family;p.mkdir(parents=True,exist_ok=True);(p/'README.md').write_text(text+'\n')
    emit('STUDY_REPORT',**{k:v for k,v in result.items() if k not in ('rows','layouts')},layouts=layouts)
    return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args();report()
