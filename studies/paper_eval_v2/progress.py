"""Reconcile component state with published manifests and verified worker records."""
import sqlite3
from .common import *


def refresh(run=RUN):
    run=Path(run)
    def saved(name,default=None):
        path=run/name
        return read(path) if path.exists() else ({} if default is None else default)
    updates={}
    dataset=saved('real_dataset_manifest.json')
    if dataset.get('state')=='complete':
        updates['real_dataset']=dict(state='complete',scenes=dataset['scene_count'],manifest_sha256=sha(run/'real_dataset_manifest.json'))
    captions=saved('captions.json')
    if captions.get('state')=='complete' and captions.get('reviewed'):
        updates['captions']=dict(state='complete',reviewed=True,scenes=len(captions.get('rows',[])),manifest_sha256=sha(run/'captions.json'))
    final=saved('generation/final_method.json')
    if final:
        updates['final_method']=dict(state='complete',method_id=final['method_id'],selection=final['selection'],refinement_included=False)
    if (run/'vlm/pilot_protocol.json').exists():
        updates['vlm_discovery']=dict(state='complete',protocol_sha256=saved('vlm/pilot_protocol.json')['protocol_sha256'])
    pilot=saved('vlm/pilot_result.json')
    if pilot:
        updates['vlm_calibration']=dict(state='complete' if pilot.get('production_ready') else 'partial',
            valid_requests=pilot.get('valid_requests',0),planned_requests=pilot.get('planned_requests',30),
            sensitivity='Weak controlled-warp sensitivity; see calibration_review.json. Schema validity is not geometric validation.')
    controller=saved('controller/state.json')
    generated=controller.get('verified_generation_count',0)
    updates['confirmation_generation']=dict(state='complete' if generated==72 else 'running' if controller.get('state')=='running' else 'pending',
        verified_complete=generated,expected=72,controller_state=controller.get('state'),blockers=controller.get('blockers',{}))
    metric=saved('group1/status.json')
    if metric:
        available=min(metric.get(k,0) for k in ('common_v2_DS','common_v2_CS','MUSIQ'))
        updates['group1']=dict(state='complete' if available>=489 and generated==72 else 'partial',
            development_state='complete' if available>=417 else 'partial',scored_available=available,development_selected=417,
            execution_failures=metric.get('failures',0),confirmation_expected=72)
    group2=saved('group2/status.json')
    if group2:
        comparisons=group2.get('paired_comparisons',[])
        updates['group2']=dict(state=group2.get('state','partial'),groups=len(group2.get('groups',[])),
            paired_contrasts=len(comparisons),complete_uncertainty=sum(row.get('state')=='complete' for row in comparisons),
            pending_uncertainty=sum(row.get('state')=='pending_uncertainty_compute' for row in comparisons),
            blockers=group2.get('blockers',[]))
    production=saved('vlm/production_result.json')
    scheduler=saved('scheduler.json')
    live_ids={row['job_id'].split('_')[0] for row in scheduler.get('queue_snapshot',[])}
    api_active=any(job.get('role')=='api' and job['job_id'] in live_ids for job in scheduler.get('jobs',{}).values())
    production_done=production.get('state')=='complete' and production.get('valid_requests')==production.get('planned_requests')
    production_full=production_done and len(saved('vlm/production_views.json').get('cohort_prompt_ids',[]))==48
    database=run/'api/ledger.sqlite3'
    if database.exists():
        with sqlite3.connect(str(database)) as connection:
            ledger=json.loads(connection.execute('SELECT value FROM ledger WHERE id=1').fetchone()[0])
        attempts=list(ledger['attempts'].values())
        selected=[r for r in attempts if r.get('stage')=='production']
        updates['vlm_production']=dict(state='running' if api_active else 'complete' if production_full else 'blocked' if production.get('state')=='blocked' else 'partial' if selected else 'pending',
            development_state='complete' if production_done else 'partial' if selected else 'pending',
            current_manifest_planned=production.get('planned_requests'),current_manifest_valid=production.get('valid_requests'),
            attempted_requests=len(selected),valid_attempts=sum(r['state']=='valid' for r in selected),
            last_worker_state=production.get('state'),blocker=production.get('blocker'),budget_mode='provider_managed')
    with locked_state(run/'status.json') as status:
        for name,value in updates.items():status.setdefault('components',{})[name]=dict(value,updated_utc=now())
        status['updated_utc']=now()
    return updates

if __name__=='__main__':print(refresh())
