"""Descriptive reproducibility of the fixed repeated-view subset, by judge."""
from .common import *


def evaluate(run=RUN):
    from . import vlm,vlm_protocol as vp
    from .api_ledger import Ledger
    run=Path(run);manifest=read(run/'vlm/production_views.json');frozen=read(run/'vlm/pilot_protocol.json')
    ledger=Ledger(run/'api/ledger.sqlite3',mode='provider_managed')
    repeat_keys={(r['image_sha256'],r['view_id']) for r in manifest['repeat_view_identities']}
    rows=[];groups=[]
    for view in manifest['views']:
        if (view['image_sha256'],view['view_id']) not in repeat_keys:continue
        for model in vp.MODELS:
            primary_id,_=vp.cache_identity(view,frozen['judges'][model],frozen)
            repeat_id,_=vp.cache_identity(view,frozen['judges'][model],frozen,replicate_id='repeat-1')
            first,second=ledger.observation(primary_id),ledger.observation(repeat_id)
            for criterion in vp.CRITERIA:
                a=first['scores'][criterion] if first else None;b=second['scores'][criterion] if second else None
                rows.append(dict(judge=model,criterion=criterion,case_id=view['case_id'],prompt_id=view['prompt_id'],
                    view_id=view['view_id'],primary_request_id=primary_id,repeat_request_id=repeat_id,
                    first_score=a,repeat_score=b,difference=None if a is None or b is None else b-a))
    for model in vp.MODELS:
        for criterion in vp.CRITERIA:
            selected=[r for r in rows if r['judge']==model and r['criterion']==criterion];valid=[r for r in selected if r['difference'] is not None]
            groups.append(dict(judge=model,criterion=criterion,planned_views=len(selected),completed_paired_views=len(valid),
                exact_agreement=sum(r['difference']==0 for r in valid)/len(valid) if valid else None,
                mean_absolute_difference=sum(abs(r['difference']) for r in valid)/len(valid) if valid else None,
                mean_difference=sum(r['difference'] for r in valid)/len(valid) if valid else None,
                interpretation='Descriptive fixed-view repeat diagnostic; correlated views are not independent scene observations.'))
    write(run/'vlm/repeat_observations.json',rows);write(run/'vlm/repeat_reliability.json',groups)
    for filename,values in (('repeat_observations',rows),('repeat_reliability',groups)):
        if values:csv_write(run/('vlm/'+filename+'.csv'),values,sorted({k for r in values for k in r}))
    return dict(completed_pairs=sum(r['difference'] is not None for r in rows)//4,planned_pairs=len(rows)//4)

if __name__=='__main__':print(evaluate())
