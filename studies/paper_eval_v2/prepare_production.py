"""Prepare views against the already frozen pilot, using canonical JSON equality.

JSON reload turns renderer tuples into lists. The original preparation function
compares Python containers directly; this adapter preserves the exact frozen
scientific protocol and existing request identities without rewriting it.
"""
import argparse
from decimal import Decimal
from .common import RUN,Path,read,write,sha,now
from . import vlm,vlm_protocol as vp


def prepare(run=RUN,views_path=None):
    from .rendering import protocol_definitions
    from .cli import verify_protocol
    run=Path(run);verify_protocol(run)
    folder=run/'vlm';frozen=read(folder/'pilot_protocol.json')
    discovery=max((folder/'discovery').glob('discovery-*.json'),key=lambda path:read(path)['observed_utc'])
    proposed=vp.protocol(read(discovery),protocol_definitions()['common_v2'])
    if vp.digest(proposed)!=vp.digest(frozen):raise ValueError('Frozen VLM protocol truly changed')
    value=read(views_path or folder/'views.json')
    views=value['views'] if isinstance(value,dict) else value
    views,cohorts=vlm.select_cohort_views(views,read(run/'cohorts.json'))
    if not views:raise ValueError('No complete paired production views are available')
    repeats=[v for v in views if int(vp.digest({'image':v['image_sha256'],'seed':20261026})[:8],16)<int(.05*2**32)]
    manifest=dict(views=views,cohort_prompt_ids=sorted({v['prompt_id'] for v in views}),cohorts=cohorts,
        repeat_view_identities=[dict(image_sha256=v['image_sha256'],view_id=v['view_id']) for v in repeats],
        selection_before_scores=True,registry_sha256=vp.digest(read(run/'protocol_registry.json')))
    previous=folder/'production_views.json'
    if previous.exists():
        old=read(previous)
        write(folder/'manifest_history'/('views-'+vp.digest(old)+'.json'),old)
    write(previous,manifest)
    requests=vlm.request_records(views,frozen)
    result=dict(state='production_prepared',paired_prompts=len(manifest['cohort_prompt_ids']),
        primary_unique_requests=len(requests),preselected_repeat_images=len(repeats),
        production_max_usd_before_repeats_retries=str(sum((Decimal(r['max_cost_usd']) for r in requests),Decimal(0))),
        protocol_sha256=frozen['protocol_sha256'],frozen_protocol_file_sha256=sha(folder/'pilot_protocol.json'),
        canonical_protocol_equality_verified=True,adapter_source_sha256=sha(__file__),prepared_utc=now(),
        cohorts=[{k:v for k,v in c.items() if k!='case_ids'} for c in cohorts])
    write(folder/'preparation.json',result)
    old_result=folder/'production_result.json'
    if old_result.exists() and read(old_result).get('planned_requests')==0:
        write(folder/'manifest_history/empty-production-preflight.json',read(old_result))
        write(old_result,dict(stage='production',state='pending',planned_requests=len(requests),valid_requests=0,
            reason='Previous empty preflight made zero API calls; nonempty manifest now prepared'))
    return {k:v for k,v in result.items() if k!='cohorts'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=RUN);p.add_argument('--views',type=Path);a=p.parse_args()
    with vlm.worker_lock(a.run):
        print(prepare(a.run,a.views))

if __name__=='__main__':main()
