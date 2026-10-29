"""Resumable entry points; submission, output verification and scoring are distinct."""
import argparse
import json
from pathlib import Path
from .common import *


def inventory(run):
    if (run/'protocol_registry.json').exists():
        value=read(run/'inventory/records.json')
        return dict(state='reused_frozen_development_inventory',records=value if isinstance(value,list) else value['records'],
            confirmation_index='generation/inventory_records.json',reason='A frozen run preserves its initial audit; fresh cases use the separate confirmation index.')
    from .inventory import collect
    update_component(run,'inventory','running')
    result=collect(run)
    for name in ('cohorts.json','inventory.csv','provenance_audit.json','missing_cases.csv'):
        source=run/'inventory'/name
        if source.exists():atomic_text(run/name,source.read_text())
    update_component(run,'inventory','complete',records=len(result['records']),audit=str(run/'inventory/provenance_audit.json'))
    return result


def prepare(run):
    if (run/'protocol_registry.json').exists():
        verify_protocol(run)
        return dict(protocol_sha256=sha(run/'protocol_registry.json'),state='reused_verified_protocol')
    from .rendering import protocol_definitions as render_protocols
    from .metrics import protocol_definitions as metric_protocols
    from .statistics import protocol_definitions as statistics_protocols
    from .distribution import protocol_definitions as distribution_protocols
    from studies.panorama_metrics.protocols import protocols
    if not (run/'inventory/records.json').exists():raise RuntimeError('Run inventory first')
    from .scoring import prepare as prepare_scoring
    prepare_scoring(run)
    combined_metrics=metric_protocols();combined_metrics.update(distribution_protocols())
    registry=dict(scoring_descriptions_sha256=sha(run/'scoring_prompts.json'),schema_version=2,protocols=render_protocols(),metrics=combined_metrics,statistics=statistics_protocols(),
        legacy=protocols(),faed_rgb=protocols()['FAED'],
        primary_methods=list(PRIMARY_METHODS),primary_cohort='development15',confirmation_cohort='confirmation24',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'studies/paper_eval_v2'/n for n in
              ('rendering.py','metrics.py','statistics.py','distribution.py','metric_assets.py')] if p.exists()},
        reporting='Never mix native/common resolutions, real/native references, legacy/v2 scores, or judges; missing values remain missing')
    freeze(run/'protocol_registry.json',registry)
    update_component(run,'protocol_freeze','complete',sha256=sha(run/'protocol_registry.json'))
    return dict(protocol_sha256=sha(run/'protocol_registry.json'))


def verify_protocol(run):
    registry=read(run/'protocol_registry.json')
    for name,expected in registry['source_hashes'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen scientific source changed: '+name)
    if sha(run/'scoring_prompts.json')!=registry['scoring_descriptions_sha256']:
        raise ValueError('Frozen scoring descriptions changed')
    from .rendering import protocol_definitions
    if digest(protocol_definitions())!=digest(registry['protocols']):
        raise ValueError('Frozen rendering runtime or settings changed')
    return registry


def evaluate(run,group,stage):
    registry=verify_protocol(run)
    result=_evaluate(run,group,stage,registry)
    verify_protocol(run)
    summary={key:(len(value) if isinstance(value,list) else value) for key,value in result.items()}
    update_component(run,group+'_'+stage,'complete',result_summary=summary)
    return result


def _evaluate(run,group,stage,registry):
    from .scoring import evaluation_records
    records=evaluation_records(run)
    cohorts=read(run/'inventory/cohorts.json')
    if group=='group1':
        from .metrics import evaluate as execute
        return execute(run,records,registry,stage=stage,cohorts=cohorts)
    if group=='group2':
        from .distribution import evaluate as execute
        return execute(run,records,registry)
    if group=='views':
        from .rendering import cache_views
        chosen={x['case_id'] for x in cohorts['selected_cells'] if x['method_id'] in PRIMARY_METHODS+('spherope',)}
        views=cache_views(run,[r for r in records if r['case_id'] in chosen and r['state']=='complete'])
        write(run/'vlm/views.json',dict(views=views,registry_sha256=sha(run/'protocol_registry.json')))
        return dict(view_count=len(views))
    raise ValueError(group)


def status(run):
    from .resources import reconcile
    scheduler=reconcile(run)
    value=read(run/'status.json')
    value['scheduler']=scheduler
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=RUN)
    subs=parser.add_subparsers(dest='action')
    subs.add_parser('inventory');subs.add_parser('prepare');subs.add_parser('status');subs.add_parser('report')
    p=subs.add_parser('evaluate');p.add_argument('--group',choices=['group1','group2','views'],default='group1');p.add_argument('--stage',choices=['cpu','gpu','aggregate'],default='cpu')
    p=subs.add_parser('launch');p.add_argument('--component',choices=['inventory','cpu','gpu','views','native','confirmation','api'],required=True)
    args=parser.parse_args(argv);run=args.run
    if args.action=='inventory':result=inventory(run)
    elif args.action=='prepare':result=prepare(run)
    elif args.action=='evaluate':result=evaluate(run,args.group,args.stage)
    elif args.action=='status':result=status(run)
    elif args.action=='report':
        from .report import report
        result=report(run)
    elif args.action=='launch':
        from .launch import launch
        result=launch(run,args.component)
    else:parser.print_help();return
    if isinstance(result,dict):print(json.dumps({k:(len(v) if isinstance(v,list) else v) for k,v in result.items() if k not in ('records','figures')},indent=2,default=str))

if __name__=='__main__':main()
