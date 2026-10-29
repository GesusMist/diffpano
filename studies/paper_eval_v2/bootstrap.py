"""Initialize durable run intent before expensive work; idempotent on resume."""
from .common import *


def initialize(run=RUN):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    request = ROOT/'docs/DiffPano_Codex_Evaluation_Prompts_20261026.txt'
    config = dict(run_id=run.name, selected_generation_tier='core',
        nominal_erp_slots=dict(core=72,seeds=24,trained=24,generality=32,full=152),
        native_reference_slots=240, primary_judge='openai/gpt-4o', secondary_judge='openai/gpt-6-sol',
        api_budget_mode='provider_managed', local_api_budget_usd=None, local_api_max_requests=None,
        api_budget_authorization='User explicitly instructed no local limit; limit is already set in OpenRouter.',
        generation_gpu_limit=2,evaluation_gpu_limit=1,api_concurrency_limit=2,
        request_path=str(request),request_sha256=sha(request))
    freeze(run/'run_config.json',config)
    if not (run/'status.json').exists():
        write(run/'status.json',dict(run_id=run.name,created_utc=now(),updated_utc=now(),
            complete=False,components={name:dict(state='pending') for name in
            ('inventory','protocol_freeze','group1','native_references','real_dataset','captions',
             'group2','vlm_discovery','vlm_calibration','vlm_production','final_method',
             'confirmation_generation','report','human_ratings')},
            scheduler_jobs={},next_actions=['Finish inventory and protocol freeze','Prepare independent metrics, native bank and API pilots']))
    if not (run/'environment.json').exists():
        write(run/'environment.json',dict(created_utc=now(),root=str(ROOT),output_symlink=str((ROOT/'outputs').resolve()),
             run_storage=str(run.resolve()),head=command(['git','rev-parse','HEAD']),
             local_modifications=command(['git','status','--short']),host=command(['hostname']),
             generation_environment=str(ROOT/'studies/all_prompts/environment.sh'),
             evaluation_environment=str(ROOT/'studies/panorama_metrics/environment.sh'),
             account=ACCOUNT,initial_balance_su=7353.35,initial_active_jobs=[],
             storage_decision='New run uses home-backed outputs/paper-eval-v2 symlink because scratch has only 2118 free inodes; all legacy paths unchanged.',
             credentials='Private file verified present, directory0700/file0600; secret excluded from all records.'))
    dependencies = {
      'inventory':[], 'protocol_freeze':['inventory'], 'group1':['protocol_freeze'],
      'native_references':['protocol_freeze'], 'real_dataset':[], 'captions':['real_dataset'],
      'group2':['protocol_freeze','real_dataset'], 'vlm_discovery':[],
      'vlm_calibration':['vlm_discovery','protocol_freeze'], 'vlm_production':['vlm_calibration'],
      'final_method':['inventory'], 'confirmation_generation':['captions','final_method'],
      'report':['inventory'], 'human_materials':['real_dataset'], 'human_ratings':['human_materials']}
    freeze(run/'dependency_graph.json',dependencies)
    if not (run/'resource_plan.json').exists():
        write(run/'resource_plan.json',dict(account=ACCOUNT,selected_tier='core',
           caps=dict(generation_gpus=2,evaluation_gpus=1,api_concurrency=2),
           api_caps='Provider-managed by explicit user override; maintain a cumulative cost/attempt ledger.',
           balance_su=7353.35,generation_rate_su_per_gpu_hour=146.,
           cpu_rate_su_per_core_hour=1.,native_reference_measured_seconds_per_image=31.32779559120536,
           native_reference_expected_images=240,native_reference_nominal_gpu_hours=240*31.32779559120536/3600,
           confirmation_erp_slots=72,extensions=dict(seeds='not_selected',trained='not_selected',generality='not_selected'),
           reservation_policy='One locked scheduler journal; finite groups; no duplicate/ambiguous submissions; recheck live balance and all user GPU allocations before each submission.',
           storage=dict(home_available_gb=9.7,home_free_inodes_approx=5008,scratch_free_inodes_at_start=2118,
                        cache_policy='Pack viewport and raw-response caches; do not emit one file per score/crop.'),
           state='preparing_exact_case_manifest'))
    return config


if __name__ == '__main__':
    initialize()
