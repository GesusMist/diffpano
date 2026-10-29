"""Finite private API jobs using the shared scheduler and billing ledgers."""
from .common import *
from .resources import submit


def launch(run,stage='pilot',version=1,dependency=None):
    if stage not in ('pilot','production'):raise ValueError(stage)
    if not (run/'vlm/pilot_protocol.json').exists():raise ValueError('Prepare pilot first')
    if stage=='production' and not read(run/'vlm/pilot_result.json').get('production_ready'):
        raise ValueError('Valid pilot required')
    if stage=='production' and not read(run/'vlm/production_views.json').get('views'):
        raise ValueError('Nonempty paired production view manifest required')
    options=[]
    if dependency:options.append('--dependency=afterany:'+str(dependency))
    if stage=='pilot':options.append('--time=01:00:00')
    return submit(run,'api-'+stage+'-v'+str(version),ROOT/'studies/paper_eval_v2/api.slurm',role='api',
        script_args=(stage,'--run',str(run),'--budget-mode','provider_managed'),options=options,reserve_su=2 if stage=='pilot' else 24)
