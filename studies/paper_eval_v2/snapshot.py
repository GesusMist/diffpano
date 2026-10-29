"""Read saved observations and refresh a partial report without paid requests."""
import argparse
import os
from .common import *


def refresh(run=RUN):
    run=Path(run).absolute()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Report refresh requires a CPU allocation')
    from .native_index import refresh as native_index
    from .vlm import aggregate
    from .repeat_analysis import evaluate as repeats
    from .report import report
    counts=native_index(run)
    update_component(run,'native_references','complete' if sum(counts.values())==240 else 'running',verified_counts=counts,expected=240)
    if (run/'vlm/production_views.json').exists():
        aggregate(run)
        repeats(run)
    from .progress import refresh as progress
    progress(run)
    return report(run)


def launch(run=RUN):
    from .resources import reconcile,submit
    from .controller import active_jobs
    run=Path(run).absolute()
    with lock(run/'report/snapshot-launch.lock'):
        scheduler=reconcile(run)
        active={key:value for key,value in active_jobs(scheduler).items()
                if any(token in key for token in ('report','finalize','snapshot'))}
        if active:return dict(state='already_pending_or_active',jobs=active)
        key='report-snapshot-'+digest(dict(time=now(),jobs=sorted(scheduler.get('jobs',{}))))[:16]
        return submit(run,key,ROOT/'studies/paper_eval_v2/snapshot.slurm',role='cpu',
            script_args=('--run',str(run)),reserve_su=4*40/60)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=RUN);p.add_argument('--launch',action='store_true');a=p.parse_args()
    print(launch(a.run) if a.launch else refresh(a.run))

if __name__=='__main__':main()
