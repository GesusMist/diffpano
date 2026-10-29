"""One recorded infrastructure retry; original failed jobs remain in journal."""
from .common import *
from .resources import submit,reconcile
from .api_launch import launch as launch_api

def main():
    reconcile(RUN)
    write(RUN/'environment_retry.json',dict(reason='Inherited evaluation virtualenv prevented activate_venv from locating libpython3.11 after module purge',
        failed_jobs=['3220222','3220223','3220224'],scientific_code_changed=False,repair='clean_environment.sh strips inherited virtualenv PATH before activating modules',utc=now()))
    for component in ('cpu','gpu','views'):
        gpu=component=='gpu'
        result=submit(RUN,component+'-v2',ROOT/('studies/paper_eval_v2/gpu.slurm' if gpu else 'studies/paper_eval_v2/cpu.slurm'),
            role='evaluation' if gpu else 'cpu',script_args=('--run',str(RUN),'evaluate','--group','views' if component=='views' else 'group1','--stage','gpu' if gpu else 'cpu'),
            reserve_su=584 if gpu else 8)
        print(component,result['job_id'],flush=True)
    print('pilot',launch_api(RUN)['job_id'],flush=True)

if __name__=='__main__':main()
