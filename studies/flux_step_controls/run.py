import argparse
import traceback
from .common import *

def main(index=None,preflight=None):
    import torch
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    m=manifest();assert_preserved(m)
    cpu=read(ROOT/'validation/cpu.json');assert cpu['passed'] and cpu['manifest_sha256']==sha(ROOT/'manifest.json')
    row=next(r for r in rows() if r['method']==preflight and r['prompt']=='ruins') if preflight else rows()[index]
    status=ROOT/'status'/('preflight-'+preflight+'.json' if preflight else str(index)+'.json')
    with lock('flux-step-'+('preflight-'+preflight if preflight else str(index))):
        if not preflight and complete(row):emit('SKIP_COMPLETE',row=row);return
        if not preflight:preflight_gate(row['method'])
        atomic(status,dict(state='running',job=os.environ['SLURM_JOB_ID'],row=row))
        try:
            from .runtime import diffpano,original
            with effective_prompt(row['prompt']) as (prompt,path):
                image,record=(diffpano if row['method']=='diffpano' else original)(row,path,preflight=bool(preflight))
            assert_preserved(m)
            record.update(prompt=prompt,source=dict(study_sources=source_hashes(),protected_sources=m['protected_sources'],
                git_commit=m['git_commit']))
            record['execution'].update(job=os.environ['SLURM_JOB_ID'],array_job=os.environ.get('SLURM_ARRAY_JOB_ID'),
                array_task=os.environ.get('SLURM_ARRAY_TASK_ID'),gpu=torch.cuda.get_device_name(),torch=torch.__version__)
            if preflight:
                atomic(ROOT/'validation'/('preflight-'+preflight+'.json'),dict(passed=True,record=record,
                    manifest_sha256=sha(ROOT/'manifest.json'),study_sources=source_hashes(),job=os.environ['SLURM_JOB_ID']))
            else:publish(row,image,record)
            atomic(status,dict(state='complete',job=os.environ['SLURM_JOB_ID'],row=row,
                config_sha256=None if preflight else sha(row['config'])))
            emit('CONTROL_SUCCESS',row=row,preflight=bool(preflight),record=record)
        except BaseException:
            atomic(status,dict(state='failed',job=os.environ['SLURM_JOB_ID'],row=row,error=traceback.format_exc()))
            raise

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--index',type=int);g.add_argument('--preflight',choices=('spherediff','diffpano'))
    a=p.parse_args();main(a.index,a.preflight)
