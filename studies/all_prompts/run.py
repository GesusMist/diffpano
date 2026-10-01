"""Fresh-process worker. The sole persistent result is an atomic final.png."""
import argparse
import hashlib
import os
import platform
import resource
import time
import traceback
from studies.all_prompts.common import *

def main(index,validation_job,code_hash,prompt_hash):
    require_validation(validation_job,code_hash,prompt_hash)
    row=rows()[index];target=Path(row['output'])
    with lock('case-'+hashlib.sha256(str(target).encode()).hexdigest()):
        if image_valid(target,True):
            emit('SKIP_VALID',row=row)
            emit('SUCCESS_CASE',row=row,image_sha256=sha(target),width=4096,height=2048,reused_existing_valid=True)
            return
        import torch
        import importlib.metadata
        if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('GPU Slurm worker required')
        started=time.perf_counter()
        emit('WORKER_START',row=row,seed=0,batch_size=1,job=os.environ['SLURM_JOB_ID'],
            array_job=os.environ.get('SLURM_ARRAY_JOB_ID'),array_task=os.environ.get('SLURM_ARRAY_TASK_ID'),
            node=platform.node(),gpu=torch.cuda.get_device_name(),python=platform.python_version(),torch=torch.__version__,
            cuda=torch.version.cuda,diffusers=importlib.metadata.version('diffusers'),
            transformers=importlib.metadata.version('transformers'),source_fingerprint=code_hash,prompt_inventory_sha256=prompt_hash)
        try:
            with effective_prompt(row['prompt']) as (record,prompt_path):
                from studies.all_prompts.runtime import diffpano,original
                image=(diffpano if row['method']=='diffpano' else original)(row,prompt_path)
                assert fingerprint()==code_hash and inventory_hash()==prompt_hash
                # All scientific checks happen before this single final commit.
                publish(image,target)
            emit('SUCCESS_CASE',row=row,image_sha256=sha(target),width=4096,height=2048,
                total_worker_seconds=time.perf_counter()-started,
                host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        except BaseException:
            emit('FAILED_CASE',row=row,job=os.environ['SLURM_JOB_ID'],error=traceback.format_exc())
            raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True)
    p.add_argument('--validation-job',required=True);p.add_argument('--code-hash',required=True);p.add_argument('--prompt-hash',required=True)
    a=p.parse_args();main(a.index,a.validation_job,a.code_hash,a.prompt_hash)
