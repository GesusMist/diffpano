"""One fresh GPU process per scientific sample."""
import argparse
import traceback
from .common import *

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',type=int)
    p.add_argument('--strategy',choices=STRATEGIES);p.add_argument('--camera-count',type=int,default=89)
    p.add_argument('--layout-seed',type=int);p.add_argument('--backend',choices=BACKENDS)
    p.add_argument('--projection',choices=('erp','cea'));p.add_argument('--prompt',choices=PROMPTS)
    a=p.parse_args()
    assert a.camera_count==89,'Only N=89 scientific runs authorized'
    if a.index is not None:row=rows()[a.index]
    else:
        assert a.layout_seed in (None,0)
        row=next(r for r in rows() if (r['strategy'],r['backend'],r['projection'],r['prompt'])==(a.strategy,a.backend,a.projection,a.prompt))
    v=validation_gate();assert row['strategy'] in v['allowed_strategies'],v['blocked_strategies']
    pf=preflight_gate(row['backend']);source=source_record()
    with lock('camera-patching-'+digest(row)):
        if complete(row,verbose=True):emit('SKIP_COMPLETE',row=row);return
        import torch
        assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
        emit('WORKER_START',row=row,source=source,validation_job=v['job'],preflight_job=pf['job'])
        try:
            from .runtime import generate
            with effective_prompt(row['prompt']) as (_,path):image,record=generate(row,path,source,pf)
            assert fingerprint()==source['fingerprint']
            publish(image,record,row)
            emit('SUCCESS_CASE',row=row,output_sha256=record['output']['sha256'])
        except BaseException:
            emit('FAILED_CASE',row=row,job=os.environ['SLURM_JOB_ID'],error=traceback.format_exc());raise

if __name__=='__main__':main()
