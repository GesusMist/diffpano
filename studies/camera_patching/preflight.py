"""Real native initialization and one ordinary synchronized interval per layout/projection."""
import argparse
import gc
from .common import *

def run(backend):
    import torch
    from .runtime import setup,group,initialize,count_calls
    v=validation_gate();before=fingerprint()
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    hashes={};checks=[]
    with effective_prompt('ruins') as (_,path):
        c,b,old,table,bank=setup(backend,path)
        for strategy in v['allowed_strategies']:
            seen=[]
            for projection in ('erp','cea'):
                row=next(r for r in rows() if (r['backend'],r['strategy'],r['projection'],r['prompt'])==(backend,strategy,projection,'ruins'))
                p,conditions,ids,provenance,route=group(row,c,b,old,bank)
                states,record=initialize(b,backend,p.cameras,ids,reverse=projection=='cea')
                seen.append(record['initial_local_sha256'])
                with torch.no_grad(),count_calls(b,backend) as counts:
                    next_states,summary=p.advance_interval(states,conditions,table[0],pass_kind='initial')
                assert counts==expected_counts(len(p.cameras),1,backend=='pixeldit',terminal=False)
                assert summary['pass_kind']=='initial' and summary['min_contributors']>0 and summary['minimum_weight_sum']>0
                assert len(next_states)==len(states) and all(torch.isfinite(x).all() for x in next_states)
                assert summary['current_state_error_max']<1e-4
                assert all(not x.eligible for x in table)
                checks.append(dict(strategy=strategy,projection=projection,counts=counts,summary=summary))
                emit('INTERVAL_PREFLIGHT_PASSED',backend=backend,strategy=strategy,projection=projection,counts=counts,summary=summary)
                p.canvas.clear_cache();del p,states,next_states,conditions;gc.collect();torch.cuda.empty_cache()
            assert seen[0]==seen[1],'ERP/CEA deterministic initialization mismatch'
            hashes[strategy]=seen[0]
        del bank,b
    gc.collect();torch.cuda.empty_cache()
    assert fingerprint()==before
    emit('CAMERA_PREFLIGHT_PASSED',fingerprint=before,job=os.environ['SLURM_JOB_ID'],backend=backend,
        initialization_hashes=hashes,checks=checks,validation_job=v['job'],time_travel=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=BACKENDS);a=p.parse_args()
    for backend in ([a.backend] if a.backend else BACKENDS):run(backend)
