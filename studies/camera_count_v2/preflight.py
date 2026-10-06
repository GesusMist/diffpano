"""One real interval at the full production schedule; no scientific PNG."""
import argparse
import traceback
from .common import *

def execute(row):
    gate=validation_gate();before=source_hashes();layout=load_layout(row)
    import torch
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    with lock('preflight-'+group_key(row)):
        if preflight_path(row).exists():
            preflight_gate(row);emit('PREFLIGHT_REUSED',group=group_key(row));return
        runner='diffpano_runner' if row['family']=='diffpano' else 'spherediff_runner'
        module=__import__('studies.camera_count_v2.'+runner,fromlist=['generate'])
        with effective_prompt(row['prompt']) as (prompt,path):
            image,record=module.generate(row,path,preflight=True)
        assert image is None
        assert record['execution']['counts']==expected_counts(row,steps=1,terminal=False)
        assert record['generation']['steps']==row['steps']
        assert_preserved()
        assert source_hashes()==before==gate['source_hashes'], 'Source changed while preflight ran'
        atomic(preflight_path(row),dict(passed=True,row=row,group=group_key(row),record=record,
            layout_hash=layout['angular_geometry_sha256'],geometry_identity=geometry_identity(),
            source_hashes=before,time=now(),job=os.environ['SLURM_JOB_ID']))
        emit('PREFLIGHT_PASSED',group=group_key(row),initialization=record['initialization'],
             counts=record['execution']['counts'])
def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
    row=rows()[a.index]
    try:execute(row)
    except BaseException:
        ledger('PREFLIGHT_FAILED',group=group_key(row),job=os.environ.get('SLURM_JOB_ID'),error=traceback.format_exc())
        raise
if __name__=='__main__':main()
