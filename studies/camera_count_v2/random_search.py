"""Resumable, geometry-only first-passing-seed rejection search for DiffPano."""
import argparse
import os
import time
from .common import *
from .cameras import build,layout_record
from .coverage import quick_pass,scan,diffpano_support

def search(n,seconds=2700,device='cuda'):
    key=layout_key('random',n);target=layout_path('random',n)
    progress_path=ROOT/'layouts'/('random_search_n'+str(n)+'.json')
    identity=geometry_identity()
    if target.exists():
        assert read(target)['geometry_identity']==identity;return True
    progress=read(progress_path) if progress_path.exists() else dict(
        n=n,base_seed=0,next_seed=0,last_attempted_seed=None,rejected_by_gate={},elapsed_seconds=0.,
        geometry_identity=identity,rng='torch CPU Generator, manual_seed(candidate_seed), rand((N,2),float32)',
        status='searching',selection='first candidate passing all frozen numerical DiffPano coverage gates')
    assert progress['geometry_identity']==identity,'Coverage protocol changed; cannot silently reuse rejection history'
    start=time.monotonic()
    while time.monotonic()-start<seconds:
        seed=progress['next_seed'];cams=build('random',n,seed);failed=None;geom=runtime=None
        for gate,count in [('A',8192),('B',65536)]:
            if not quick_pass(cams,count,device):failed=gate;break
        if failed is None:
            geom=scan(cams,device)
            if not geom['passed']:failed='C_D_poles_wrap'
        if failed is None:
            runtime=diffpano_support(cams,device)
            if not runtime['passed']:failed='E_actual_diffpano_support'
        progress.update(last_attempted_seed=seed,next_seed=seed+1,last_job=os.environ['SLURM_JOB_ID'])
        if failed:
            progress['rejected_by_gate'][failed]=progress['rejected_by_gate'].get(failed,0)+1
            progress['last_rejection']=dict(seed=seed,gate=failed)
        else:
            record=layout_record('random',n,seed)
            record.update(geometry_identity=identity,coverage=dict(geometric=geom,diffpano=dict(passed=True,runtime=runtime)),
                random_search=dict(accepted_seed=seed,attempt_index=seed,number_rejected=seed,
                    base_seed=0,proposal_rng=progress['rng'],rejected_by_gate=progress['rejected_by_gate'],
                    selected_using_images=False,spherediff_tests_required=False),
                created=now(),job=os.environ['SLURM_JOB_ID'])
            immutable(target,record);progress.update(status='accepted',accepted_seed=seed)
        elapsed=time.monotonic()-start
        current=dict(progress,elapsed_seconds=progress['elapsed_seconds']+elapsed)
        atomic(progress_path,current)
        if failed is None:
            emit('RANDOM_ACCEPTED',n=n,seed=seed,rejections=seed,layout_hash=record['angular_geometry_sha256'])
            return True
        if (seed+1)%100==0:emit('RANDOM_PROGRESS',n=n,next_seed=seed+1,rejections=progress['rejected_by_gate'],elapsed=elapsed)
    emit('RANDOM_BATCH_CHECKPOINT',n=n,next_seed=progress['next_seed'])
    return False
def main():
    p=argparse.ArgumentParser();p.add_argument('--camera-count',type=int,choices=COUNTS,required=True)
    p.add_argument('--seconds',type=int,default=2700);a=p.parse_args()
    import torch
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    assert_preserved()
    with lock('geometry-'+layout_key('random',a.camera_count)):search(a.camera_count,a.seconds)
if __name__=='__main__':main()
