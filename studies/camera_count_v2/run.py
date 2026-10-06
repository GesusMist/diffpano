"""Fresh-process scientific worker. A matching PNG/config/status triple completes it."""
import argparse
import platform
import traceback
from .common import *
def resolve(args):
    if args.index is not None:return rows()[args.index]
    if args.family=='spherediff_camera_override' and args.strategy!='old':
        raise ValueError('SphereDiff Fibonacci/random is not authorized; old/rings only')
    found=[r for r in rows() if (r['family'],r['strategy'],r['num_cameras'],r['backend'],r['prompt'])==
           (args.family,args.strategy,args.camera_count,args.backend,args.prompt)]
    if len(found)!=1:raise ValueError('Exactly one authorized ERP row required')
    return found[0]
def execute(row):
    import torch
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    validation_gate();layout=load_layout(row);pf=preflight_gate(row);before=source_hashes()
    with lock('sample-'+str(row['index'])):
        if complete(row):emit('REUSE_COMPLETE',row=row);return
        attempt=str(os.environ['SLURM_JOB_ID'])
        atomic(ROOT/'status'/(str(row['index'])+'.json'),dict(state='running',row=row,attempt=attempt,time=now()))
        ledger('STARTED',row=row,attempt=attempt)
        try:
            runner='diffpano_runner' if row['family']=='diffpano' else 'spherediff_runner'
            module=__import__('studies.camera_count_v2.'+runner,fromlist=['generate'])
            with effective_prompt(row['prompt']) as (prompt,path):
                image,record=module.generate(row,path)
            assert source_hashes()==before;assert_preserved()
            record.update(version=VERSION,row=row,prompt=prompt,source_hashes=before,
                git_commit=shell_command(['git','rev-parse','HEAD']),git_dirty_status=shell_command(['git','status','--short']),
                camera=dict(strategy=row['strategy'],num_cameras=row['num_cameras'],
                    layout_reference=str(layout_path(row['strategy'],row['num_cameras'])),
                    layout_sha256=sha(layout_path(row['strategy'],row['num_cameras'])),
                    angular_geometry_sha256=layout['angular_geometry_sha256'],geometry_identity=geometry_identity(),
                    ring_counts=layout['ring_counts'],fibonacci_phase=layout['fibonacci_phase'],
                    accepted_seed=layout['accepted_seed'],fov=[80.,80.],fixed_during_trajectory=True,
                    runtime_pose_sha256=record.get('camera_override',{}).get('runtime_bf16_sha256',
                        record.get('runtime_audit',{}).get('camera_sha256')),
                    stable_id_policy='stored ordered IDs in shared layout; accepted Random seed encoded',
                    random_rejections=layout.get('random_search',{}).get('number_rejected')),
                coverage=dict(passed=True,layout_sha256=sha(layout_path(row['strategy'],row['num_cameras'])),
                    geometry_identity=geometry_identity(),family_gate='spherediff' if row['family']=='spherediff_camera_override' else 'diffpano'),
                preflight=dict(path=str(preflight_path(row)),sha256=sha(preflight_path(row))))
            record['execution'].update(job_id=attempt,node=platform.node(),gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version())
            publish(row,image,record)
            ledger('COMPLETE',row=row,job=attempt,image_sha256=record['output']['sha256'])
        except BaseException:
            error=traceback.format_exc()
            atomic(ROOT/'status'/(str(row['index'])+'.json'),dict(state='failed',row=row,attempt=attempt,time=now(),error=error))
            ledger('FAILED',row=row,job=attempt,error=error);raise
def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int)
    p.add_argument('--family',choices=('diffpano','spherediff_camera_override'))
    p.add_argument('--strategy',choices=STRATEGIES);p.add_argument('--camera-count',type=int,choices=COUNTS)
    p.add_argument('--backend',choices=BACKENDS);p.add_argument('--prompt',choices=PROMPTS)
    execute(resolve(p.parse_args()))
if __name__=='__main__':main()
