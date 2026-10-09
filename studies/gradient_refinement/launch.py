"""Dry-run by default. Only exact missing slots; max two generation allocations."""
import argparse,fcntl,time
from studies.gradient_refinement.common import *
from studies.gradient_refinement.status import inventory,active_jobs

def submit(args):
    raw=subprocess.check_output(['sbatch','--parsable']+args,cwd=str(ROOT),universal_newlines=True).strip()
    job=raw.split(';')[0]
    if not job.isdigit():raise RuntimeError(raw)
    return job

def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');p.add_argument('--retry',action='store_true');p.add_argument('--representative',action='store_true');p.add_argument('--after',default=None);a=p.parse_args()
    lock=(OUT/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(OUT/'manifest.json');status=inventory();allowed={'missing'}|({'failed','interrupted','submitted_inactive'} if a.retry else set())
    indices=[r['index'] for r in status['cases'] if r['state'] in allowed]
    if a.representative:indices=[i for i in indices if i==47] # ruins/coarse/max/f=.1, an exact production slot
    print(json.dumps(dict(counts=status['counts'],submit_indices=indices,missing=len(indices)),indent=2))
    if not a.submit or not indices:return
    assert manifest['source_hashes']==source_hashes(),'Manifest differs from implementation'
    for gate in ('validation.json','gpu-smoke.json'):
        m=read(OUT/gate);assert m['passed'] and m['source_hashes']==source_hashes(),'Invalid/stale '+gate
    active=[q for q in active_jobs() if q['name']=='gradref-gen']
    if active:
        # A pending sweep may depend on one active representative. Never launch
        # another independent array while generation is already scheduled.
        assert a.after and len(active)==1 and active[0]['job']==a.after,'Generation already active; avoid duplicate arrays'
    args=['--job-name=gradref-gen']
    array=not a.representative
    if array:args+=['--array='+','.join(map(str,indices))+'%2']
    if a.after:args+=['--dependency=afterok:'+a.after]
    args+=['studies/gradient_refinement/gpu.slurm','studies.gradient_refinement.run']
    if a.representative:args+=['--index','47']
    if a.retry:args+=['--retry']
    job=submit(args)
    write(OUT/'submissions'/('%d-%s.json'%(int(time.time()),job)),dict(job=job,array=array,indices=indices,command=args,dependency=a.after,source_hashes=source_hashes()))
    print('SUBMITTED',job,indices,flush=True)
    if array:
        metrics=submit(['--dependency=afterok:'+job,'studies/gradient_refinement/evaluate.slurm'])
        partial=submit(['--dependency=afterany:'+job,'studies/gradient_refinement/finish.slurm'])
        final=submit(['--dependency=afterok:'+metrics+':'+partial,'studies/gradient_refinement/finish.slurm'])
        write(OUT/'dependent-jobs.json',dict(generation=job,evaluation=metrics,partial_report=partial,completion_report=final))
        print('DEPENDENT_JOBS',metrics,partial,final,flush=True)
if __name__=='__main__':main()
