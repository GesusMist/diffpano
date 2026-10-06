"""Nonblocking bounded evaluation submission. Never submits generation."""
import argparse
import fcntl
import subprocess
from .common import *

def queue():
    out=subprocess.check_output(['squeue','-u','shig','-h','-o','%i|%j|%T|%b'],universal_newlines=True)
    return [dict(zip(('job','name','state','gres'),line.split('|'))) for line in out.splitlines()]

def ledger(record):
    path=ROOT/'submissions.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(dict(time=now(),**record),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

def submit(follow_generation=None):
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'.submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        frozen=read(ROOT/'validation/frozen.json')
        assert frozen['source_hashes']==source_hashes(),'Evaluator source changed since validated freeze'
        q=queue();eval_gpu=[r for r in q if r['name'].startswith('pmetrics-') and 'gpu' in r['gres']]
        existing_follow=[r for r in q if r['name']=='pmetrics-followup']
        if existing_follow or (not follow_generation and eval_gpu):
            print(json.dumps(dict(status='already_submitted',jobs=existing_follow or eval_gpu),indent=2));return
        # Known combined production array is capped at four. Do not overlap another evaluation GPU.
        unrelated=[r for r in q if 'gpu' in r['gres'] and not r['name'].startswith(('pmetrics-','fsctrl-'))]
        deps={r['job'].split('_')[0] for r in eval_gpu+unrelated}
        if follow_generation:
            assert follow_generation.isdigit();deps.add(follow_generation)
            previous=[json.loads(l) for l in (ROOT/'submissions.jsonl').read_text().splitlines()] if (ROOT/'submissions.jsonl').exists() else []
            matches=[r for r in previous if r.get('role')=='followup' and r.get('generation_job')==follow_generation and r.get('job')]
            if matches:print(json.dumps(dict(status='followup_already_submitted',records=matches),indent=2));return
        args=['sbatch','--parsable','--job-name='+('pmetrics-followup' if follow_generation else 'pmetrics-update')]
        if deps:args+=['--dependency=afterany:'+':'.join(sorted(deps))]
        args += [str(REPO/'studies/panorama_metrics/update_gpu.slurm')]
        ledger(dict(event='intent',command=args,generation_job=follow_generation))
        response=subprocess.check_output(args,universal_newlines=True).strip();job=response.split(';')[0];assert job.isdigit(),response
        record=dict(event='submitted',role='followup' if follow_generation else 'update',job=job,response=response,
            generation_job=follow_generation,dependencies=sorted(deps),command=args,frozen_sha256=sha(ROOT/'validation/frozen.json'))
        ledger(record);print(json.dumps(record,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--follow-generation');a=p.parse_args();submit(a.follow_generation)
