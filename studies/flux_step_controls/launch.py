import argparse
from .common import *

def main(submit=False,revision_reason=None):
    ROOT.mkdir(parents=True,exist_ok=True)
    with lock('flux-step-global-submit'):
        if revision_reason and (ROOT/'manifest.json').exists():
            queue=shell_command(['squeue','-u','shig','-h','-o','%i|%j|%T'])
            assert not any('|fsctrl-' in l for l in queue.splitlines()),queue
            old=manifest();assert old['rows']==rows() and old['protected_sources']==protected_hashes()
            archive=ROOT/'manifests'/('manifest-'+sha(ROOT/'manifest.json')+'.json');archive.parent.mkdir(exist_ok=True)
            os.rename(ROOT/'manifest.json',archive)
            ledger('source_revision',reason=revision_reason,prior_manifest=str(archive),prior_sources=old['study_sources'])
        if not (ROOT/'manifest.json').exists():
            protected=protected_hashes();assert protected['historical']==HISTORICAL
            immutable(ROOT/'manifest.json',dict(study='FLUX step controls',created=now(),rows=rows(),
                prompts=inventory(),prompt_inventory_sha256=inventory_hash(),study_sources=source_hashes(),
                protected_sources=protected,git_commit=shell_command(['git','rev-parse','HEAD']),
                DIFFPANO_28_PROJECTION=DIFFPANO_28_PROJECTION,
                maximum_generation_gpus=4,evaluation_gpu_reservation=1,account=ACCOUNT))
            (ROOT/'README.md').write_text('''# FLUX step controls\n\nExactly 42 new seed-0 panoramas: original SphereDiff FLUX at 20 steps (21 prompts), and DiffPano FLUX ERP at 28 steps (21 prompts). DIFFPANO_28_PROJECTION = "erp". CEA-28 is absent by design.\n\nWithin each method only step count and output bookkeeping change. Both construct fresh complete schedules. Model sources remain pinned independently: ModelsLab for DiffPano, Black Forest Labs for SphereDiff. Equal-step comparisons are step-matched, not compute-matched.\n\nOne combined array has at most four generation GPUs; evaluator capacity is reserved. CPU validation and two bounded real-backend preflights gate production. Configs include full actual schedules, prompt hashes, source provenance, initial hashes, forward/VAE counts, input shapes, timing, GPU memory, and output checksum. A final PNG/config pair must match before reuse or evaluation.\n\nImmutable manifest: manifest.json. Slurm submission evidence: submissions.jsonl. Gates: validation/. Completion/failures: status/ and completion.json. No historical source or result is modified.\n''')
        assert_preserved(manifest())
        queue=shell_command(['squeue','-u','shig','-h','-o','%i|%j|%T|%b'])
        emit('QUEUE',text=queue)
        if any('|fsctrl-' in l for l in queue.splitlines()):emit('ALREADY_SUBMITTED');return
        if not revision_reason and (ROOT/'submissions.jsonl').exists() and 'generation_submitted' in (ROOT/'submissions.jsonl').read_text():
            raise RuntimeError('Prior generation submission exists; inspect completion/accounting before selective retry')
        if queue.strip():raise RuntimeError('Inspect other active jobs before reserving four generation GPUs')
        if not submit:emit('DRY_RUN',matrix=rows());return
        def send(stage,args):
            ledger('intent',stage=stage,command=args)
            reply=shell_command(['sbatch','--parsable']+args);job=reply.split(';')[0];assert job.isdigit(),reply
            ledger(stage+'_submitted',job=job,response=reply,args=args);return job
        cpu=send('validation',[str(STUDY/'validate.slurm')])
        pf=send('preflight',['--dependency=afterok:'+cpu,'--array=0-1%2',str(STUDY/'preflight.slurm')])
        gen=send('generation',['--dependency=afterok:'+pf,'--array=0-41%4',str(STUDY/'run.slurm')])
        report=send('report',['--dependency=afterany:'+gen,str(STUDY/'report.slurm')])
        emit('SUBMITTED',validation=cpu,preflight=pf,generation=gen,report=report)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');p.add_argument('--revision-reason')
    a=p.parse_args();main(a.submit,a.revision_reason)
