"""Gated finite PixelDiT/SD3.5 extension and automatic evaluation/figures/report dependencies."""
import subprocess
from studies.gradient_blending.common import *

def submit(args):
    output=subprocess.check_output(['sbatch','--parsable']+args,cwd=str(ROOT),universal_newlines=True).strip()
    return next(line.split(';')[0] for line in output.splitlines() if line.split(';')[0].isdigit())

def main():
    preserved();gate=read(BASE_OUT/'extension-validation.json')
    assert gate['passed'] and gate['source_hashes']==source_hashes() and gate['core_hashes']==core_hashes()
    for name in ('validation.json','gpu-smoke.json','offline/replay.json'):
        shared=read(GATES/name);assert shared['passed'] and shared['core_hashes']==core_hashes()
    assert read(GATES/'offline/replay.json')['pilot_lambda_color']==.1
    active=subprocess.check_output(['squeue','-u','shig','-h','-o','%i|%j'],universal_newlines=True)
    assert not any(line.split('|')[-1] in ('grad-pilot','grad-extension') for line in active.splitlines()),'Gradient generation already active'
    for backend in ('pixeldit','sd35'):
        root=BASE_OUT/backend;audit=read(root/'backend-validation.json')
        assert audit['passed'] and audit['source_hashes']==source_hashes()
        assert audit['baseline_sha256']==sha(root/'baseline.json')
        for prompt in PROMPTS:
            folder=root/'cases'/prompt/'rgb'
            assert read(folder/'status.json')['state']=='complete'
            assert sha(folder/'final.png')==read(folder/'metadata.json')['artifacts']['final.png']
    pending=[]
    for index in range(12):
        backend=('pixeldit','sd35')[index//6];local=index%6
        folder=BASE_OUT/backend/'cases'/PROMPTS[local//2]/MODES[1+local%2]
        if (folder/'status.json').exists() and read(folder/'status.json')['state']=='complete':
            assert sha(folder/'final.png')==read(folder/'metadata.json')['artifacts']['final.png'];continue
        pending.append(index)
    if not pending:print('All extension generations already complete; no duplicate submission');return
    manifest=BASE_OUT/'extension-submission.json'
    assert not manifest.exists(),'Prior extension submission exists; inspect it before resubmitting'
    array=submit(['--array='+','.join(map(str,pending))+'%2','studies/gradient_blending/extension.slurm'])
    record=dict(job=array,indices=pending,max_concurrent_generations=2,source_hashes=source_hashes(),
        validation_job=gate['job'],logical_case_count=36,new_logical_cases=18,new_gradient_runs=len(pending),
        reused_controls=[backend+'/'+prompt+'/rgb' for backend in ('pixeldit','sd35') for prompt in PROMPTS],
        backend_steps=dict(pixeldit=50,sd35=40),followups={})
    write(manifest,record)
    for index in pending:
        backend=('pixeldit','sd35')[index//6];local=index%6;prompt=PROMPTS[local//2];mode=MODES[1+local%2]
        write(BASE_OUT/backend/'cases'/prompt/mode/'status.json',dict(state='submitted',job=array+'_'+str(index),backend=backend,prompt=prompt,mode=mode))
    # Followups start after the whole array, so evaluation does not add GPUs during generation.
    figure_jobs=[]
    for backend in ('pixeldit','sd35'):
        evaluation=submit(['--dependency=afterok:'+array,'--export=ALL,DIFFPANO_GRADIENT_BACKEND='+backend,
            '--job-name=grad-eval-'+backend,'studies/gradient_blending/evaluate.slurm'])
        record['followups'][backend+'_evaluation']=evaluation;write(manifest,record)
        figures=submit(['--dependency=afterok:'+evaluation,'--export=ALL,DIFFPANO_GRADIENT_BACKEND='+backend,
            '--job-name=grad-figures-'+backend,'studies/gradient_blending/figures.slurm'])
        record['followups'][backend+'_figures_report']=figures;write(manifest,record);figure_jobs.append(figures)
    finish=submit(['--dependency=afterok:'+':'.join(figure_jobs),'studies/gradient_blending/extension_finish.slurm'])
    record['followups']['four_model_report']=finish;write(manifest,record)
    from studies.gradient_blending.extension_status import main as status
    status()
    print('EXTENSION_SUBMITTED',array,'indices',pending,'followups',record['followups'],flush=True)
if __name__=='__main__':main()
