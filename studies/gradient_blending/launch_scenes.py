"""Submit only the 34 requested new FLUX trajectories, plus dependent reporting."""
import subprocess
from studies.gradient_blending.common import *
from studies.gradient_blending.launch_extension import submit

def main():
    assert BACKEND=='flux' and SUITE=='flux-scenes20'
    preserved();preserve_prior_results()
    gate=read(OUT/'validation.json');manifest=read(OUT/'manifest.json')
    assert gate['passed'] and gate['source_hashes']==source_hashes() and gate['core_hashes']==core_hashes()
    assert gate['baseline_sha256']==sha(OUT/'baseline.json') and gate['manifest_sha256']==sha(OUT/'manifest.json')
    assert manifest['rows']==scene_rows() and len(manifest['rows'])==34
    for name in ('validation.json','gpu-smoke.json','offline/replay.json'):
        shared=read(GATES/name);assert shared['passed'] and shared['core_hashes']==core_hashes()
    active=subprocess.check_output(['squeue','-u','shig','-h','-o','%i|%j'],universal_newlines=True)
    assert not any(line.split('|')[-1] in ('grad-pilot','grad-extension','grad-flux-scenes') for line in active.splitlines()),'Gradient generation already active'
    for prompt in PROMPTS:
        for mode in (MODES if prompt in PILOT_PROMPTS else ('rgb',)):
            folder=OUT/'cases'/prompt/mode;status=read(folder/'status.json')
            assert status['state']=='complete' and sha(folder/'final.png')==read(folder/'metadata.json')['artifacts']['final.png']
    record_path=OUT/'submission.json'
    assert not record_path.exists(),'A prior sweep submission exists; inspect before resubmitting'
    indices=[]
    for row in scene_rows():
        folder=OUT/'cases'/row['prompt']/row['mode'];path=folder/'status.json'
        if path.exists() and read(path)['state']=='complete':
            assert sha(folder/'final.png')==read(folder/'metadata.json')['artifacts']['final.png'];continue
        indices.append(row['index'])
    if not indices:print('All 34 new trajectories already complete');return
    array=submit(['--array='+','.join(map(str,indices))+'%2','studies/gradient_blending/scenes.slurm'])
    record=dict(job=array,indices=indices,max_concurrent_generations=2,source_hashes=source_hashes(),
        validation_job=gate['job'],new_gradient_runs=len(indices),new_prompts=NEW_FLUX_PROMPTS,
        prompts=PROMPTS,logical_outputs=60,reused_rgb_controls=17,reused_pilot_outputs=9,followups={})
    write(record_path,record)
    for index in indices:
        row=scene_rows()[index];folder=OUT/'cases'/row['prompt']/row['mode']
        write(folder/'status.json',dict(state='submitted',job=array+'_'+str(index),**row))
    export='--export=ALL,DIFFPANO_GRADIENT_BACKEND=flux,DIFFPANO_GRADIENT_SUITE=flux-scenes20'
    evaluation=submit(['--dependency=afterok:'+array,export,'--job-name=grad-flux-scenes-eval',
        '--time=01:30:00','studies/gradient_blending/evaluate.slurm'])
    record['followups']['evaluation']=evaluation;write(record_path,record)
    figures=submit(['--dependency=afterok:'+evaluation,export,'--job-name=grad-flux-scenes-figures',
        '--time=01:00:00','studies/gradient_blending/figures.slurm'])
    record['followups']['figures_report']=figures;write(record_path,record)
    finish=submit(['--dependency=afterok:'+figures,'studies/gradient_blending/scenes_finish.slurm'])
    record['followups']['completion_audit']=finish;write(record_path,record)
    for name,dependency in [('generation_status',array),('final_status',finish)]:
        job=submit(['--dependency=afterany:'+dependency,'studies/gradient_blending/scenes_status.slurm'])
        record['followups'][name]=job;write(record_path,record)
    from studies.gradient_blending.scenes_status import main as status
    status();print('FLUX_SCENES_SUBMITTED',array,len(indices),record['followups'],flush=True)
if __name__=='__main__':main()
