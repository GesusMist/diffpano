"""Report current FLUX scene-sweep states, including Slurm failures before Python startup."""
import collections,subprocess
from studies.gradient_blending.common import *

def main():
    assert BACKEND=='flux' and SUITE=='flux-scenes20'
    submission=read(OUT/'submission.json') if (OUT/'submission.json').exists() else {}
    accounting=''
    if submission:
        accounting=subprocess.check_output(['sacct','-j',submission['job'],'-n','-P','--format=JobID,State,ExitCode'],universal_newlines=True)
    states={line.split('|')[0]:line.split('|')[1] for line in accounting.splitlines() if '|' in line}
    planned={(r['prompt'],r['mode']):r['index'] for r in scene_rows()};rows=[]
    failed=('FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL')
    for prompt in PROMPTS:
        for mode in MODES:
            folder=OUT/'cases'/prompt/mode;path=folder/'status.json'
            status=read(path) if path.exists() else dict(state='not_submitted')
            row=dict(prompt=prompt,mode=mode,recorded_state=status['state'],state=status['state'],job=status.get('job'))
            index=planned.get((prompt,mode))
            if index is not None and submission:
                scheduler_state=states.get(submission['job']+'_'+str(index))
                row['slurm_state']=scheduler_state
                if status['state']!='complete' and scheduler_state:
                    if scheduler_state.startswith(failed):row['state']='failed'
                    elif scheduler_state=='RUNNING':row['state']='running'
                    elif scheduler_state=='PENDING':row['state']='queued'
                    elif scheduler_state=='COMPLETED':row['state']='missing_completion_record'
            progress=folder/'progress.json'
            if row['state']=='running' and progress.exists():
                current=read(progress);row['progress']={k:current[k] for k in ('step','total','elapsed_seconds')}
            rows.append(row)
    queue=subprocess.check_output(['squeue','-u','shig','-h','-o','%i|%j|%T|%M|%E'],universal_newlines=True)
    record=dict(rows=rows,counts=dict(collections.Counter(r['state'] for r in rows)),accounting=accounting,
        queue=queue,report_complete=(OUT/'completion.json').exists())
    write(OUT/'progress.json',record)
    lines=['FLUX gradient blending: 20 scene prompts, 60 logical outputs; 17 new prompts, 34 new trajectories.',str(record['counts']),'']
    for row in rows:
        lines.append('%s/%s: %s%s'%(row['prompt'],row['mode'],row['state'],
            ' (%d/%d)'%(row['progress']['step'],row['progress']['total']) if 'progress' in row else ''))
    lines+=['','Slurm queue:',queue]
    (OUT/'progress-report.txt').write_text('\n'.join(lines)+'\n')
    print(record['counts']);print(queue)
if __name__=='__main__':main()
