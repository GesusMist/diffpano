"""Read statuses without loading models or modifying historical results."""
import collections,subprocess
from studies.gradient_blending.common import *

def main():
    cases=[]
    for backend in BACKENDS:
        base=BASE_OUT if backend=='sana' else BASE_OUT/backend
        for prompt in PROMPTS:
            for mode in MODES:
                folder=base/'cases'/prompt/mode;path=folder/'status.json'
                status=read(path) if path.exists() else dict(state='not_submitted')
                row=dict(status);row.update(backend=backend,prompt=prompt,mode=mode)
                progress=folder/'progress.json'
                if status['state']=='running' and progress.exists():
                    row['progress']={k:read(progress)[k] for k in ('step','total','elapsed_seconds')}
                cases.append(row)
    active=subprocess.check_output(['squeue','-u','shig','-h','-o','%i|%j|%T|%M|%R'],universal_newlines=True)
    record=dict(cases=cases,counts=dict(collections.Counter(r['state'] for r in cases)),active_queue=active,
        final_report_available=(BASE_OUT/'extension-completion.json').is_file())
    write(BASE_OUT/'extension-progress.json',record)
    lines=['Gradient study: 36 logical cases across SANA, FLUX, PixelDiT and SD3.5',str(record['counts']),'']
    for row in cases:
        lines.append('%s/%s/%s: %s%s'%(row['backend'],row['prompt'],row['mode'],row['state'],
            ' (%d/%d)'%(row['progress']['step'],row['progress']['total']) if 'progress' in row else ''))
    lines+=['','Slurm queue:',active]
    (BASE_OUT/'extension-progress-report.txt').write_text('\n'.join(lines)+'\n')
    print(record['counts']);print(active)
if __name__=='__main__':main()
