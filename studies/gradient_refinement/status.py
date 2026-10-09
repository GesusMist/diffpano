import collections,subprocess
from studies.gradient_refinement.common import *

def active_jobs():
    result=subprocess.run(['squeue','-u',os.environ.get('USER','shig'),'-h','-r','-o','%i|%j|%T'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    if result.returncode:raise RuntimeError('Cannot inspect Slurm queue: '+result.stderr)
    return [dict(job=x.split('|')[0],name=x.split('|')[1],state=x.split('|')[2]) for x in result.stdout.splitlines() if x]

def inventory(verify=True):
    manifest=read(OUT/'manifest.json');queue=active_jobs();byjob={x['job']:x for x in queue};records=[]
    submitted={}
    for history in sorted((OUT/'submissions').glob('*.json')):
        m=read(history)
        for i in m.get('indices',[]):submitted[i]=m['job']+'_'+str(i) if m.get('array') else m['job']
    for row in manifest['rows']:
        path=folder(row);status=read(path/'status.json') if (path/'status.json').exists() else {}
        if case_complete(row,verify):state='reused' if status['state']=='reused' else 'complete'
        else:
            job=submitted.get(row['index'],status.get('job'))
            active=byjob.get(job)
            if active:state=active['state'].lower()
            elif status.get('state')=='failed':state='failed'
            elif status.get('state')=='running':state='interrupted'
            elif row['index'] in submitted:state='submitted_inactive'
            else:state='missing'
        records.append(dict(index=row['index'],key=row['key'],state=state,job=submitted.get(row['index'],status.get('job'))))
    return dict(counts=dict(collections.Counter(r['state'] for r in records)),cases=records,active_jobs=[q for q in queue if q['name'].startswith('gradref')])
if __name__=='__main__':
    result=inventory();write(OUT/'latest-status.json',result);print(json.dumps(result,indent=2))
