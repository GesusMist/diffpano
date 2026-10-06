import concurrent.futures
from .common import *

def main():
    assert os.environ.get('SLURM_JOB_ID')
    m=read(REPO/'outputs/flux-step-controls-seed0/checkpoint_comparison.json');sources=m['sources']
    shared=sorted(set(sources['diffpano']['files'])&set(sources['spherediff']['files']))
    def audit(name):
        out={}
        for method,s in sources.items():
            p=Path(s['snapshot'])/name;h=sha(p)
            out[method]=dict(path=str(p),sha256=h,bytes=p.stat().st_size,
                matches_cache_address=(p.resolve().name==h) if p.suffix=='.safetensors' else None)
        return dict(name=name,methods=out,identical=out['diffpano']['sha256']==out['spherediff']['sha256'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:result=list(pool.map(audit,shared))
    record=dict(time=now(),job=os.environ['SLURM_JOB_ID'],files=result,all_shared_files_identical=all(r['identical'] for r in result),
        note='Independent full SHA256 of every shared diffusers pipeline JSON and weight shard in both pinned snapshots. Extra raw BFL-format weights are not used by these diffusers runtimes.')
    atomic(ROOT/'checkpoint_weight_audit.json',record);print('CHECKPOINT_AUDIT',record['all_shared_files_identical'],len(result),flush=True)
if __name__=='__main__':main()
