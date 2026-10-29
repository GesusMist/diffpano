"""Initial finite launch after inventory and targeted verification; resumable keys."""
from .common import RUN,ROOT,read,write,sha,update_component
from .cli import prepare
from .launch import launch


def main():
    run=RUN
    audit=read(run/'inventory/provenance_audit.json')
    assert (run/'inventory/cohorts.json').exists()
    result=prepare(run)
    from .native import frozen
    native=frozen(run)
    assert len(native['rows'])==len({r['seed'] for r in native['rows']})==240
    assert all(r['seed']==r['semantic']['initialization']['seed'] and r['seed']==r['semantic']['case']['seed'] for r in native['rows'])
    assert all(r['semantic']['terminal_decode_only'] and not r['semantic']['panorama_processing'] for r in native['rows'])
    write(run/'native/seed_validation.json',dict(passed=True,unique_seeds=240,identities=240,semantic_seeds_match=True,manifest_sha256=sha(run/'native/manifest.json')))
    from .vlm import prepare as prepare_vlm
    prepare_vlm(run)
    for component in ('cpu','gpu','views','native'):
        print('SUBMITTED_COMPONENT',component,launch(run,component),flush=True)
    print('INITIAL_EVALUATION_STARTED',result,flush=True)

if __name__=='__main__':main()
