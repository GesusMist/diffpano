"""Execute one immutable grouped A task or one missing B target."""
import argparse
import os
from studies.followup24.common import ROOT,OUT_A,OUT_B
from studies.perspective24.common import read,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--stage',choices=('A','B-profile','B'),required=True);args=parser.parse_args()
    plan=read(args.plan)
    assert all(sha(ROOT/p)==h for p,h in plan['sources'].items()),'Orchestration source changed after launch'
    assert sha(OUT_A/'manifest.json')==plan['A_manifest_sha256']
    assert sha(OUT_B/'manifest.json')==plan['B_manifest_sha256']
    index=int(os.environ['SLURM_ARRAY_TASK_ID'])
    if args.stage=='A':
        from studies.perspective24.run import run
        run(plan['A_groups'][index],retry=plan['retry'])
    else:
        if args.stage=='B':
            from studies.layout24.common import freeze_check,complete
            manifest=freeze_check()
            assert all(complete(manifest['rows'][i]) for i in plan['B_profile_requirements']),'A required actual-target profile failed; bulk generation remains blocked'
        from studies.layout24.run import run
        indices=plan['B_profiles'] if args.stage=='B-profile' else plan['B_bulk']
        run(indices[index],retry=plan['retry'])

if __name__=='__main__':main()
