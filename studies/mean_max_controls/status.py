"""Login-safe status inventory for the supplementary controls."""
import argparse,json
from studies.mean_max_controls.common import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    manifest=freeze_check();summary=dict(complete=[],failed=[],running=[],missing=[],invalid_complete=[])
    for row in manifest['rows']:
        path=folder(row)/'status.json'
        if complete(row,verify=args.verify):state='complete'
        elif not path.exists():state='missing'
        else:
            state=read(path)['state']
            if state=='complete':state='invalid_complete'
        summary[state].append(row['index'])
    print(json.dumps(dict(total=len(manifest['rows']),counts={key:len(value) for key,value in summary.items()},indices=summary),indent=2))


if __name__=='__main__':main()
