"""Incremental verified native-bank indexes without modifying generation source."""
from pathlib import Path
from .common import RUN,read,write,sha
from .native import complete


def refresh(run=RUN):
    run=Path(run);manifest=read(run/'native/manifest.json');result={}
    for bank in ('A','B'):
        rows=[]
        for row in manifest['rows']:
            if row['bank']!=bank:continue
            directory=run/'native/cases'/('%03d'%row['index'])
            valid=complete(run,row)
            rows.append(dict(bank=bank,prompt_id=row['prompt_relative_path']+'@'+row['effective_prompt_sha256'],
                direction_index=row['line_index']-1,case_id=row['case_id'],seed=row['seed'],
                image_path=str(directory/'final.png'),image_sha256=sha(directory/'final.png') if valid else None,
                state='complete' if valid else 'pending',metadata_path=str(directory/'metadata.json')))
        value=dict(bank=bank,records=rows,expected=120,complete_count=sum(r['state']=='complete' for r in rows),
                   generation_manifest_sha256=sha(run/'native/manifest.json'))
        write(run/('native/reference-'+bank+'.json'),value);result[bank]=value['complete_count']
    return result

if __name__=='__main__':print(refresh())
