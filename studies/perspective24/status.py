"""Scene/line-associated index; deliberately no standalone metrics."""
from studies.perspective24.common import *
def refresh():
    m=read(OUT/'manifest.json');index=[]
    for row in m['rows']:
        valid=complete(row);d=folder(row)
        record=dict(index=row['index'],scene=row['prompt'],source_group=row['source_group'],prompt_path=row['prompt_path'],
            direction_label=row['direction_label'],line_index=row['line_index'],line_text=row['line_text'],line_sha256=row['line_sha256'],
            valid=valid,image=str(d/'final.png') if valid else None,metadata=str(d/'metadata.json') if valid else None)
        if valid:record.update(image_sha256=sha(d/'final.png'),metadata_sha256=sha(d/'metadata.json'))
        index.append(record)
    write(OUT/'index.json',dict(artifact_kind='standalone_perspective',expected=120,valid=sum(x['valid'] for x in index),rows=index))
    result=status();write(OUT/'completion.json',result);print(json.dumps(result,indent=2),flush=True);return result
if __name__=='__main__':refresh()
