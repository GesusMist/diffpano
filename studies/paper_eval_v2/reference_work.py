"""Adapt frozen source manifests and extract references under the shared protocol."""
import argparse
from .common import *


def prepare(run):
    dataset=read(run/'real_dataset_manifest.json')
    rows=dataset['rows']
    assert dataset['state']=='complete' and len(rows)==24
    adapted=dict(dataset_label=dataset['dataset_label'],source_split=dataset['source_split'],
        source_manifest_path=str(run/'real_dataset_manifest.json'),source_manifest_sha256=sha(run/'real_dataset_manifest.json'),
        category_counts=dataset['category_counts'],records=[dict(reference_id=row['scene_id'],
            prompt_id=row['prompt_id'],image_path=row['image_path'],image_sha256=row['image_sha256']) for row in rows])
    path=run/'group2/real_reference_inputs.json'
    freeze(path,adapted)
    return path


def execute(run):
    from .cli import verify_protocol
    from .native_index import refresh
    from .distribution import extract_real_reference,extract_native_reference
    verify_protocol(run)
    real=extract_real_reference(run,prepare(run))
    counts=refresh(run)
    native={}
    for bank in ('A','B'):
        if counts[bank]==120:
            native[bank]=extract_native_reference(run,run/('native/reference-'+bank+'.json'))['complete']
        else:native[bank]=False
    verify_protocol(run)
    result=dict(real_complete=real['complete'],real_count=len(real['records']),native_complete=native,native_counts=counts)
    update_component(run,'reference_features','complete' if real['complete'] and all(native.values()) else 'partial',**result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=RUN);p.add_argument('--prepare',action='store_true');a=p.parse_args()
    print(prepare(a.run) if a.prepare else execute(a.run))

if __name__=='__main__':main()
