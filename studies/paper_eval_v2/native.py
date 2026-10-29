"""Two fresh scene/direction-seeded FLUX20 banks using the audited native sampler."""
import argparse
import copy
import os
from pathlib import Path
import tarfile
import time
import traceback
from .common import ROOT,RUN,read,write,sha,digest,freeze,lock,now


def prepare(run=RUN):
    from studies.perspective24.common import definitions,sources,prompt_record,prompt_names
    from studies.perspective24.configuration import semantic,WEIGHTS,BASELINE
    rows=[];seeds=set()
    source=sources()
    for name in ('native.py','common.py','native.slurm'):
        path=ROOT/'studies/paper_eval_v2'/name;source[str(path.relative_to(ROOT))]=sha(path)
    for bank in ('A','B'):
        for original in definitions():
            row=copy.deepcopy(original)
            seed_identity=dict(namespace='paper_eval_v2/native_reference_seed/v1',bank=bank,
                prompt_identity=row['prompt_relative_path'],prompt_sha256=row['effective_prompt_sha256'],
                direction_index=row['line_index'])
            seed=int(digest(seed_identity)[:16],16)%(2**31-1)
            if seed in seeds:raise ValueError('Native seed collision')
            seeds.add(seed)
            row.update(index=len(rows),bank=bank,seed=seed,seed_identity=seed_identity,
                case_id='native/'+bank+'/'+row['case_id'])
            settings=semantic(row)
            settings['initialization']['seed']=seed
            row.update(semantic=settings,semantic_sha256=digest(settings),source_sha256=digest(source))
            rows.append(row)
    assert len(rows)==240 and len(seeds)==240
    output=Path(run)/'native'
    output.mkdir(exist_ok=True)
    manifest=dict(schema_version=2,artifact_kind='standalone_perspective',expected=240,rows=rows,
        prompts={name:prompt_record(name) for name in prompt_names()},source_hashes=source,
        input_artifacts={str(path):sha(path) for path in (WEIGHTS,BASELINE)},
        expected_counts=dict(denoiser=20,encode=0,decode=1,initialize=1),
        banks=dict(A='primary',B='reference sampling sensitivity'),
        cohort='development24',derived_cohort='development15',
        seed_policy='SHA256 of bank/full prompt identity/prompt bytes/direction, reduced modulo2**31-1, all240 unique',
        final_decode_only=True,panorama_processing=False,model_settings='same verified native FLUX20 loader/checkpoint/conditioning as perspective24',
        generation_groups=[list(range(start,min(start+30,240))) for start in range(0,240,30)])
    freeze(output/'manifest.json',manifest)
    freeze(Path(run)/'native_reference_manifest.json',manifest)
    freeze(Path(run)/'reference_seed_map.json',[{key:row[key] for key in ('case_id','bank','seed','seed_identity')} for row in rows])
    archive=output/'scientific-source.tar.gz'
    if not archive.exists():
        with tarfile.open(str(archive),'w:gz') as stream:
            for name in source:stream.add(str(ROOT/name),arcname=name)
    write(output/'freeze.json',dict(manifest_sha256=sha(output/'manifest.json'),source_archive_sha256=sha(archive),source_sha256=digest(source)))
    return manifest


def frozen(run):
    output=Path(run)/'native';manifest=read(output/'manifest.json');record=read(output/'freeze.json')
    assert sha(output/'manifest.json')==record['manifest_sha256']
    for path,expected in manifest['source_hashes'].items():
        assert sha(ROOT/path)==expected,'Frozen generation source changed: '+path
    for path,expected in manifest['input_artifacts'].items():assert sha(path)==expected,path
    return manifest


def complete(run,row):
    directory=Path(run)/'native/cases'/('%03d'%row['index'])
    try:
        record=read(directory/'metadata.json')
        if record['semantic_sha256']!=row['semantic_sha256'] or record['source_sha256']!=row['source_sha256']:return False
        if record['counts']!=dict(denoiser=20,encode=0,decode=1,initialize=1):return False
        if record['initialization']['seed']!=row['seed'] or record['guided_steps']!=20:return False
        if record['panorama_trajectory'] or record['fusion_stages'] or record['poisson_solves']:return False
        return all(sha(directory/name)==value for name,value in record['artifacts'].items())
    except (OSError,KeyError,ValueError):return False


def execute(run,group):
    assert os.environ.get('SLURM_JOB_ID'),'Use a finite generation allocation'
    manifest=frozen(run)
    rows=[manifest['rows'][i] for i in manifest['generation_groups'][group]]
    from studies.perspective24.runtime import load_backend,generate
    pending=[row for row in rows if not complete(run,row)]
    if not pending:return
    # The original audited loader can share immutable embeddings and model weights;
    # sample() constructs a fresh generator and resets the schedule for EACH image.
    backend,encoded,setup=load_backend(pending)
    for row in pending:
        directory=Path(run)/'native/cases'/('%03d'%row['index']);directory.mkdir(parents=True,exist_ok=True)
        with lock(directory/'case.lock'):
            if complete(run,row):continue
            status=directory/'status.json'
            if status.exists():
                previous=read(status)
                old_job=str(previous.get('job',''))
                if previous.get('state')=='running' and old_job and old_job!=os.environ['SLURM_JOB_ID']:
                    from .common import command
                    if command(['squeue','-h','-j',old_job,'-o','%i']):raise RuntimeError('Case still owned by active job')
                history=directory/('attempt-'+str(time.time_ns()));history.mkdir()
                for name in ('status.json','metadata.json','config.json','progress.json','final.png'):
                    path=directory/name
                    if path.exists():path.replace(history/name)
            write(status,dict(state='running',case_id=row['case_id'],job=os.environ['SLURM_JOB_ID'],started_utc=now()))
            try:
                record=generate(backend,row,encoded.pop(row['index']),directory,setup)
                frozen(run)
                record.update(case=row,semantic_sha256=row['semantic_sha256'],source_sha256=row['source_sha256'],
                    source_hashes=manifest['source_hashes'],job=os.environ['SLURM_JOB_ID'],completed_utc=now())
                write(directory/'metadata.json',record)
                assert complete(run,row)
                write(status,dict(state='complete',case_id=row['case_id'],job=os.environ['SLURM_JOB_ID'],
                                  metadata_sha256=sha(directory/'metadata.json'),image_sha256=sha(directory/'final.png')))
                print('NATIVE_COMPLETE',row['case_id'],row['seed'],record['generation_seconds'],flush=True)
            except BaseException:
                write(status,dict(state='failed',job=os.environ['SLURM_JOB_ID'],case_id=row['case_id'],error=traceback.format_exc()))
                raise


def launch(run=RUN):
    from .resources import submit
    manifest=frozen(run)
    groups=[i for i,indices in enumerate(manifest['generation_groups']) if not all(complete(run,manifest['rows'][j]) for j in indices)]
    if not groups:return dict(state='complete',images=240)
    return submit(run,'native-v1',ROOT/'studies/paper_eval_v2/native.slurm',role='generation',
        options=('--array='+','.join(map(str,groups))+'%2',),script_args=('--run',str(run)),reserve_su=len(groups)*146)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run','launch'))
    parser.add_argument('--run',type=Path,default=RUN);parser.add_argument('--group',type=int)
    args=parser.parse_args()
    if args.action=='prepare':print('NATIVE_PREPARED',prepare(args.run)['expected'])
    elif args.action=='launch':print(launch(args.run))
    else:execute(args.run,args.group if args.group is not None else int(os.environ['SLURM_ARRAY_TASK_ID']))

if __name__=='__main__':main()
